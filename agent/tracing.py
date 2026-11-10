"""Our own Langfuse spans for the graph: one span per node, one generation per model call.

Task #6 replaced Lesson 11's `langfuse.langchain.CallbackHandler`. The handler turned
every LangChain runnable into an observation (a structured-output call alone left
`RunnableSequence`, `RunnableParallel<raw>`, `RunnableAssign<...>`, `RunnableLambda`
and a parser span, nine in all), so the Rachel trace had 93 observations. Its
`langsmith:hidden` tag only marks a run DEBUG; the observation is still stored and still
counts as a Langfuse unit. Its node spans also never became the current OpenTelemetry
context inside the node, so the `tool:<name>` spans that `agent/mcp_tools.py` opens
attached to the root `turn` instead of the node that made the call.

The tree a reader sees now:

    turn                         (agent, agent/run.py)
      classify                   (span, one per node run, named after the node)
        model:planner            (generation: model id, tokens, cost)
      scope
      gather
        model:assistant
      tools
        tool:get_policy          (tool, agent/mcp_tools.py)
          server:get_policy      (span, mcp_server/server.py)
      answer

`node()` wraps a node function in a span that is current while the node runs, so the
spans and generations the node opens nest under it. `Traced` wraps a model runnable
and records one generation per call, with the model id and token usage that Langfuse
prices into cost. Node spans carry a compact output (what the node changed), not the
whole graph state.
"""

import inspect
import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolMessage
from langfuse import get_client
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel

TEXT_CHARS = 800  # per message in a node span's output; generations keep full text


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def message_view(message: BaseMessage, chars: int | None = TEXT_CHARS) -> dict:
    """One message as a reader wants to see it in a trace."""
    if isinstance(message, RemoveMessage):
        return {"removed": message.id}
    text = _text(message)
    if chars is not None and len(text) > chars:
        text = text[:chars] + f"... [+{len(text) - chars} chars]"
    view: dict[str, Any] = {"role": message.type, "text": text}
    if isinstance(message, ToolMessage):
        view.update(tool=message.name, status=message.status)
    if isinstance(message, AIMessage) and message.tool_calls:
        view["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
    return view


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, BaseMessage):
        return message_view(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def update_view(update: dict | None) -> dict | None:
    """What a node changed, for its span output."""
    if not update:
        return update
    return {key: _plain(value) for key, value in update.items()}


def node(name: str, fn):
    """`fn` wrapped in a span named `name`. LangGraph reads a node's signature to decide
    whether to pass `config`, so the wrapper keeps the same parameters.

    An `interrupt()` inside the node raises GraphInterrupt, which is how LangGraph
    pauses, not a failure: the span ends with the interrupt payload as output and the
    exception is raised again outside it, so Langfuse does not mark the span as an
    error."""
    takes_config = "config" in inspect.signature(fn).parameters

    async def run(state, config=None):
        langfuse = get_client()
        paused = None
        with langfuse.start_as_current_observation(as_type="span", name=name) as span:
            try:
                result = fn(state, config) if takes_config else fn(state)
                if inspect.isawaitable(result):
                    result = await result
            except GraphInterrupt as exc:
                paused = exc
                span.update(output={"interrupt": _plain([i.value for i in exc.args[0]])
                                    if exc.args else None})
            else:
                span.update(output=update_view(result))
        if paused is not None:
            raise paused
        return result

    if takes_config:
        async def wrapped(state, config):
            return await run(state, config)
    else:
        async def wrapped(state):
            return await run(state)
    wrapped.__name__ = name
    return wrapped


def _generation_output(result: Any) -> Any:
    if isinstance(result, dict) and "raw" in result:  # with_structured_output(include_raw)
        return {"parsed": _plain(result.get("parsed")),
                "parsing_error": str(result["parsing_error"]) if result.get("parsing_error")
                else None}
    if isinstance(result, BaseMessage):
        return message_view(result, chars=None)
    return _plain(result)


def _usage(result: Any) -> dict[str, int]:
    message = result.get("raw") if isinstance(result, dict) else result
    usage = getattr(message, "usage_metadata", None) or {}
    return {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0)}


class Traced:
    """A model runnable whose every `ainvoke` is one Langfuse generation named
    `model:<role>`, with the model id, the messages, the output and token usage.
    Langfuse computes cost from the model id and usage when a price is registered for
    that id (scripts/register_model_prices.py)."""

    def __init__(self, runnable, role: str, model_name: str | None):
        self.runnable = runnable
        self.role = role
        self.model_name = model_name

    async def ainvoke(self, messages: list) -> Any:
        with get_client().start_as_current_observation(
                as_type="generation", name=f"model:{self.role}", model=self.model_name,
                input=[message_view(m, chars=None) for m in messages],
                model_parameters={"temperature": 0}) as generation:
            result = await self.runnable.ainvoke(messages)
            generation.update(output=_generation_output(result), usage_details=_usage(result))
        return result


def model_name_of(model) -> str | None:
    """The Bedrock model id of a ChatBedrockConverse; None for a test stub."""
    name = getattr(model, "model_id", None)
    return name if isinstance(name, str) else None


def trajectory(values: dict, outcome) -> dict:
    """The compact digest on the root `turn` span that the LLM judges read (selected
    with a jsonPath, Lesson 11). Tool calls with their results cut short, the state
    the turn carried in, and what it ended with. Never the whole metadata."""
    from agent.graph import current_turn  # the graph imports this module

    calls = []
    turn = current_turn(values.get("messages", []))
    results = {m.tool_call_id: m for m in turn if isinstance(m, ToolMessage)}
    for message in turn:
        for call in getattr(message, "tool_calls", None) or []:
            reply = results.get(call["id"])
            text = _text(reply) if reply is not None else ""
            calls.append({"tool": call["name"], "args": call["args"],
                          "status": reply.status if reply is not None else "not run",
                          "result": text[:1500] + ("..." if len(text) > 1500 else "")})
    plan = values.get("plan") or {}
    return {
        "question": _text(turn[0]) if turn else "",
        "intent": outcome.intent, "scope": outcome.scope,
        "constraints": plan.get("active_constraints", []),
        "facts_carried": plan.get("facts", []),
        "tool_calls": calls,
        "status": outcome.status,
        "approval_id": outcome.approval_id,
        "answer": outcome.answer,
    }


def compact_json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)
