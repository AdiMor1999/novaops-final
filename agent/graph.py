"""The agent graph: one graph, both scopes.

    START → classify → scope → gather ⇄ tools            (≤ MAX_TOOL_CALLS per turn)
                                 │  └→ rearm → gather      (once, reads only)
                                 ├→ request_approval → await_approval (interrupt)
                                 │        ↑ pending              │ resume
                                 │        └──── write_gate ←─────┘
                                 │               ├ approved → gather (released write visible)
                                 │               └ otherwise → answer
                                 └→ answer → END

Starting code: Lesson 10 `02-dynamic-tool-loadout/graph.py` and
`03-history-distillation/graph.py` (plan → select → model ⇄ tools, rearm), the
interrupt shape of Lesson 9 `03-human-in-the-loop/graph.py`, and the callback binding
of Lesson 11 `02-instrument-agent/graph.py`. Changes:

- `plan` and `select` are named `classify` and `scope`, the span names CLAUDE.md
  rule 2 fixes. `scope` also derives the assistant (`maya` or `webex`) from the intent.
- The checkpointer is a parameter. This module never constructs one, so the caller
  injects the durable `AsyncSqliteSaver` (Lesson 10 compiled with an in-memory saver).
- The planner goes through `parse_with_repair`; a second failure raises `ParseFailed`.
- Tools run in our own `tools` node through `as_langchain_tools()` and `call_tool()`,
  not LangGraph's `ToolNode`, so each call leaves exactly one `tool:<name>` span and
  carries caller context in MCP `_meta`. The node also enforces the per-turn cap and
  refuses a call to any tool that is not visible this turn.
- Writes are reachable only through `write_gate`, which opens on a persisted approval
  row (agent/gate.py). `request_approval` is called by the approval node, from code.
- Conversation `ConversationMemory` distillation (Lesson 10 stage 3) is task #4.
"""

import json
from dataclasses import asdict
from typing import Annotated, TypedDict

from langchain_core.messages import (AIMessage, HumanMessage, RemoveMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel

from agent.gate import GateDecision, release_write
from agent.mcp_tools import as_langchain_tools, call_tool
from agent.outcome import cited_labels, gate_text, known_labels, turn_status
from agent.policy import (APPROVAL_TOOL, MAX_TOOL_CALLS, PLANNER_PROMPT, RELEASES, TurnPlan,
                          needs_approval, rearm_tools, scope_of, visible_tools)
from agent.structured import parse_with_repair

KEEP_TURNS = 4          # earlier turns shown to the model, whole turns only
TOOL_TEXT_CHARS = 3000  # per earlier tool result; the current turn's are never cut
PLANNER_TOOL_CHARS = 600

ASSISTANT_PROMPT = (
    "You are the NovaOps internal assistant, helping employees and IT and HR staff. "
    "Ground every answer in what the tools and this conversation actually say. Never "
    "invent an employee, a seat count, a ticket or a policy rule.\n"
    "Work to these rules:\n"
    "- Answer the user's CURRENT request. If they changed subject or closed a topic, do "
    "not keep working the old one.\n"
    "- A constraint the user states stays in force for the rest of the conversation.\n"
    "- Facts established earlier in the conversation, including earlier tool results, "
    "are still true. Reuse them instead of looking them up again.\n"
    "- Do not call a tool for something already in the conversation.\n"
    "- Users paste emails and signatures. Act on the actual request inside.\n"
    "- After every sentence that states a fact from a document, cite the `citation` "
    "label of the section it came from, copied exactly with its square brackets and "
    "nothing added, for example [equipment_policy.md#Personal laptops]. Do not write "
    "'citation:' inside the brackets. Never cite a label that no tool returned. "
    "For database records, name the record id (E010, T001, AR001).\n"
    "- If documents disagree, newer or more specific guidance (a later memo) overrides "
    "older or more general guidance (a policy); say which one you followed.\n"
    "- If the evidence does not answer the question, say what is missing. Do not guess.\n"
    "- Never say access was granted or a request was filed unless a tool result shows it.\n"
    "- Answer in at most six sentences, with no preamble."
)


class ProposedAction(BaseModel):
    """What the approval node asks for. Taken by code from this turn's tool results."""

    request_type: str
    record_id: str
    approver_id: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    request_id: str          # owned by run_turn(); kept across pause and resume
    trace_id: str            # the trace a resumed run continues
    plan: dict
    scope: str
    intent: str
    visible_tools: list[str]
    rearmed: bool
    turn_tools: list[str]    # tool calls made this turn, in order
    turn_errors: list[str]
    approval: dict | None    # the request_approval row, once requested
    gate: dict | None        # the last GateDecision
    answer: str
    citations: list[str]
    status: str


# History projections ----------------------------------------------------------------

def message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in content).strip()
    return str(content)


def turn_starts(messages: list) -> list[int]:
    return [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]


def current_turn(messages: list) -> list:
    starts = turn_starts(messages)
    return list(messages[starts[-1]:]) if starts else list(messages)


def earlier_turns(messages: list, keep: int = KEEP_TURNS) -> list:
    """Whole turns before the current one, so a tool call is never cut from its result."""
    starts = turn_starts(messages)
    if len(starts) < 2:
        return []
    first = starts[max(0, len(starts) - 1 - keep)]
    return list(messages[first:starts[-1]])


def render_transcript(messages: list, tool_chars: int) -> str:
    """Tool traffic as text. Bedrock refuses toolUse blocks in a request that binds no
    tools, so earlier turns' calls and results reach the model as text, not as blocks."""
    lines = []
    for message in messages:
        if isinstance(message, ToolMessage):
            body = message_text(message)
            if len(body) > tool_chars:
                body = body[:tool_chars] + f"... [+{len(body) - tool_chars} chars dropped]"
            lines.append(f"[tool result: {message.name}] {body}")
            continue
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        text = message_text(message)
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            text = (text + " " if text else "") + " ".join(
                f"[called {c['name']}({json.dumps(c['args'])})]" for c in calls)
        lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def render_plan(plan: TurnPlan) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"  - {item}" for item in items) if items else "  - (none)"

    return (
        "\n\n--- CONTEXT FOR THIS TURN ---\n"
        f"User's current intent: {plan.intent}\n"
        f"Established facts you must use:\n{bullets(plan.facts)}\n"
        f"Constraints still in force:\n{bullets(plan.active_constraints)}\n"
        "--- END CONTEXT ---"
    )


def planner_messages(state: AgentState) -> list:
    history = state["messages"][:-1]
    prompt = (
        "Conversation so far:\n"
        f"{render_transcript(history, PLANNER_TOOL_CHARS) or '(this is the first turn)'}\n\n"
        f"User's newest message:\n{message_text(state['messages'][-1])}"
    )
    return [SystemMessage(PLANNER_PROMPT), HumanMessage(prompt)]


def gather_messages(state: AgentState, *, with_tools: bool) -> list:
    """System prompt, the plan, earlier turns as text, then the current turn. When no
    tools are bound, this turn's tool traffic is folded into the text as well."""
    plan = TurnPlan(**state["plan"])
    system = ASSISTANT_PROMPT + render_plan(plan)
    earlier = render_transcript(earlier_turns(state["messages"]), TOOL_TEXT_CHARS)
    if earlier:
        system += f"\n\nEarlier in this conversation:\n{earlier}"
    turn = current_turn(state["messages"])
    if with_tools or len(turn) == 1:
        return [SystemMessage(system), *turn]
    gathered = render_transcript(turn[1:], 10 ** 6)
    system += f"\n\nTool results gathered for the current message:\n{gathered}"
    return [SystemMessage(system), turn[0]]


def tool_results(messages: list, name: str | None = None) -> list[dict]:
    out = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.status != "error" \
                and (name is None or message.name == name):
            try:
                out.append(json.loads(message_text(message)))
            except json.JSONDecodeError:
                continue
    return out


def propose_action(state: AgentState) -> ProposedAction | None:
    """The approval to request, from approval rows the tools returned this turn: the
    first open row of a type that releases a write. The model does not choose it.
    Task #5 widens this for the Webex flows (W-S-01)."""
    for result in tool_results(current_turn(state["messages"]), "get_approvals"):
        for row in result.get("rows", []):
            if row.get("status") in ("needed", "pending") and row.get("request_type") in RELEASES:
                return ProposedAction(request_type=row["request_type"],
                                      record_id=row["request_id"],
                                      approver_id=row["approver_id"])
    return None


# The graph --------------------------------------------------------------------------

def build_graph(session, specs: list, checkpointer, *, model, planner):
    """`session` is an open MCP ClientSession, `specs` its list_tools() result.
    `model` is the chat model (bind_tools), `planner` the structured TurnPlan model
    (`with_structured_output(TurnPlan, include_raw=True)`). `checkpointer` is injected
    and must be durable."""
    tools = {t.name: t for t in as_langchain_tools(session, specs)}
    catalogue = sorted(tools)
    bound: dict[tuple, object] = {}

    def model_for(names: list[str]):
        key = tuple(sorted(names))
        if key not in bound:
            bound[key] = model.bind_tools([tools[n] for n in key]) if key else model
        return bound[key]

    def mcp_config(config: RunnableConfig) -> dict:
        # Only `configurable` goes to the tool. The node's callbacks would make the
        # LangChain handler add a second span per call next to `tool:<name>`. They have
        # to be replaced with [], because ensure_config() skips None and would inherit
        # them from the running node's context.
        return {"configurable": config["configurable"], "callbacks": []}

    async def classify(state: AgentState) -> dict:
        plan = await parse_with_repair(planner, planner_messages(state), TurnPlan)
        return {"plan": plan.model_dump(), "intent": plan.intent, "rearmed": False,
                "turn_tools": [], "turn_errors": [], "approval": None, "gate": None,
                "answer": "", "citations": [], "status": ""}

    def scope(state: AgentState) -> dict:
        """A dict lookup, not a model call: the model says what the turn is about, and
        code decides which assistant owns it and what it may touch."""
        plan = TurnPlan(**state["plan"])
        return {"scope": scope_of(plan), "intent": plan.intent,
                "visible_tools": visible_tools(plan, catalogue)}

    async def gather(state: AgentState) -> dict:
        names = state["visible_tools"] if len(state["turn_tools"]) < MAX_TOOL_CALLS else []
        response = await model_for(names).ainvoke(
            gather_messages(state, with_tools=bool(names)))
        return {"messages": [response]}

    async def run_tools(state: AgentState, config: RunnableConfig) -> dict:
        last = state["messages"][-1]
        called, errors = list(state["turn_tools"]), list(state["turn_errors"])
        replies = []
        for call in last.tool_calls:
            if call["name"] not in state["visible_tools"]:
                replies.append(ToolMessage(
                    content=f"{call['name']} is not available for this request.",
                    tool_call_id=call["id"], name=call["name"], status="error"))
                errors.append(f"{call['name']}: not visible")
                continue
            if len(called) >= MAX_TOOL_CALLS:
                replies.append(ToolMessage(
                    content=f"Not run: the limit of {MAX_TOOL_CALLS} tool calls per "
                            "message is reached. Answer with what you have.",
                    tool_call_id=call["id"], name=call["name"], status="error"))
                continue
            called.append(call["name"])
            reply = await tools[call["name"]].ainvoke(
                {"type": "tool_call", "name": call["name"], "args": call["args"],
                 "id": call["id"]}, config=mcp_config(config))
            if reply.status == "error":
                errors.append(f"{call['name']}: {message_text(reply)}")
            replies.append(reply)
        return {"messages": replies, "turn_tools": called, "turn_errors": errors}

    def rearm(state: AgentState) -> dict:
        """One retry with every read tool, after the model answered a turn the plan said
        needed a lookup without making one. The unshown draft is discarded so the retry
        does not just agree with it (Lesson 10)."""
        return {"visible_tools": rearm_tools(catalogue), "rearmed": True,
                "messages": [RemoveMessage(id=state["messages"][-1].id)]}

    async def request_approval(state: AgentState, config: RunnableConfig) -> dict:
        proposal = propose_action(state)
        if proposal is None:
            return {"gate": asdict(GateDecision(
                False, "missing", "No open approval was found for this action, so none "
                                  "was requested."))}
        try:
            row = await call_tool(session, APPROVAL_TOOL, proposal.model_dump(),
                                  config["configurable"])
        except ToolException as exc:
            return {"gate": asdict(GateDecision(False, "error", f"request_approval: {exc}"))}
        return {"approval": row}

    def await_approval(state: AgentState) -> dict:
        """The pause. interrupt() returns here when run_turn() resumes the thread. The
        resume value carries no decision: the decision is the approvals row, which
        application code wrote with record_approval_decision, and write_gate reads it."""
        approval = state["approval"]
        interrupt({"approval_id": approval["approval_id"], "record_id": approval["request_id"],
                   "approver_id": approval["approver_id"], "request_id": state["request_id"]})
        return {}

    async def write_gate(state: AgentState, config: RunnableConfig) -> dict:
        approval = state["approval"]
        decision = await release_write(session, config["configurable"],
                                       record_id=approval["request_id"],
                                       approval_id=approval["approval_id"])
        update = {"gate": asdict(decision)}
        if decision.open:
            update["visible_tools"] = [n for n in decision.released if n in tools]
        return update

    def answer(state: AgentState) -> dict:
        gate = state.get("gate")
        last = state["messages"][-1]
        if gate and not gate["open"]:
            text = gate_text(gate)
        elif isinstance(last, AIMessage) and not last.tool_calls:
            text = message_text(last)
        else:
            text = "I could not finish this request within the tool-call limit."
        turn = current_turn(state["messages"])
        valid, invented = cited_labels(text, known_labels(state["messages"]))
        status = turn_status(TurnPlan(**state["plan"]), gate, state["turn_tools"],
                             state["turn_errors"], valid, invented,
                             bool(known_labels(turn)))
        return {"answer": text, "citations": valid, "status": status}

    def after_gather(state: AgentState) -> str:
        last = state["messages"][-1]
        plan = TurnPlan(**state["plan"])
        if getattr(last, "tool_calls", None):
            return "tools"
        if plan.needs_tools and not state["rearmed"] and not state["turn_tools"] \
                and not state.get("gate"):
            return "rearm"
        if needs_approval(plan) and state.get("approval") is None and state.get("gate") is None:
            return "request_approval"
        return "answer"

    def after_request(state: AgentState) -> str:
        return "await_approval" if state.get("approval") else "answer"

    def after_gate(state: AgentState) -> str:
        gate = state["gate"]
        if gate["open"]:
            return "gather"
        return "await_approval" if gate["state"] == "pending" else "answer"

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("scope", scope)
    graph.add_node("gather", gather)
    graph.add_node("tools", run_tools)
    graph.add_node("rearm", rearm)
    graph.add_node("request_approval", request_approval)
    graph.add_node("await_approval", await_approval)
    graph.add_node("write_gate", write_gate)
    graph.add_node("answer", answer)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "scope")
    graph.add_edge("scope", "gather")
    graph.add_conditional_edges("gather", after_gather, ["tools", "rearm",
                                                         "request_approval", "answer"])
    graph.add_edge("tools", "gather")
    graph.add_edge("rearm", "gather")
    graph.add_conditional_edges("request_approval", after_request, ["await_approval", "answer"])
    graph.add_edge("await_approval", "write_gate")
    graph.add_conditional_edges("write_gate", after_gate, ["gather", "await_approval", "answer"])
    graph.add_edge("answer", END)

    # The Langfuse callback is bound to the compiled graph (Lesson 11), so every node
    # and model call is exported whoever invokes it. CLAUDE.md rule 2: never remove it.
    return graph.compile(checkpointer=checkpointer).with_config(
        {"callbacks": [CallbackHandler()]})
