"""Stand-ins for the two models, so graph tests run without Bedrock.

The MCP server is the real one over the in-memory transport, on a reseeded file DB.
`use_stubs()` swaps `agent.run.make_models` and `agent.run.mcp_session`; everything
else, including run_turn(), the AsyncSqliteSaver and the tool wrapper, is the real code.
"""

import itertools
from contextlib import asynccontextmanager

from langchain_core.messages import AIMessage
from mcp.shared.memory import create_connected_server_and_client_session

from agent.policy import TurnPlan

_ids = itertools.count(1)


def plan(intent, *, needs_tools=True, action_confirmed=False, **extra) -> TurnPlan:
    return TurnPlan(intent=intent, needs_tools=needs_tools,
                    action_confirmed=action_confirmed, **extra)


def call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call-{next(_ids)}", "type": "tool_call"}])


def calls(*pairs) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": n, "args": a, "id": f"call-{next(_ids)}", "type": "tool_call"} for n, a in pairs])


def say(text: str) -> AIMessage:
    return AIMessage(content=text)


class ScriptedPlanner:
    """Returns the scripted results in order. None means a parse failure."""

    def __init__(self, *plans):
        self.plans = list(plans)
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        parsed = self.plans.pop(0)
        return {"raw": AIMessage(content="stub"), "parsed": parsed,
                "parsing_error": None if parsed is not None else "stub: not a TurnPlan"}


class ScriptedModel:
    """A chat model that answers from a script and records which tools were bound."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.bound: list[list[str]] = []
        self.seen: list[list] = []

    def bind_tools(self, tools):
        return _Bound(self, sorted(t.name for t in tools))

    async def ainvoke(self, messages):
        return self._next(messages, [])

    def _next(self, messages, names):
        self.bound.append(names)
        self.seen.append(messages)
        return self.responses.pop(0)


class _Bound:
    def __init__(self, model, names):
        self.model, self.names = model, names

    async def ainvoke(self, messages):
        return self.model._next(messages, self.names)


@asynccontextmanager
async def memory_session():
    from mcp_server.server import mcp
    async with create_connected_server_and_client_session(mcp) as client:
        yield client


def use_stubs(monkeypatch, planner, model):
    import agent.run
    monkeypatch.setattr(agent.run, "make_models", lambda: {"model": model, "planner": planner})
    monkeypatch.setattr(agent.run, "mcp_session", memory_session)
