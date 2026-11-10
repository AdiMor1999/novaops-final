"""Stand-ins for the models, so graph tests and the durability scenarios run without
Bedrock (moved here from tests/agent_stubs.py in task #6).

The MCP server is the real one over the in-memory transport, on a reseeded file DB.
`use_stubs()` swaps `agent.run.make_models`, `agent.run.mcp_session` and the server's
`retrieval.search`; everything else, including run_turn(), the AsyncSqliteSaver and the
tool wrapper, is the real code.
"""

import itertools
import re
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
    """Returns the scripted results in order. None means a parse failure. Also stands
    in for the distiller and the checklist model, which have the same shape."""

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

    def bind_tools(self, tools, **_kwargs):
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


class AutoDecision:
    """Stands in for the forced submit_access_decision model. It cites the first row
    label in what it is shown, so the draft passes the known-label check, and records
    every prompt it got."""

    LABEL = re.compile(r"\[db:[a-z_]+#[A-Za-z0-9_-]+\]")

    def __init__(self, statement: str = "The records were read.",
                 next_step: str = "The approver decides on the approval."):
        self.statement, self.next_step = statement, next_step
        self.seen: list[list] = []

    async def ainvoke(self, messages):
        self.seen.append(messages)
        labels = self.LABEL.findall(str(messages[-1].content))
        fact = {"statement": self.statement, "citations": [{"label": l} for l in labels[:1]]}
        parsed = {"observed_facts": [fact], "recommended_next_steps": [self.next_step]}
        return {"raw": AIMessage(content="stub"), "parsed": parsed, "parsing_error": None}


def no_documents(query, corpora, caller_id, **_kwargs) -> dict:
    """Stands in for retrieval.search in graph tests: the graph is under test, not
    OpenSearch or Bedrock (tests/test_prefilter.py and the live tests cover those)."""
    return {"query": query, "corpora": list(corpora), "count": 0, "chunks": [],
            "note": "No document matched this query in the searched collections."}


@asynccontextmanager
async def memory_session():
    from mcp_server.server import mcp
    async with create_connected_server_and_client_session(mcp) as client:
        yield client


def use_stubs(monkeypatch, planner, model, *, distiller=None, checklist=None, decision=None,
              port=None):
    """`distiller` and `checklist` default to None, which leaves those nodes out of
    the graph. `decision` defaults to an `AutoDecision`. `port` replaces the
    McpWebexPort; without it the real port runs against the in-memory server."""
    import agent.run
    decision = decision or AutoDecision()
    monkeypatch.setattr(agent.run, "make_models", lambda: {
        "model": model, "planner": planner, "distiller": distiller, "checklist": checklist,
        "decision": decision})
    monkeypatch.setattr(agent.run, "mcp_session", memory_session)
    monkeypatch.setattr("mcp_server.retrieval.search", no_documents)
    if port is not None:
        monkeypatch.setattr(agent.run, "make_port", lambda _session, _config: port)
