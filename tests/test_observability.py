"""Task #6: the trace-shape check, the heuristic switches, and the node-span wrapper."""

from types import SimpleNamespace

import pytest

from agent import heuristics
from agent.policy import checked_plan
from agent.schemas import ObservedFact
from agent.tracing import node
from evals.trace_tree import graph_nodes, problems
from tests.agent_stubs import plan

NODES = frozenset({"classify", "scope", "gather", "tools", "answer"})


def obs(id_, name, type_, parent=None, model=None, usage=None):
    return SimpleNamespace(id=id_, name=name, type=type_, parent_observation_id=parent,
                           model=model, usage_details=usage, start_time=id_)


def good_tree():
    return [obs("1", "turn", "AGENT"), obs("2", "classify", "SPAN", "1"),
            obs("3", "model:planner", "GENERATION", "2", "m", {"input": 10}),
            obs("4", "tools", "SPAN", "1"), obs("5", "tool:get_policy", "TOOL", "4"),
            obs("6", "server:get_policy", "SPAN", "5"), obs("7", "answer", "SPAN", "1")]


def test_the_readable_tree_passes():
    assert problems(good_tree(), NODES) == []


@pytest.mark.parametrize("extra, expected", [
    (obs("9", "RunnableSequence", "CHAIN", "2"), "anonymous runnable"),
    (obs("9", "ChatBedrockConverse", "GENERATION", "2"), "anonymous runnable"),
    (obs("9", "tool:get_policy", "TOOL", "3"), "not a node span"),
    (obs("9", "model:assistant", "GENERATION", "1", "m", {"input": 1}), "not a node span"),
    (obs("9", "model:assistant", "GENERATION", "2", None, {"input": 1}), "no model id"),
    (obs("9", "server:get_policy", "SPAN", "2"), "not its tool span"),
])
def test_each_break_of_the_shape_is_named(extra, expected):
    assert any(expected in p for p in problems([*good_tree(), extra], NODES))


def test_a_resumed_turn_with_a_dangling_parent_is_a_root():
    """The SDK gives a span started with only trace_context a random parent id."""
    resumed = [obs("10", "turn", "AGENT", "ffff"), obs("11", "gather", "SPAN", "10")]
    assert problems([*good_tree(), *resumed], NODES) == []


def test_the_node_names_come_from_the_graph():
    assert {"classify", "scope", "gather", "tools", "write_gate", "answer",
            "await_approval"} <= graph_nodes()


async def test_a_node_span_keeps_the_signature_langgraph_reads():
    async def with_config(state, config):
        return {"seen": config["configurable"]["x"]}

    def without(state):
        return {"n": state["n"] + 1}

    assert await node("a", with_config)({}, {"configurable": {"x": 1}}) == {"seen": 1}
    assert await node("b", without)({"n": 1}) == {"n": 2}


# Heuristic switches -----------------------------------------------------------------

def test_an_unknown_heuristic_name_is_refused(monkeypatch):
    monkeypatch.setenv("NOVAOPS_HEURISTICS_OFF", "needs_tools_overide")
    with pytest.raises(heuristics.UnknownHeuristic):
        heuristics.enabled("needs_tools_override")


def test_needs_tools_override_fires_and_can_be_switched_off(monkeypatch):
    first_turn = plan("policy_question", needs_tools=False)
    with heuristics.collect() as fired:
        assert checked_plan(first_turn, has_history=False).needs_tools is True
    assert fired == ["needs_tools_override"]
    monkeypatch.setenv("NOVAOPS_HEURISTICS_OFF", "needs_tools_override")
    with heuristics.collect() as fired:
        assert checked_plan(first_turn, has_history=False).needs_tools is False
    assert fired == []


def test_self_access_confirmation_can_be_switched_off(monkeypatch):
    ask = plan("access_request")
    message = "I need Webex access. Webex says my account is not licensed."
    assert checked_plan(ask, False, message, "E010").action_confirmed is True
    monkeypatch.setenv("NOVAOPS_HEURISTICS_OFF", "self_access_confirmation")
    assert checked_plan(ask, False, message, "E010").action_confirmed is False


def test_citation_brace_repair_fires_only_on_a_repair(monkeypatch):
    braced = {"statement": "Seats are full.", "citations": [{"label": "[db:tickets#T001}"}]}
    with heuristics.collect() as fired:
        ObservedFact.model_validate(braced)
        ObservedFact.model_validate({"statement": "x", "citations": [{"label": "[db:tickets#T001]"}]})
    assert fired == ["citation_brace_repair"]
    monkeypatch.setenv("NOVAOPS_HEURISTICS_OFF", "citation_brace_repair")
    with pytest.raises(ValueError):
        ObservedFact.model_validate(braced)
