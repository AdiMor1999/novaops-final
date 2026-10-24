from langchain_core.messages import ToolMessage

from agent.outcome import cited_labels, known_labels, turn_status
from tests.agent_stubs import plan

LABEL = "[equipment_policy.md#Personal laptops]"
KNOWN = {LABEL}


def test_cited_labels_split_known_and_invented():
    answer = f"Not allowed {LABEL}. Also [made_up.md#Nowhere]."
    assert cited_labels(answer, KNOWN) == ([LABEL], ["[made_up.md#Nowhere]"])


def test_known_labels_come_from_successful_tool_results_only():
    ok = ToolMessage(content=f'{{"citation": "{LABEL}"}}', tool_call_id="1", name="get_policy")
    bad = ToolMessage(content="[x.md#y] failed", tool_call_id="2", name="get_policy",
                      status="error")
    assert known_labels([ok, bad]) == KNOWN


def test_status_comes_from_tools_gate_and_citations():
    needs = plan("policy_question")
    assert turn_status(needs, None, ["get_policy"], [], [LABEL], [], True) == "completed"
    assert turn_status(needs, None, ["get_policy"], [], [], [], True) == "needs_human"
    assert turn_status(needs, None, ["get_policy"], [], [LABEL], ["[x.md#y]"], True) == "needs_human"
    assert turn_status(needs, None, ["get_policy"], ["get_policy: boom"], [], [], False) == "failed"
    assert turn_status(needs, None, [], [], [], [], False) == "needs_human"
    # A follow-up answered from an earlier turn's evidence.
    assert turn_status(plan("policy_question", needs_tools=False), None, [], [], [LABEL], [],
                       False) == "completed"


def test_shut_gate_is_blocked_only_when_rejected():
    act = plan("access_request", action_confirmed=True)
    assert turn_status(act, {"open": False, "state": "rejected"}, [], [], [], [], False) == "blocked"
    assert turn_status(act, {"open": False, "state": "missing"}, [], [], [], [], False) == "needs_human"
