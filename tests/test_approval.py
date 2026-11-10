"""The write gate and the durable approval pause."""

import pytest

from agent.gate import judge_approval, release_write
from agent.run import RequestIdMismatch, run_turn
from tests.agent_stubs import ScriptedModel, ScriptedPlanner, call, memory_session, plan, say, use_stubs
from evals import durability

# A confirmed access request goes anywhere only for a subject a get_employee row in the
# thread verifies (agent/handoff.py), so each approval test looks Rachel up first.
LOOKUP = plan("employee_lookup")
GO_AHEAD = plan("access_request", action_confirmed=True, subject_employee_id="E010",
                system="Webex")


def lookup_replies():
    return [call("get_employee", query="E010"), say("Rachel Stein is E010. [db:employees#E010]")]


async def look_up_rachel(thread_id: str, request_id: str | None = None):
    return await run_turn("Look up Rachel Stein.", caller_id="E007", user_group="UG_IT",
                          thread_id=thread_id, request_id=request_id)


CONFIG = {"thread_id": "t-gate", "caller_id": "E007", "user_group": "UG_IT",
          "request_id": "REQ-GATE"}
TODAY = "2026-07-01"


async def test_release_write_refuses_when_no_approval_row_exists(seeded):
    async with memory_session() as session:
        decision = await release_write(session, CONFIG, record_id="AR003")  # has no approvals
    assert decision.open is False and decision.state == "missing" and decision.released == []


async def test_release_write_refuses_a_seed_approval_with_no_decision(seeded):
    async with memory_session() as session:
        decision = await release_write(session, CONFIG, record_id="AR001", approval_id="AP001")
    assert decision.open is False and decision.state == "pending"


@pytest.mark.parametrize("row, state", [
    ({"approval_id": "A", "status": "rejected"}, "rejected"),
    ({"approval_id": "A", "status": "needed"}, "pending"),
    ({"approval_id": "A", "status": "approved", "consumed": 1, "expires_on": TODAY}, "consumed"),
    ({"approval_id": "A", "status": "approved", "consumed": 0, "expires_on": "2026-06-30"}, "expired"),
    ({"approval_id": "A", "status": "approved", "consumed": 0, "expires_on": None}, "expired"),
])
def test_gate_stays_shut_unless_approved_unused_and_unexpired(row, state):
    decision = judge_approval({"request_type": "access_request", **row}, TODAY)
    assert decision.open is False and decision.state == state and decision.released == []


def test_gate_opens_only_the_write_its_approval_type_releases():
    row = {"approval_id": "A", "status": "approved", "consumed": 0, "expires_on": TODAY}
    assert judge_approval({**row, "request_type": "access_request"}, TODAY).released == \
        ["create_access_request"]
    assert judge_approval({**row, "request_type": "subscription_expansion"}, TODAY).released == []


def rows(conn) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("tickets", "access_requests", "approvals")}


# The durability scenarios live in evals/durability.py; these tests run them on the
# temporary database and checkpoint file the fixtures set up.

def _assert_scenario(result: dict) -> None:
    failed = [check for check, ok in result["checks"].items() if not ok]
    assert not failed, failed


async def test_paused_thread_survives_a_killed_process_and_resumes_in_another(seeded):
    """Kill between the pause and the decision, restart, resume the same thread, release
    the write once, end blocked by the seat limit."""
    result = await durability.restart()
    _assert_scenario(result)
    from mcp_server import db
    with db.session() as conn:  # the scenario reseeded the file, so open it again
        control = conn.execute("SELECT requested_by, workflow_request_id, decided_by, consumed_key "
                               "FROM approval_controls WHERE approval_id = 'AP001'").fetchone()
    assert tuple(control[:3]) == ("E010", result["request_id"], "E006")
    assert control[3] == f"create_access_request:E010:S001:{result['request_id']}"


async def test_a_replayed_request_adds_no_row_and_consumes_the_approval_once(seeded):
    _assert_scenario(await durability.replay())


async def test_the_gate_opens_only_on_a_recorded_approval(seeded):
    _assert_scenario(await durability.gate())


async def test_a_recorded_decision_resumes_the_paused_trace(seeded):
    _assert_scenario(await durability.resume())


async def test_paused_thread_rejects_a_different_request_id(seeded, monkeypatch):
    use_stubs(monkeypatch, ScriptedPlanner(LOOKUP, GO_AHEAD),
              ScriptedModel(*lookup_replies(), call("get_approvals", record_id="AR001"),
                            say("Waits on AP001.")))
    await look_up_rachel("t-mismatch", "REQ-0")
    paused = await run_turn("Go ahead.", caller_id="E007", user_group="UG_IT",
                            thread_id="t-mismatch", request_id="REQ-1")
    assert paused.status == "pending"
    with pytest.raises(RequestIdMismatch):
        await run_turn("Approved?", caller_id="E007", user_group="UG_IT",
                       thread_id="t-mismatch", request_id="REQ-2")


async def test_resume_while_still_undecided_stays_pending(seeded, monkeypatch, conn):
    use_stubs(monkeypatch, ScriptedPlanner(LOOKUP, GO_AHEAD),
              ScriptedModel(*lookup_replies(), call("get_approvals", record_id="AR001"),
                            say("Waits on AP001.")))
    await look_up_rachel("t-wait")
    first = await run_turn("Go ahead.", caller_id="E007", user_group="UG_IT", thread_id="t-wait")
    again = await run_turn("Is it done?", caller_id="E007", user_group="UG_IT", thread_id="t-wait")
    assert first.status == again.status == "pending"
    assert again.request_id == first.request_id
    assert conn.execute("SELECT status FROM approvals WHERE approval_id = 'AP001'").fetchone()[0] \
        == "needed"


async def test_rejected_approval_ends_blocked_without_a_write(seeded, monkeypatch, conn):
    from agent.approvals import decide
    use_stubs(monkeypatch, ScriptedPlanner(LOOKUP, GO_AHEAD),
              ScriptedModel(*lookup_replies(), call("get_approvals", record_id="AR001"),
                            say("Waits on AP001.")))
    await look_up_rachel("t-reject")
    await run_turn("Go ahead.", caller_id="E007", user_group="UG_IT", thread_id="t-reject")
    decide("AP001", "rejected", "E006", "t-reject", reason="No seats this quarter.")
    out = await run_turn("Well?", caller_id="E007", user_group="UG_IT", thread_id="t-reject")
    assert out.status == "blocked" and "Nothing was filed" in out.answer
    assert "create_access_request" not in out.tools_called


async def test_maya_turn_with_action_confirmed_never_requests_an_approval(seeded, monkeypatch, conn):
    before = conn.execute("SELECT COUNT(*) FROM approvals").fetchone()[0]
    use_stubs(monkeypatch, ScriptedPlanner(plan("onboarding_status", action_confirmed=True)),
              ScriptedModel(call("get_approvals", record_id="AR001"), say("AP001 is needed."),
                            say("Still no approval tool."), say("Done.")))
    out = await run_turn("Go ahead and sort it.", caller_id="E004", user_group="UG_HR",
                         thread_id="t-maya")
    assert out.scope == "maya" and out.status != "pending" and out.approval_id is None
    assert "get_approvals" not in out.tools_called  # not in Maya's onboarding loadout
    assert conn.execute("SELECT COUNT(*) FROM approvals").fetchone()[0] == before
