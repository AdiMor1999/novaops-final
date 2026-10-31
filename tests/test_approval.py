"""The write gate and the durable approval pause."""

import json
import signal
import subprocess
import sys

import pytest

from agent.gate import judge_approval, release_write
from agent.run import RequestIdMismatch, run_turn
from tests.agent_stubs import ScriptedModel, ScriptedPlanner, call, memory_session, plan, say, use_stubs
from tests.conftest import REPO

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


def _phase(phase: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "tests.approval_process", phase], cwd=REPO,
                          capture_output=True, text=True, timeout=120)


def _outcome(proc: subprocess.CompletedProcess) -> dict:
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    assert lines, proc.stderr[-3000:]
    return json.loads(lines[-1])


def rows(conn) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("tickets", "access_requests", "approvals")}


def test_paused_thread_survives_a_killed_process_and_resumes_in_another(seeded, conn):
    """Done-when of Milestone 4: kill between the pause and the decision, restart,
    resume the same thread, release the write once, end blocked by the seat limit.
    A replay after the release returns AR001 again and adds no row."""
    before = rows(conn)
    first = _phase("start")
    assert first.returncode == -signal.SIGKILL, first.stderr[-3000:]
    paused = _outcome(first)
    assert paused["status"] == "pending" and paused["approval_id"] == "AP001"
    assert paused["approvals"] == ["AP001", "AP002"]
    assert paused["tools_called"][0] == "get_employee"  # the subject lookup, from code
    assert "granted" not in paused["answer"].replace("no access is granted", "")
    assert rows(conn) == before

    second = _phase("resume")
    assert second.returncode == 0, second.stderr[-3000:]
    resumed = _outcome(second)
    assert resumed["request_id"] == paused["request_id"]
    assert resumed["trace_id"] == paused["trace_id"]
    assert resumed["status"] == "blocked"
    assert resumed["tools_called"].count("create_access_request") == 1
    decision = resumed["access_decision"]
    assert decision["status"] == "blocked" and "42 active seats" in decision["blocking_reason"]
    write = [a for a in decision["actions_taken"] if a["tool"] == "create_access_request"]
    assert [(a["record_id"], a["reused"]) for a in write] == [("AR001", True)]
    assert rows(conn) == before
    control = conn.execute("SELECT requested_by, workflow_request_id, decided_by, consumed_key "
                           "FROM approval_controls WHERE approval_id = 'AP001'").fetchone()
    assert tuple(control[:3]) == ("E010", paused["request_id"], "E006")
    assert control[3] == "create_access_request:E010:S001:REQ-RACHEL"

    third = _phase("replay")
    assert third.returncode == 0, third.stderr[-3000:]
    replayed = _outcome(third)
    assert replayed["status"] == "blocked" and replayed["approvals"][0] == "AP001"
    write = [a for a in replayed["access_decision"]["actions_taken"]
             if a["tool"] == "create_access_request"]
    assert [(a["record_id"], a["reused"]) for a in write] == [("AR001", True)]
    assert rows(conn) == before
    assert conn.execute("SELECT COUNT(*) FROM approval_controls WHERE approval_id = 'AP001' "
                        "AND consumed_at IS NOT NULL").fetchone()[0] == 1


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
    from mcp_server.writes import record_approval_decision
    use_stubs(monkeypatch, ScriptedPlanner(LOOKUP, GO_AHEAD),
              ScriptedModel(*lookup_replies(), call("get_approvals", record_id="AR001"),
                            say("Waits on AP001.")))
    await look_up_rachel("t-reject")
    await run_turn("Go ahead.", caller_id="E007", user_group="UG_IT", thread_id="t-reject")
    record_approval_decision("AP001", "rejected", "E006", reason="No seats this quarter.")
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
