"""One half of the pause and resume test, run in its own process.

    python -m tests.approval_process start   # runs until the approval interrupt, then SIGKILLs itself
    python -m tests.approval_process resume  # records the decision, resumes the same thread

Each half prints its TurnOutcome as one JSON line. The database and the checkpoint
file come from NOVAOPS_DB_PATH and NOVAOPS_CHECKPOINT_DB, set by the parent test.
"""

import asyncio
import os
import signal
import sys

import pytest

from agent.run import run_turn
from mcp_server.writes import record_approval_decision
from tests.agent_stubs import ScriptedModel, ScriptedPlanner, call, plan, say, use_stubs

THREAD = "t-approval-AR001"
CALLER = {"caller_id": "E007", "user_group": "UG_IT"}


def start():
    """Two turns: look Rachel up (the subject check needs her row in the thread),
    then confirm the request, which pauses on the approval."""
    planner = ScriptedPlanner(
        plan("employee_lookup"),
        plan("access_request", action_confirmed=True, subject_employee_id="E010",
             system="Webex"))
    model = ScriptedModel(call("get_employee", query="E010"),
                          say("Rachel Stein is E010. [db:employees#E010]"),
                          call("get_approvals", record_id="AR001"),
                          say("AR001 for E010 on Webex waits on AP001 from E006."))
    return planner, model, ["Look up Rachel Stein.",
                            "Go ahead and get Rachel Stein (E010) her Webex access."]


def resume():
    record_approval_decision("AP001", "approved", "E006", reason="Seat freed by IT.")
    model = ScriptedModel(
        call("create_access_request", employee_id="E010", software="Webex",
             business_justification="Customer onboarding calls require Webex."),
        say("Access request AR001 is on file with status blocked; no new row was made."))
    return ScriptedPlanner(), model, ["E006 has approved it."]


async def main(phase: str) -> None:
    planner, model, messages = start() if phase == "start" else resume()
    with pytest.MonkeyPatch.context() as patch:
        use_stubs(patch, planner, model)
        for message in messages:
            outcome = await run_turn(message, thread_id=THREAD, **CALLER)
    print(outcome.model_dump_json(), flush=True)
    if phase == "start":
        # No clean shutdown: whatever survives this is what the checkpointer wrote.
        os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
