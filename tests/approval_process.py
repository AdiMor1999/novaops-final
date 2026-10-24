"""One phase of the pause, resume and replay test, run in its own process.

    python -m tests.approval_process start    # Rachel's request, until the approval interrupt; then SIGKILLs itself
    python -m tests.approval_process resume   # records E006's decision, resumes the same thread
    python -m tests.approval_process replay   # sends the same request again, same thread and request_id

Each phase prints its TurnOutcome as one JSON line. The database and the checkpoint
file come from NOVAOPS_DB_PATH and NOVAOPS_CHECKPOINT_DB, set by the parent test.

The scenario is spec §9's example: Rachel Stein (E010) asks for Webex access for
herself. No earlier turn looked her up, so code does (agent/handoff.py).
"""

import asyncio
import os
import signal
import sys

import pytest

from agent.run import run_turn
from mcp_server.writes import record_approval_decision
from tests.agent_stubs import ScriptedModel, ScriptedPlanner, call, plan, say, use_stubs

THREAD = "t-approval-rachel"
REQUEST = "REQ-RACHEL"
CALLER = {"caller_id": "E010", "user_group": "UG_REGULAR"}
RACHEL = ("I joined Customer Success and need Webex access. Webex says my account is not "
          "licensed.")
ASK = plan("access_request", action_confirmed=True, subject_employee_id="E010", system="Webex")


def start():
    model = ScriptedModel(call("check_software_subscription", software="Webex"),
                          say("Webex is over its seat limit."))
    return ScriptedPlanner(ASK), model, RACHEL


def resume():
    record_approval_decision("AP001", "approved", "E006", reason="Seat freed by IT.")
    model = ScriptedModel(call("create_access_request", employee_id="E010", software="Webex",
                               business_justification="Customer onboarding calls require Webex."))
    return ScriptedPlanner(), model, "E006 has approved it."


def replay():
    model = ScriptedModel(call("check_software_subscription", software="Webex"),
                          say("Webex is over its seat limit."),
                          call("create_access_request", employee_id="E010", software="Webex",
                               business_justification="Customer onboarding calls require Webex."))
    return ScriptedPlanner(ASK), model, RACHEL


PHASES = {"start": start, "resume": resume, "replay": replay}


async def main(phase: str) -> None:
    planner, model, message = PHASES[phase]()
    with pytest.MonkeyPatch.context() as patch:
        use_stubs(patch, planner, model)
        outcome = await run_turn(message, thread_id=THREAD,
                                 request_id=None if phase == "resume" else REQUEST, **CALLER)
    print(outcome.model_dump_json(), flush=True)
    if phase == "start":
        # No clean shutdown: whatever survives this is what the checkpointer wrote.
        os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
