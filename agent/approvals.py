"""Recording a human approval decision: `decide()`, the one application function for it.

The model never records a decision (CLAUDE.md rule 3), and neither does the graph:
`record_approval_decision` in `mcp_server/writes.py` is not an MCP tool. Whatever
stands for the approver (a test, an eval scenario, later an approval screen) calls
`decide()`, and the paused thread's next `run_turn()` reads the row in `write_gate`.

`decide()` runs under its own Langfuse span, `approval_decision`, in its own trace.
The trace carries the thread's `session_id`, so the session view in Langfuse shows the
pause, the decision and the resume in order, and `user_id` is the approver, who made
this decision, not the caller of the thread.
"""

from langfuse import get_client, propagate_attributes

from mcp_server import db
from mcp_server.writes import record_approval_decision

DECIDED = ("approved", "rejected")


def decide(approval_id: str, decision: str, actor_id: str, thread_id: str, *,
           reason: str = "") -> dict:
    """Record `decision` ('approved' or 'rejected') on `approval_id` as `actor_id`, for
    the paused thread `thread_id`. Returns the updated approvals row. The server-side
    rules hold: only the named approver may decide, and only an open approval."""
    langfuse = get_client()
    request = {"approval_id": approval_id, "decision": decision, "actor_id": actor_id,
               "thread_id": thread_id, "reason": reason}
    try:
        with propagate_attributes(session_id=thread_id, user_id=actor_id,
                                  trace_name="approval_decision",
                                  metadata={"approval_id": approval_id,
                                            "decision": decision}):
            with langfuse.start_as_current_observation(
                    as_type="span", name="approval_decision", input=request) as span:
                row = record_approval_decision(approval_id, decision, actor_id, reason=reason)
                span.update(output=row)
    finally:
        langfuse.flush()
    return row


def is_decided(approval_id: str) -> bool:
    """True once a decision is recorded on the row. `run_turn()` asks this of a paused
    thread's approval: decided resumes the pause, undecided runs the message as a
    normal turn and the thread stays paused."""
    with db.session() as conn:
        row = conn.execute("SELECT status FROM approvals WHERE approval_id = ?",
                           (approval_id,)).fetchone()
    return row is not None and row["status"] in DECIDED
