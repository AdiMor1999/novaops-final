"""The one place the web chat records an approval decision.

The approvals panel calls `decide()`; chat text never does. The decision is written to
`approvals` by application code, and the next message on the paused thread resumes it
through `run_turn()`, whose write gate reads that row (agent/run.py). The MCP server
checks the approval again before any write.

Switching to task #6's `agent.approvals.decide()`, which adds an `approval_decision`
span, changes the import and the call below. That function also takes the paused
thread's id for its Langfuse session, which the panel would have to look up first.
"""

from mcp_server.writes import record_approval_decision

DECISIONS = ("approved", "rejected")


def decide(approval_id: str, decision: str, actor_id: str) -> dict:
    """Record `decision` on `approval_id` as `actor_id`, the caller selected in the page.
    `record_approval_decision` refuses an actor who is not the named approver, a row that
    is already decided and an unknown id, each with a ValueError subclass."""
    return record_approval_decision(approval_id, decision, actor_id,
                                    reason=f"{decision} by {actor_id} in the web chat")
