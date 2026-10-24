"""The graph's write gate: `release_write()`.

Starting code: Lesson 10 `select_tools`, which released `create_access_request` when
the planner said `action_confirmed`. That let model output open a write. Here the gate
reads the approval row over MCP and opens only for one that is `approved`, not consumed
and not expired.

This gate is a convenience, not authorization. `get_approvals` does not return the
`approval_controls` scope (employee, system, caller, workflow request_id), so the gate
cannot check it. The server's own check in `mcp_server/writes.py` does, inside the
write's transaction, and that check is the one that holds.
"""

from dataclasses import dataclass, field

from langchain_core.tools import ToolException

from agent import config
from agent.mcp_tools import call_tool
from agent.policy import RELEASES


@dataclass(frozen=True)
class GateDecision:
    open: bool
    state: str  # approved | pending | rejected | consumed | expired | missing | error
    reason: str
    approval: dict | None = None
    released: list[str] = field(default_factory=list)


def judge_approval(row: dict | None, today: str) -> GateDecision:
    """Pure decision over one get_approvals row. Kept separate so it is testable
    without a server."""
    if row is None:
        return GateDecision(False, "missing", "No approval row exists for this action.")
    status = row.get("status")
    if status == "rejected":
        return GateDecision(False, "rejected", f"{row['approval_id']} was rejected.", row)
    if status != "approved":
        return GateDecision(False, "pending",
                            f"{row['approval_id']} is {status}; no decision yet.", row)
    if row.get("consumed"):
        return GateDecision(False, "consumed", f"{row['approval_id']} was already used.", row)
    if not row.get("expires_on") or row["expires_on"] < today:
        return GateDecision(False, "expired",
                            f"{row['approval_id']} has no valid expiry (expires_on "
                            f"{row.get('expires_on')}).", row)
    return GateDecision(True, "approved", f"{row['approval_id']} is approved and unused.",
                        row, RELEASES.get(row.get("request_type"), []))


async def release_write(session, configurable: dict, *, record_id: str,
                        approval_id: str | None = None) -> GateDecision:
    """Open the gate only against a persisted approval row for `record_id`.

    With `approval_id`, only that row counts; without it, the newest row for the
    record does. Reading fails closed: an MCP error keeps the gate shut."""
    try:
        result = await call_tool(session, "get_approvals", {"record_id": record_id},
                                 configurable)
    except ToolException as exc:
        return GateDecision(False, "error", f"get_approvals failed: {exc}")
    rows = result.get("rows", [])
    if approval_id:
        rows = [r for r in rows if r.get("approval_id") == approval_id]
    return judge_approval(rows[-1] if rows else None, config.today().isoformat())
