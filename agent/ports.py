"""The Webex handoff boundary (Lesson 10 homework, milestone 4).

Maya does not import the Webex workflow. She calls a `WebexPort`, and the graph gets
one injected by run_turn(). Two implementations:

- `McpWebexPort`, the real one. It asks the Webex approver for an approval through the
  MCP server's `request_approval` tool, keyed on the handoff's idempotency key, and
  reports the pending row. `request_approval` reuses an open row for the same key, so
  a repeated handoff never creates a second approval. Task #5 extends this class; it
  does not replace it.
- `FakeWebexPort`, for tests. It records every call and returns a fixed pending result.

Neither can say access was granted: `PendingAccessResult.status` only admits pending.
"""

from typing import Protocol

from agent.mcp_tools import call_tool
from agent.policy import APPROVAL_TOOL
from agent.schemas import PendingAccessResult, WebexHandoff

# AP001 names the IT Manager (E006) as the approver on the Webex access path.
WEBEX_APPROVER = "E006"


class WebexPort(Protocol):
    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult: ...


class McpWebexPort:
    """`configurable` is the turn's graph config, so the MCP call carries the same
    caller, user group and request_id as every other call in the turn."""

    def __init__(self, session, configurable: dict):
        self.session = session
        self.configurable = configurable

    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult:
        row = await call_tool(self.session, APPROVAL_TOOL, {
            "request_type": "access_request",
            "record_id": handoff.idempotency_key,
            "approver_id": WEBEX_APPROVER,
            "employee_id": handoff.subject_employee_id,
            "software": handoff.software,
        }, self.configurable)
        return PendingAccessResult(approval_id=row["approval_id"],
                                   approver_id=row["approver_id"],
                                   reused=bool(row.get("reused")))


class FakeWebexPort:
    def __init__(self):
        self.calls: list[WebexHandoff] = []

    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult:
        self.calls.append(handoff)
        return PendingAccessResult(approval_id="AP-FAKE", approver_id=WEBEX_APPROVER)
