"""Every write in the system, enforced here and not in the graph.

Starting code: Lesson 13 `SDD/shared/harness/enforcement.py` (one BEGIN IMMEDIATE
transaction that checks a single-use grant, inserts and consumes it; the confirmation
step is not an MCP tool). Lesson 13's `filing_grants` table becomes `approval_controls`,
tied to a row in the course's `approvals` table, and idempotency keys are added
(Lesson 13 replay attack, Lesson 14 at-least-once delivery).

Three kinds of write (rules in `mcp_server/CLAUDE.md`):

- Approval-gated: `create_access_request`, `create_ticket`.
- `request_approval`: creates a pending approval row, idempotent, needs no approval.
- `record_approval_decision`: application code only. Not an MCP tool.

`approval_controls` and `idempotency_keys` are side tables created by reseed. The
course schema and seed are not changed.
"""

import sqlite3
from datetime import timedelta

from agent import config
from mcp_server import db
from mcp_server.context import CallerContext
from mcp_server.db import ToolInputError

CONTROL_SCHEMA = """
CREATE TABLE IF NOT EXISTS approval_controls (
    approval_id TEXT PRIMARY KEY REFERENCES approvals(approval_id),
    employee_id TEXT NOT NULL,
    system_id TEXT,
    requested_by TEXT NOT NULL,
    workflow_request_id TEXT NOT NULL,
    expires_on TEXT NOT NULL,
    decided_by TEXT,
    decided_at TEXT,
    consumed_key TEXT,
    consumed_at TEXT
);
CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    workflow_request_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

APPROVAL_TTL_DAYS = 7
REQUEST_TYPES = ("access_request", "subscription_expansion", "ticket")
# Which approval type opens which gated write.
GATING_TYPE = {"create_access_request": "access_request", "create_ticket": "ticket"}
# Seed statuses: 'needed' and 'pending' are both still waiting on a decision.
OPEN_APPROVAL = ("needed", "pending")
OPEN_ACCESS_REQUEST = ("pending_approval", "blocked", "planned", "waiting_on_manager")
CLOSED_TICKET = ("resolved", "closed", "cancelled")
TICKET_TEAMS = {"access": "IT", "hardware": "IT", "security": "Security",
                "onboarding": "IT", "software": "IT", "network": "IT"}


class WriteRefused(ToolInputError):
    """A gated write without a matching approved, unused, unexpired approval."""


# Shared helpers ------------------------------------------------------------------

def _next_id(conn, table: str, column: str, prefix: str, width: int = 3) -> str:
    """Continue the seed's readable scheme (AR005, T007, AP005, EV003)."""
    numbers = [
        int(r[0][len(prefix):]) for r in conn.execute(f"SELECT {column} FROM {table}")
        if r[0].startswith(prefix) and r[0][len(prefix):].isdigit()
    ]
    return f"{prefix}{(max(numbers) if numbers else 0) + 1:0{width}d}"


def _audit(conn, entity_type: str, entity_id: str, event_type: str, actor: str, notes: str):
    conn.execute(
        "INSERT INTO audit_log (event_id, entity_type, entity_id, event_type, "
        "actor_employee_id, created_at, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (_next_id(conn, "audit_log", "event_id", "EV"), entity_type, entity_id,
         event_type, actor, config.now_iso(), notes),
    )


def _transaction(conn, work):
    """Run `work(conn)` inside BEGIN IMMEDIATE. IMMEDIATE takes the write lock up front,
    so the approval check, the idempotency lookup and the insert are serialized even
    across server processes. Any exception rolls everything back."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = work(conn)
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return result


def _caller_known(conn, ctx: CallerContext) -> None:
    caller = conn.execute(
        "SELECT user_group_id FROM employees WHERE employee_id = ?", (ctx.caller_id,)
    ).fetchone()
    if caller is None:
        raise WriteRefused(f"Refused: caller {ctx.caller_id!r} is not a NovaOps employee.")
    if caller["user_group_id"] != ctx.user_group:
        raise WriteRefused(
            f"Refused: caller {ctx.caller_id} is in {caller['user_group_id']}, "
            f"not {ctx.user_group}."
        )


# Approval-gated writes ------------------------------------------------------------

def _matching_approval(conn, action: str, employee_id: str, system_id: str | None,
                       ctx: CallerContext, key: str) -> str:
    """The approval that opens this write, or WriteRefused.

    It must be approved, of the type this action needs, scoped to this employee and
    system, requested by this caller in this workflow request, not expired, and either
    unused or already consumed by this same idempotency key (a replay)."""
    row = conn.execute(
        "SELECT a.approval_id FROM approvals a "
        "JOIN approval_controls c ON a.approval_id = c.approval_id "
        "WHERE a.request_type = ? AND a.status = 'approved' "
        "  AND c.employee_id = ? AND c.system_id IS ? "
        "  AND c.requested_by = ? AND c.workflow_request_id = ? "
        "  AND c.expires_on >= ? "
        "  AND (c.consumed_key IS NULL OR c.consumed_key = ?) "
        "ORDER BY (c.consumed_key IS NULL), a.created_at LIMIT 1",
        (GATING_TYPE[action], employee_id, system_id, ctx.caller_id, ctx.request_id,
         config.today().isoformat(), key),
    ).fetchone()
    if row is None:
        raise WriteRefused(
            f"Refused: {action} for {employee_id}"
            f"{' on ' + system_id if system_id else ''} has no approved, unused approval "
            f"for request {ctx.request_id}. Nothing was written. An approval has to be "
            "requested and decided by its approver first."
        )
    return row["approval_id"]


def _reuse(conn, key: str, natural_sql: str, natural_args: tuple) -> str | None:
    """An existing row for this write: by idempotency key, else by its natural key (an
    open request for the same employee and system, such as AR001 or T001)."""
    hit = conn.execute(
        "SELECT entity_id FROM idempotency_keys WHERE idempotency_key = ?", (key,)
    ).fetchone()
    if hit:
        return hit["entity_id"]
    hit = conn.execute(natural_sql, natural_args).fetchone()
    return hit[0] if hit else None


def _consume(conn, approval_id: str, key: str) -> None:
    conn.execute(
        "UPDATE approval_controls SET consumed_key = ?, consumed_at = ? "
        "WHERE approval_id = ? AND consumed_key IS NULL",
        (key, config.now_iso(), approval_id),
    )


def _remember_key(conn, key: str, entity_type: str, entity_id: str, ctx: CallerContext):
    conn.execute(
        "INSERT INTO idempotency_keys VALUES (?, ?, ?, ?, ?)",
        (key, entity_type, entity_id, ctx.request_id, config.now_iso()),
    )


def _record_reuse(conn, key: str, entity_type: str, entity_id: str, approval_id: str,
                  ctx: CallerContext) -> None:
    """A released write that returns an existing row (AR001) still uses its approval.
    The first time this key is seen it is stored, the approval is consumed by it, and
    the reuse is audited. A replay with the same key finds the key already stored and
    writes nothing; the approval stays consumed by that key, so the replay is allowed
    and any other action is refused."""
    if conn.execute("SELECT 1 FROM idempotency_keys WHERE idempotency_key = ?",
                    (key,)).fetchone():
        return
    _remember_key(conn, key, entity_type, entity_id, ctx)
    _consume(conn, approval_id, key)
    _audit(conn, entity_type, entity_id, f"{entity_type}_reused", ctx.caller_id,
           f"Returned the existing {entity_id} under {approval_id}, request {ctx.request_id}.")


def create_access_request(conn, ctx: CallerContext, employee_id: str, software: str,
                          business_justification: str,
                          access_level: str = "standard_user") -> dict:
    justification = business_justification.strip()
    if not justification or len(justification) > 1000:
        raise ToolInputError("business_justification must be 1 to 1000 characters.")

    def work(c):
        _caller_known(c, ctx)
        db.require_employee(c, employee_id)
        system = db.resolve_system(c, software)
        key = f"create_access_request:{employee_id}:{system['system_id']}:{ctx.request_id}"
        approval_id = _matching_approval(
            c, "create_access_request", employee_id, system["system_id"], ctx, key)
        existing = _reuse(
            c, key,
            "SELECT request_id FROM access_requests WHERE employee_id = ? AND system_id = ? "
            f"AND status IN ({','.join('?' * len(OPEN_ACCESS_REQUEST))}) "
            "ORDER BY created_at LIMIT 1",
            (employee_id, system["system_id"], *OPEN_ACCESS_REQUEST),
        )
        if existing:
            _record_reuse(c, key, "access_request", existing, approval_id, ctx)
            row = db.get_access_requests(c, request_id=existing)[0]
            return {**row, "reused": True, "approval_id": approval_id}

        request_id = _next_id(c, "access_requests", "request_id", "AR")
        c.execute(
            "INSERT INTO access_requests (request_id, employee_id, system_id, access_level, "
            "business_justification, status, approver_id, created_at, decision_reason) "
            "VALUES (?, ?, ?, ?, ?, 'pending_approval', NULL, ?, NULL)",
            (request_id, employee_id, system["system_id"], access_level, justification,
             config.now_iso()),
        )
        _remember_key(c, key, "access_request", request_id, ctx)
        _consume(c, approval_id, key)
        _audit(c, "access_request", request_id, "access_request_filed", ctx.caller_id,
               f"Filed pending_approval under {approval_id}, request {ctx.request_id}.")
        row = db.get_access_requests(c, request_id=request_id)[0]
        return {**row, "reused": False, "approval_id": approval_id}

    return _transaction(conn, work)


def create_ticket(conn, ctx: CallerContext, employee_id: str, subject: str,
                  description: str, category: str, software: str = "",
                  priority: str = "medium") -> dict:
    if not subject.strip() or not category.strip():
        raise ToolInputError("subject and category are required.")

    def work(c):
        _caller_known(c, ctx)
        db.require_employee(c, employee_id)
        system_id = db.resolve_system(c, software)["system_id"] if software.strip() else None
        topic = system_id or f"{category.strip().lower()}:{subject.strip().lower()}"
        key = f"create_ticket:{employee_id}:{topic}:{ctx.request_id}"
        approval_id = _matching_approval(c, "create_ticket", employee_id, system_id, ctx, key)
        existing = None
        if system_id:
            existing = _reuse(
                c, key,
                "SELECT ticket_id FROM tickets WHERE employee_id = ? AND related_system_id = ? "
                f"AND status NOT IN ({','.join('?' * len(CLOSED_TICKET))}) "
                "ORDER BY created_at LIMIT 1",
                (employee_id, system_id, *CLOSED_TICKET),
            )
        else:
            existing = _reuse(c, key, "SELECT NULL WHERE 0", ())
        if existing:
            _record_reuse(c, key, "ticket", existing, approval_id, ctx)
            row = dict(c.execute("SELECT * FROM tickets WHERE ticket_id = ?", (existing,)).fetchone())
            return {**row, "reused": True, "approval_id": approval_id}

        ticket_id = _next_id(c, "tickets", "ticket_id", "T")
        team = TICKET_TEAMS.get(category.strip().lower(), "IT")
        c.execute(
            "INSERT INTO tickets (ticket_id, employee_id, category, subcategory, priority, "
            "status, subject, description, created_at, assigned_team, related_system_id) "
            "VALUES (?, ?, ?, '', ?, 'open', ?, ?, ?, ?, ?)",
            (ticket_id, employee_id, category.strip(), priority, subject.strip(),
             description.strip(), config.now_iso(), team, system_id),
        )
        _remember_key(c, key, "ticket", ticket_id, ctx)
        _consume(c, approval_id, key)
        _audit(c, "ticket", ticket_id, "ticket_created", ctx.caller_id,
               f"Created under {approval_id}, request {ctx.request_id}.")
        row = dict(c.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone())
        return {**row, "reused": False, "approval_id": approval_id}

    return _transaction(conn, work)


# request_approval -----------------------------------------------------------------

def _approval_scope(c, request_type: str, request_id: str, employee_id: str,
                    software: str) -> tuple[str, str | None]:
    """(employee, system) the approval is for. Taken from the record when request_id
    names an existing access request or ticket; otherwise from the arguments."""
    record = c.execute(
        "SELECT employee_id, system_id FROM access_requests WHERE request_id = ? "
        "UNION ALL SELECT employee_id, related_system_id FROM tickets WHERE ticket_id = ?",
        (request_id, request_id),
    ).fetchone()
    given_system = db.resolve_system(c, software)["system_id"] if software.strip() else None
    if record:
        if employee_id and employee_id != record[0]:
            raise ToolInputError(f"{request_id} is for {record[0]}, not {employee_id}.")
        if given_system and given_system != record[1]:
            raise ToolInputError(f"{request_id} is for system {record[1]}, not {given_system}.")
        return record[0], record[1]
    if not employee_id:
        raise ToolInputError(
            f"{request_id!r} is not an existing access request or ticket, so employee_id "
            "(and the system, for access) is required."
        )
    db.require_employee(c, employee_id)
    if request_type != "ticket" and given_system is None:
        raise ToolInputError(f"A {request_type} approval needs the system.")
    return employee_id, given_system


def request_approval(conn, ctx: CallerContext, request_type: str, request_id: str,
                     approver_id: str, employee_id: str = "", software: str = "") -> dict:
    """Create a pending approval row, or return the open one for (request_type,
    request_id) with reused: true. It never approves anything."""
    if request_type not in REQUEST_TYPES:
        raise ToolInputError(f"request_type must be one of {', '.join(REQUEST_TYPES)}.")
    if not request_id.strip():
        raise ToolInputError("request_id is required.")

    def work(c):
        _caller_known(c, ctx)
        db.require_employee(c, approver_id)
        subject, system_id = _approval_scope(c, request_type, request_id, employee_id, software)
        expires = (config.today() + timedelta(days=APPROVAL_TTL_DAYS)).isoformat()
        # A row this workflow request already holds comes first, whatever its status:
        # replaying the request after its write was released must find the consumed
        # AP001 again, not open a new approval for an action that already happened.
        existing = c.execute(
            "SELECT a.approval_id FROM approvals a "
            "LEFT JOIN approval_controls ctl ON a.approval_id = ctl.approval_id "
            "WHERE a.request_type = ? AND a.request_id = ? AND (a.status IN (?, ?) "
            "   OR (a.status = 'approved' AND ctl.consumed_at IS NULL) "
            "   OR (ctl.workflow_request_id = ? AND ctl.requested_by = ?)) "
            "ORDER BY (ctl.workflow_request_id IS NOT ? OR ctl.requested_by IS NOT ?), "
            "         a.created_at LIMIT 1",
            (request_type, request_id, *OPEN_APPROVAL, ctx.request_id, ctx.caller_id,
             ctx.request_id, ctx.caller_id),
        ).fetchone()
        if existing:
            approval_id, reused = existing["approval_id"], True
        else:
            approval_id, reused = _next_id(c, "approvals", "approval_id", "AP"), False
            c.execute(
                "INSERT INTO approvals (approval_id, request_type, request_id, approver_id, "
                "status, decision_reason, created_at) VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (approval_id, request_type, request_id, approver_id,
                 f"Requested by {ctx.caller_id} in request {ctx.request_id}.", config.now_iso()),
            )
            _audit(c, "approval", approval_id, "approval_requested", ctx.caller_id,
                   f"{request_type} {request_id}, approver {approver_id}.")
        # A seed row (AP001) has no control row until a workflow asks for it. The first
        # workflow binds it; later ones see the binding and cannot use it.
        c.execute(
            "INSERT OR IGNORE INTO approval_controls (approval_id, employee_id, system_id, "
            "requested_by, workflow_request_id, expires_on) VALUES (?, ?, ?, ?, ?, ?)",
            (approval_id, subject, system_id, ctx.caller_id, ctx.request_id, expires),
        )
        row = db.get_approvals(c, request_id=request_id)
        bound = c.execute(
            "SELECT workflow_request_id = ? AND requested_by = ? FROM approval_controls "
            "WHERE approval_id = ?", (ctx.request_id, ctx.caller_id, approval_id)).fetchone()
        return {**next(r for r in row if r["approval_id"] == approval_id), "reused": reused,
                "this_request": bool(bound[0])}

    return _transaction(conn, work)


# record_approval_decision: application code only -----------------------------------

def record_approval_decision(approval_id: str, decision: str, actor_id: str,
                             reason: str = "", conn: sqlite3.Connection | None = None) -> dict:
    """Set an open approval to approved or rejected. Only its named approver may decide.

    This is deliberately not an MCP tool: the model may propose an action, but only
    application code records that a human approved it (spec §6.6, Lesson 13)."""
    if decision not in ("approved", "rejected"):
        raise ToolInputError("decision must be 'approved' or 'rejected'.")
    own = conn is None
    conn = conn or db.connect()

    def work(c):
        row = c.execute("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)).fetchone()
        if row is None:
            raise ToolInputError(f"No approval {approval_id!r}.")
        if row["status"] not in OPEN_APPROVAL:
            raise WriteRefused(f"{approval_id} is already {row['status']}.")
        if row["approver_id"] != actor_id:
            raise WriteRefused(f"{approval_id} can only be decided by {row['approver_id']}.")
        now = config.now_iso()
        c.execute(
            "UPDATE approvals SET status = ?, decision_reason = ? WHERE approval_id = ?",
            (decision, reason or f"{decision} by {actor_id}", approval_id),
        )
        c.execute(
            "UPDATE approval_controls SET decided_by = ?, decided_at = ? WHERE approval_id = ?",
            (actor_id, now, approval_id),
        )
        _audit(c, "approval", approval_id, f"approval_{decision}", actor_id, reason or decision)
        return dict(c.execute("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)).fetchone())

    try:
        return _transaction(conn, work)
    finally:
        if own:
            conn.close()
