"""Read-only database access for the page: the caller list and the approvals panel.

The connection opens the file with SQLite's `mode=ro`, so nothing in this module can
write even by mistake. Every write the chat causes goes through `run_turn()` and the
MCP server, or through `record_approval_decision()` in `web/approvals.py`.
"""

import sqlite3
from contextlib import contextmanager

from web import settings

# Seed rows use 'needed', rows request_approval creates use 'pending'. Both wait on a
# decision (mcp_server/writes.py, OPEN_APPROVAL).
OPEN_STATUSES = ("needed", "pending")


class UnknownCaller(LookupError):
    """The caller id is not a seeded employee."""


@contextmanager
def read_only():
    path = settings.database()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def callers() -> list[dict]:
    with read_only() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT e.employee_id, e.full_name, e.role, e.user_group_id, e.status, "
            "       d.name AS department "
            "FROM employees e LEFT JOIN departments d ON d.department_id = e.department_id "
            "ORDER BY e.employee_id")]


def user_group(caller_id: str) -> str:
    """The caller's group from their employee row. The page never sends a group."""
    with read_only() as conn:
        row = conn.execute("SELECT user_group_id FROM employees WHERE employee_id = ?",
                           (caller_id,)).fetchone()
    if row is None:
        raise UnknownCaller(caller_id)
    return row["user_group_id"]


def open_approvals(approver_id: str) -> list[dict]:
    """Open approvals whose named approver is this caller, with the subject and system
    of the record each one is for."""
    with read_only() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT a.approval_id, a.request_type, a.request_id AS record_id, a.status, "
            "       a.decision_reason, a.created_at, "
            "       COALESCE(ar.employee_id, t.employee_id) AS subject_id, "
            "       e.full_name AS subject_name, s.system_name "
            "FROM approvals a "
            "LEFT JOIN access_requests ar ON ar.request_id = a.request_id "
            "LEFT JOIN tickets t ON t.ticket_id = a.request_id "
            "LEFT JOIN employees e ON e.employee_id = COALESCE(ar.employee_id, t.employee_id) "
            "LEFT JOIN systems s ON s.system_id = COALESCE(ar.system_id, t.related_system_id) "
            f"WHERE a.approver_id = ? AND a.status IN ({','.join('?' * len(OPEN_STATUSES))}) "
            "ORDER BY a.created_at, a.approval_id",
            (approver_id, *OPEN_STATUSES))]
