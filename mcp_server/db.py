"""SQLite access for the read tools, over the file at `NOVAOPS_DB_PATH`.

Starting code: Lesson 10 `server/db.py`. Changes: the database is a file built by
`python -m scripts.reseed` instead of `:memory:` (Lesson 10 lost every write on restart),
each call opens its own connection so the server and application code (approval
decisions) can share the file, dates come from `NOVAOPS_TODAY`, and the reads the
workflows need but Lesson 10 lacked are added: `get_access_requests`, `get_approvals`,
`get_team`, `is_manager` and `caller_access`.

Writes live in `writes.py`.
"""

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from agent import config

SCHEMA_FILE = "schema.sql"
SEED_FILE = "seed.sql"

EMPLOYEE_FIELDS = (
    "employee_id, full_name, email, role, department_id, manager_id, location, "
    "employment_type, status, start_date, end_date, user_group_id, systems_allowed_by_role"
)


class ToolInputError(ValueError):
    """Bad arguments or an unknown id. FastMCP returns the message as an is_error result,
    so the text is written for the model to act on."""


class DatabaseMissing(RuntimeError):
    """The database file does not exist. Run `python -m scripts.reseed`."""


def connect(path: Path | None = None) -> sqlite3.Connection:
    """A new connection in autocommit mode, so writes control their own transaction
    with an explicit BEGIN IMMEDIATE."""
    path = path or config.db_path()
    if not path.exists():
        raise DatabaseMissing(f"No database at {path}. Run `python -m scripts.reseed`.")
    conn = sqlite3.connect(path, isolation_level=None, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(path: Path | None = None):
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def rows(cursor) -> list[dict]:
    return [dict(row) for row in cursor.fetchall()]


def _decode_systems(employee: dict) -> dict:
    raw = employee.get("systems_allowed_by_role")
    if isinstance(raw, str):
        try:
            employee["systems_allowed_by_role"] = json.loads(raw)
        except json.JSONDecodeError:
            pass  # keep the raw string; it is still readable
    return employee


def require_employee(conn, employee_id: str) -> dict:
    row = conn.execute(
        f"SELECT {EMPLOYEE_FIELDS} FROM employees WHERE employee_id = ?", (employee_id,)
    ).fetchone()
    if row is None:
        raise ToolInputError(
            f"No employee with id {employee_id!r}. Look the person up with get_employee first."
        )
    return dict(row)


def resolve_system(conn, software: str) -> dict:
    """Exact system or vendor name first, then a unique partial match. An ambiguous
    name is an error, not a guess."""
    name = software.strip()
    if not name:
        raise ToolInputError("A system name is required, e.g. 'Webex'.")
    exact = conn.execute(
        "SELECT sys.system_id, sys.system_name FROM systems sys "
        "JOIN vendors v ON sys.vendor_id = v.vendor_id "
        "WHERE lower(sys.system_name) = lower(?) OR sys.system_id = ? "
        "   OR lower(v.vendor_name) = lower(?)",
        (name, name, name),
    ).fetchall()
    if len(exact) == 1:
        return dict(exact[0])
    partial = conn.execute(
        "SELECT system_id, system_name FROM systems WHERE lower(system_name) LIKE lower(?)",
        (f"%{name}%",),
    ).fetchall()
    if len(partial) == 1:
        return dict(partial[0])
    if not partial and not exact:
        raise ToolInputError(f"No system matching {software!r}.")
    names = ", ".join(sorted(r["system_name"] for r in (exact or partial)))
    raise ToolInputError(f"{software!r} matches several systems ({names}). Use the exact name.")


# Reads behind the MCP tools ----------------------------------------------------

def get_employee(conn, query: str) -> list[dict]:
    """Every match by id, email or partial name. The seed has two Daniels, two Mayas and
    two Saras, so a first name can return more than one person."""
    if not query.strip():
        raise ToolInputError("Give an employee id, email or name.")
    found = rows(conn.execute(
        f"SELECT {EMPLOYEE_FIELDS} FROM employees "
        "WHERE employee_id = ? OR lower(email) = lower(?) OR lower(full_name) LIKE lower(?) "
        "ORDER BY full_name",
        (query, query, f"%{query}%"),
    ))
    return [_decode_systems(e) for e in found]


def _name_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3]


def similar_employees(conn, query: str) -> list[dict]:
    """The fallback when no name contains the whole query. Users type 'rrachel stein'
    (S8) and 'rrachel steins' (S7); a substring match finds nobody, and the model then
    reports that Rachel does not exist. A name word counts when it sits inside a query
    word or the other way round; only the employees with the most such words return,
    so 'rrachel steins' gives Rachel Stein and not Rachel Green."""
    words = _name_words(query)
    if not words:
        return []
    scored = []
    for row in rows(conn.execute(f"SELECT {EMPLOYEE_FIELDS} FROM employees ORDER BY full_name")):
        score = sum(any(t in w or w in t for w in words) for t in _name_words(row["full_name"]))
        if score:
            scored.append((score, row))
    best = max((score for score, _ in scored), default=0)
    return [_decode_systems(row) for score, row in scored if score == best]


def check_software_subscription(conn, software: str, today: date) -> list[dict]:
    """Seat counts per subscription. Seat assignments per employee are not recorded
    anywhere in the database, so every result says so (spec §13, W-I-02)."""
    like = f"%{software}%"
    found = rows(conn.execute(
        "SELECT sub.subscription_id, sys.system_id, sys.system_name, v.vendor_name, "
        "       sub.seat_limit, sub.active_seats, sub.status, sub.renewal_date, "
        "       sub.annual_cost, sub.contract_id, sub.owner_department_id "
        "FROM software_subscriptions sub "
        "JOIN systems sys ON sub.system_id = sys.system_id "
        "JOIN vendors v ON sub.vendor_id = v.vendor_id "
        "WHERE lower(sys.system_name) LIKE lower(?) OR lower(v.vendor_name) LIKE lower(?) "
        "ORDER BY sys.system_name",
        (like, like),
    ))
    for sub in found:
        sub["seats_available"] = sub["seat_limit"] - sub["active_seats"]
        sub["over_limit"] = sub["active_seats"] > sub["seat_limit"]
        sub["at_or_over_limit"] = sub["active_seats"] >= sub["seat_limit"]
        sub["days_until_renewal"] = (date.fromisoformat(sub["renewal_date"]) - today).days
        sub["seat_assignments_recorded"] = False
        sub["seat_assignments_note"] = (
            "The database records only seat counts. It has no record of which employee "
            "holds a seat. systems_allowed_by_role shows entitlement by role, not an "
            "assigned seat."
        )
    return found


def list_onboarding_tasks(conn, employee_id: str, today: date) -> list[dict]:
    require_employee(conn, employee_id)
    tasks = rows(conn.execute(
        "SELECT task_id, task_type, description, owner_group, status, due_date, "
        "       related_system_id "
        "FROM onboarding_tasks WHERE employee_id = ? ORDER BY due_date, task_id",
        (employee_id,),
    ))
    for task in tasks:
        task["overdue"] = (
            task["status"] not in ("done", "completed")
            and date.fromisoformat(task["due_date"]) < today
        )
    return tasks


def check_asset_inventory(conn, asset_type: str = "", location: str = "") -> list[dict]:
    return rows(conn.execute(
        "SELECT asset_id, asset_type, model, status, location, employee_id, notes "
        "FROM assets WHERE lower(asset_type) LIKE lower(?) AND lower(location) LIKE lower(?) "
        "ORDER BY asset_type, asset_id",
        (f"%{asset_type}%", f"%{location}%"),
    ))


def list_employee_tickets(conn, employee_id: str, status: str = "") -> list[dict]:
    require_employee(conn, employee_id)
    return rows(conn.execute(
        "SELECT ticket_id, category, subcategory, priority, status, subject, description, "
        "       created_at, assigned_team, related_system_id "
        "FROM tickets WHERE employee_id = ? AND lower(status) LIKE lower(?) "
        "ORDER BY created_at DESC",
        (employee_id, f"%{status}%"),
    ))


def get_access_requests(conn, employee_id: str = "", request_id: str = "") -> list[dict]:
    if not (employee_id or request_id):
        raise ToolInputError("Give an employee_id or a request_id.")
    if employee_id:
        require_employee(conn, employee_id)
    return rows(conn.execute(
        "SELECT ar.request_id, ar.employee_id, ar.system_id, sys.system_name, "
        "       ar.access_level, ar.business_justification, ar.status, ar.approver_id, "
        "       ar.created_at, ar.decision_reason "
        "FROM access_requests ar JOIN systems sys ON ar.system_id = sys.system_id "
        "WHERE (? = '' OR ar.employee_id = ?) AND (? = '' OR ar.request_id = ?) "
        "ORDER BY ar.created_at",
        (employee_id, employee_id, request_id, request_id),
    ))


def get_approvals(conn, request_id: str = "", employee_id: str = "") -> list[dict]:
    """Approval rows for one request id, or for every request about one employee
    (their access requests and tickets, and approvals requested through this server)."""
    if not (employee_id or request_id):
        raise ToolInputError("Give a request_id or an employee_id.")
    if employee_id:
        require_employee(conn, employee_id)
    return rows(conn.execute(
        "SELECT a.approval_id, a.request_type, a.request_id, a.approver_id, a.status, "
        "       a.decision_reason, a.created_at, c.expires_on, "
        "       (c.consumed_at IS NOT NULL) AS consumed "
        "FROM approvals a LEFT JOIN approval_controls c ON a.approval_id = c.approval_id "
        "WHERE (? = '' OR a.request_id = ?) AND (? = '' "
        "   OR a.request_id IN (SELECT request_id FROM access_requests WHERE employee_id = ?) "
        "   OR a.request_id IN (SELECT ticket_id FROM tickets WHERE employee_id = ?) "
        "   OR c.employee_id = ?) "
        "ORDER BY a.created_at, a.approval_id",
        (request_id, request_id, employee_id, employee_id, employee_id, employee_id),
    ))


def get_team(conn, manager_id: str) -> list[dict]:
    require_employee(conn, manager_id)
    return [_decode_systems(e) for e in rows(conn.execute(
        f"SELECT {EMPLOYEE_FIELDS} FROM employees WHERE manager_id = ? ORDER BY full_name",
        (manager_id,),
    ))]


@dataclass(frozen=True)
class CallerAccess:
    """What one caller may retrieve. Read from the database, never from `_meta`: the
    user group in `_meta` is what the agent claims, the row is what NovaOps records."""

    caller_id: str
    manager: bool
    hr: bool
    reports: tuple[str, ...] = ()

    @property
    def employment_subjects(self) -> tuple[str, ...] | None:
        """Subjects whose employment documents this caller may read. None means every
        subject (HR). Otherwise the caller and their direct reports."""
        if self.hr:
            return None
        return (self.caller_id, *self.reports)


def caller_access(caller_id: str, path: Path | None = None) -> CallerAccess:
    """Manager status from `manager_id` (spec §12), HR status from the caller's own
    `user_group_id`. An unknown caller, a missing database or a query error give no
    manager rights, no HR rights and no reports, so the filter fails closed: at most
    the caller's own employment documents stay readable."""
    try:
        with session(path) as conn:
            row = conn.execute("SELECT user_group_id FROM employees WHERE employee_id = ?",
                               (caller_id,)).fetchone()
            reports = tuple(r[0] for r in conn.execute(
                "SELECT employee_id FROM employees WHERE manager_id = ? ORDER BY employee_id",
                (caller_id,)))
    except (sqlite3.Error, DatabaseMissing):
        return CallerAccess(caller_id, manager=False, hr=False)
    if row is None:
        return CallerAccess(caller_id, manager=False, hr=False)
    return CallerAccess(caller_id, manager=bool(reports), hr=row[0] == "UG_HR", reports=reports)


def is_manager(caller_id: str, path: Path | None = None) -> bool:
    """True when some employee's manager_id is this caller (spec §12). user_group is not
    used: E018 manages people and is UG_REGULAR. Fails closed like caller_access()."""
    return caller_access(caller_id, path).manager


def employee_names(path: Path | None = None) -> dict[str, str]:
    with session(path) as conn:
        return {r["full_name"]: r["employee_id"] for r in conn.execute(
            "SELECT employee_id, full_name FROM employees"
        )}


def table_counts(conn) -> dict[str, int]:
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    )]
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
