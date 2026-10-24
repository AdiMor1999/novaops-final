"""Where a confirmed access request goes, decided from records, not from the model.

S2 turn 10 and S6 turn 4 are the same request ("go ahead and file the Webex access
request") from the same caller (E004). Neither the intent nor the user group tells
them apart. The subject's employee record does:

- subject `preboarding` (E001, Maya Cohen, before her start date) → a Maya handoff
  through `WebexPort`. Maya owns no write; the Webex side decides and files.
- any other verified subject (E010 in S6/S8) → the Webex approval node, as in M2.
- no subject that a `get_employee` row identifies → the turn ends `needs_human` and
  nothing is handed off or requested.

The planner names the subject; code accepts it only when it matches a row a tool
returned. Task #5: when the thread has no such row yet (the Rachel scenario, or "go
ahead" for someone never looked up), the graph's `lookup_subject` node calls
`get_employee` with the id or name the planner named, from code, and routes on the
result. One match routes; no match or several matches end `needs_human`. A name or
an id the model made up matches nothing.

Starting code: none. The Lesson 10 homework (milestone 4) specifies the handoff and
its idempotency key; the routing rule is ours (README decisions table).
"""

import re
from typing import Literal

from agent.outcome import tool_results
from agent.policy import TurnPlan, scope_of
from agent.schemas import WebexHandoff

Route = Literal["handoff", "approval", "unverified"]

PREBOARDING = "preboarding"


def verified_subject(messages: list, employee_id: str | None) -> dict | None:
    """The `get_employee` row for this id, if a tool returned one in the thread."""
    if not employee_id:
        return None
    wanted = employee_id.strip().upper()
    for result in tool_results(messages, "get_employee"):
        for row in result.get("rows", []):
            if row.get("employee_id") == wanted:
                return row
    return None


def resolve_subject(messages: list, named: str | None, caller_id: str) -> str:
    """An employee id for the person a turn is about: the id the planner named, or the
    one `get_employee` row in the thread whose full name it named, else the caller
    ("am I getting a laptop?"). Used only to point a read at the right checklist."""
    if named and re.fullmatch(r"E\d{3}", named.strip().upper()):
        return named.strip().upper()
    if named:
        ids = {row["employee_id"] for result in tool_results(messages, "get_employee")
               for row in result.get("rows", [])
               if row.get("full_name", "").lower() == named.strip().lower()}
        if len(ids) == 1:
            return ids.pop()
    return caller_id


def verified_system(messages: list, software: str | None) -> dict | None:
    """{system_id, system_name} for the named software, from a row a tool returned in
    the thread (a subscription or an access request names both)."""
    if not software:
        return None
    wanted = software.strip().lower()
    for result in tool_results(messages):
        for row in result.get("rows", []):
            name, system_id = row.get("system_name"), row.get("system_id")
            if name and system_id and (name.lower() == wanted or system_id.lower() == wanted):
                return {"system_id": system_id, "system_name": name}
    return None


def system_named_in_message(messages: list) -> dict | None:
    """When the plan names no system, the one whose name the newest message contains,
    among systems a tool returned in this thread ("file the Webex access request")."""
    starts = [i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"]
    if not starts:
        return None
    text = str(messages[starts[-1]].content).lower()
    found = {}
    for result in tool_results(messages):
        for row in result.get("rows", []):
            name, system_id = row.get("system_name"), row.get("system_id")
            if name and system_id and re.search(rf"\b{re.escape(name.lower())}\b", text):
                found[system_id] = {"system_id": system_id, "system_name": name}
    return next(iter(found.values())) if len(found) == 1 else None


def subject_in_thread(messages: list, plan: TurnPlan) -> dict | None:
    """The subject's `get_employee` row: by the id the planner named, else the one row
    whose full name it named. Two people with that name verify nobody."""
    if row := verified_subject(messages, plan.subject_employee_id):
        return row
    if not plan.subject_name:
        return None
    wanted = plan.subject_name.strip().lower()
    found = {row["employee_id"]: row for result in tool_results(messages, "get_employee")
             for row in result.get("rows", []) if row.get("full_name", "").lower() == wanted}
    return next(iter(found.values())) if len(found) == 1 else None


def subject_query(plan: TurnPlan, messages: list) -> str | None:
    """What `lookup_subject` asks `get_employee` for: the id, else the name the planner
    named. None when no lookup is due: not a confirmed access request, the thread
    already verifies the subject, or the planner named nobody."""
    if not (plan.action_confirmed and plan.intent == "access_request"):
        return None
    if subject_in_thread(messages, plan) is not None:
        return None
    named = (plan.subject_employee_id or plan.subject_name or "").strip()
    return named or None


def lookup_outcome(rows: list[dict]) -> tuple[str | None, str]:
    """(the subject's id, why not) from the code lookup's rows."""
    if len(rows) == 1:
        return rows[0]["employee_id"], ""
    if not rows:
        return None, "no employee matches"
    names = ", ".join(f"{r['full_name']} ({r['employee_id']})" for r in rows)
    return None, f"several employees match: {names}"


def action_route(plan: TurnPlan, messages: list) -> Route | None:
    """None for every turn that is not a confirmed access request."""
    if not (plan.action_confirmed and plan.intent == "access_request"):
        return None
    subject = subject_in_thread(messages, plan)
    if subject is None:
        return "unverified"
    return "handoff" if subject.get("status") == PREBOARDING else "approval"


def route_scope(plan: TurnPlan, route: Route | None) -> str:
    """A handoff turn is Maya's: she proposes, the Webex side acts."""
    return "maya" if route == "handoff" else scope_of(plan)


def handoff_key(thread_id: str, employee_id: str, system_id: str) -> str:
    return f"handoff:{thread_id}:{employee_id}:{system_id}"


def justification(messages: list, subject: dict, system: dict) -> str:
    """The business reason, from the subject's onboarding task for this system when the
    thread has one (OT005), otherwise from their record."""
    for result in tool_results(messages, "list_onboarding_tasks"):
        for task in result.get("rows", []):
            if task.get("related_system_id") == system["system_id"]:
                return (f"Onboarding task {task['task_id']} for {subject['full_name']} "
                        f"({subject['employee_id']}): {task['description']}")
    return (f"{subject['full_name']} ({subject['employee_id']}), {subject.get('role')}, "
            f"starts {subject.get('start_date')} and needs {system['system_name']}.")


def build_handoff(messages: list, plan: TurnPlan, configurable: dict) -> WebexHandoff | str:
    """The typed handoff, or the reason it cannot be built."""
    subject = subject_in_thread(messages, plan)
    if subject is None:
        return f"No employee record in this conversation matches {plan.subject_employee_id!r}."
    system = verified_system(messages, plan.system) or (
        None if plan.system else system_named_in_message(messages))
    if system is None:
        return (f"No record in this conversation identifies the system {plan.system!r}; "
                "look up its subscription first.")
    thread_id = configurable["thread_id"]
    return WebexHandoff(
        caller_id=configurable["caller_id"],
        subject_employee_id=subject["employee_id"],
        system_id=system["system_id"],
        software=system["system_name"],
        justification=justification(messages, subject, system),
        source_thread_id=thread_id,
        request_id=configurable["request_id"],
        idempotency_key=handoff_key(thread_id, subject["employee_id"], system["system_id"]),
    )
