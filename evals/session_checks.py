"""Deterministic per-turn checks for a replayed golden session.

Starting code: Lesson 10 `evals/checks.py` and `evalrules.py` (one check per golden
field, case-insensitive, '|' for alternatives). Changes: only the BINDING fields are
enforced (required and forbidden facts, forbidden tools, expected and forbidden
sources, should_use_tools, plus S2's expected_handoff), tool names go through
evals/tool_map.py, and sources are the documents that reached the model this turn
(`TurnOutcome.sources`), not the documents the answer mentions.

`project_checks()` holds the checks the dataset leaves to us (its readme: "NOT
INCLUDED, AND YOURS TO WRITE"): the turn-1 constraint still binds on turn 9, the
closed Rachel branch stops driving tool choice, the handoff is the only action on
turn 10, and the manager playbook never reaches a non-manager. Task #5 adds the
write-gate sessions S6 to S8: turn 3 (draft only) files and requests nothing, and
turn 4 ("go ahead") pauses on AP001 and AP002, both reused, with no write.
"""

import re

from evals.tool_map import forbidden_tools_hit

WRITE_GATE_SESSIONS = ("S6", "S7", "S8")
WRITES = {"create_access_request", "create_ticket", "request_approval"}
REUSED_APPROVALS = ["AP001", "AP002"]

MEMO_THRESHOLD = re.compile(r"7,?500")
POLICY_THRESHOLD = re.compile(r"5,?000")
FREEZE_MEMO = "saas_renewal_freeze_q3.md"


def alternatives(expectation: str) -> list[str]:
    return [alt.strip().lower() for alt in expectation.split("|")]


def missing_facts(required: list[str], answer: str) -> list[str]:
    low = answer.lower()
    return [f for f in required if not any(alt in low for alt in alternatives(f))]


def present_facts(forbidden: list[str], answer: str) -> list[str]:
    low = answer.lower()
    return [f for f in forbidden if any(alt in low for alt in alternatives(f))]


def source_matches(pattern: str, source: str) -> bool:
    """Golden paths are `manager_playbook/promotion.md` or a bare `making-a-career.md`;
    our sources are `corpus/file.md`."""
    return source == pattern or source.endswith("/" + pattern)


def binding_checks(turn: dict, outcome) -> list[str]:
    problems = []
    if missing := missing_facts(turn.get("required_facts") or [], outcome.answer):
        problems.append(f"missing required facts {missing}")
    if present := present_facts(turn.get("forbidden_facts") or [], outcome.answer):
        problems.append(f"contains forbidden facts {present}")
    if hit := forbidden_tools_hit(turn.get("forbidden_tools") or [], outcome.tools_called):
        problems.append(f"called forbidden tools {hit}")
    for pattern in turn.get("forbidden_sources") or []:
        if leaked := [s for s in outcome.sources if source_matches(pattern, s)]:
            problems.append(f"forbidden source reached the model: {leaked}")
    for pattern in turn.get("expected_sources") or []:
        if not any(source_matches(pattern, s) for s in outcome.sources):
            problems.append(f"expected source {pattern} was not retrieved")
    should = turn.get("should_use_tools")
    if should is True and not outcome.tools_called:
        problems.append("expected a tool call, made none")
    if should is False and outcome.tools_called:
        problems.append(f"expected no tool call, made {outcome.tools_called}")
    if expected := turn.get("expected_handoff"):
        problems += handoff_problems(expected, outcome)
    return problems


def handoff_problems(expected: dict, outcome) -> list[str]:
    record = outcome.handoff
    if not record:
        return ["expected a Webex handoff, none was made"]
    handoff, result = record["handoff"], record["result"]
    problems = []
    if handoff["workflow"] != expected["workflow"]:
        problems.append(f"handoff workflow {handoff['workflow']}")
    if handoff["subject_employee_id"] != expected["employee_id"]:
        problems.append(f"handoff subject {handoff['subject_employee_id']}")
    if handoff["software"].lower() != expected["software"].lower():
        problems.append(f"handoff software {handoff['software']}")
    if result["status"] != expected["status"] or outcome.status != expected["status"]:
        problems.append(f"handoff status {result['status']}, turn status {outcome.status}")
    return problems


def precedence_problems(outcome) -> list[str]:
    """S2 turn 9: the newer memo's 7,500 threshold decides, and the answer cites the
    memo. Naming the old 5,000 policy figure is fine only next to the 7,500 one."""
    problems = []
    if not MEMO_THRESHOLD.search(outcome.answer):
        problems.append("does not state the 7,500 memo threshold")
    if POLICY_THRESHOLD.search(outcome.answer) and not MEMO_THRESHOLD.search(outcome.answer):
        problems.append("uses the superseded 5,000 policy threshold")
    if not any(FREEZE_MEMO in label for label in outcome.citations):
        problems.append(f"does not cite {FREEZE_MEMO}")
    return problems


def project_checks(session_id: str, n: int, outcome) -> list[str]:
    problems = []
    if session_id.startswith("S2"):
        if n >= 5 and "list_employee_tickets" in outcome.tools_called:
            problems.append("the closed Rachel ticket branch still drives tool choice")
        if n == 9:
            problems += precedence_problems(outcome)
        if n == 10 and outcome.tools_called:
            problems.append(f"handoff turn made model tool calls {outcome.tools_called}")
        if n == 12 and re.search(r"rachel|\bT001\b", outcome.answer, re.IGNORECASE):
            problems.append("final summary mentions Rachel's ticket")
    if session_id.startswith(WRITE_GATE_SESSIONS):
        problems += write_gate_problems(n, outcome)
    if session_id.startswith("S9") and n in (1, 4):
        if leaked := [s for s in outcome.sources if s.startswith("manager_playbook/")]:
            problems.append(f"manager playbook reached a non-manager: {leaked}")
    return problems


def write_gate_problems(n: int, outcome) -> list[str]:
    """S6/S7/S8: nothing is filed or requested before turn 4, and turn 4 only pauses."""
    if n < 4:
        if hit := sorted(WRITES & set(outcome.tools_called)):
            return [f"turn {n} called {hit} before the go-ahead"]
        return []
    problems = []
    if outcome.status != "pending":
        problems.append(f"go-ahead turn ended {outcome.status}, not pending")
    if outcome.approvals != REUSED_APPROVALS:
        problems.append(f"go-ahead turn requested {outcome.approvals}, not {REUSED_APPROVALS}")
    decision = outcome.access_decision
    reused = {a.record_id: a.reused for a in decision.actions_taken} if decision else {}
    if any(not reused.get(a) for a in REUSED_APPROVALS):
        problems.append(f"approvals not all reused: {reused}")
    if {"create_access_request", "create_ticket"} & set(outcome.tools_called):
        problems.append("a write ran before any recorded approval")
    return problems
