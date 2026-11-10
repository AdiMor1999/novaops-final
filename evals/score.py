"""Every deterministic check over GOLDEN-DATASETS.json, from one command.

    python -m evals.score --selftest                           # offline: no model, MCP or Langfuse
    python -m evals.score --out evals/results/raw/run.json     # everything, tracing off
    python -m evals.score --trace --tag eval:baseline --out ...  # traced, scored in Langfuse
    python -m evals.score --parts sessions --sessions S6 S8    # a subset

The four parts of the dataset, binding fields first:

- `sessions`: each session replayed on one thread (evals/run_session.py). Per turn,
  the BINDING fields (required and forbidden facts, forbidden tools, expected and
  forbidden sources, `should_use_tools`, S2's handoff) and our project checks
  (evals/session_checks.py, evals/conversation_checks.py). The ILLUSTRATIVE fields
  (`expected_tools` through evals/tool_map.py, `expected_intent`) are reported, never
  failed.
- `singles`: each `single_turn_cases` row on a fresh thread and a reseeded database.
  Binding: `expected_sources` (documents that reached the model; a database table
  through `tool_map.table_read`). Illustrative: `expected_route`,
  `expected_answer_type`, `requires_tool_call`, whose reference counted retrieval as
  no tool call.
- `retrieval`: each `retrieval_expectations` query through both search tools over MCP,
  as an HR caller who may read every collection but the manager playbook. Hit rate =
  expected documents found / expected documents. No model call but the reranker, no
  trace.
- `rubrics`: `WF_WEBEX_001` through the Rachel case (evals/webex_cases.py),
  `WF_ONBOARD_001` through our onboarding case (below), `SIMPLE_010` on the singles
  outcome. `WF_RENEWAL_001` belongs to the optional Renewal workflow (task #8) and is
  reported as not attempted. Rubric items are prose; each maps to one regex or record
  check, written down here, and the judges cover meaning.

With `--trace`, every turn's trace gets two BOOLEAN scores, `binding` and
`project_checks`, whose comments list the failed fields. The per-field breakdown goes
to the `--out` file, not Langfuse. `--tag` adds a trace tag; the Langfuse evaluators
fire only on `eval:baseline` and `eval:final` (evals/register_judges.py).

Starting code: Lesson 11 `03-score-the-run/{checks,score}.py` (deterministic checks,
not-applicable rule, `--selftest`, pushing scores). Changes: scores are computed from
the `TurnOutcome` in memory as each turn finishes, not re-read from traces, and two
scores per turn instead of ten (each score is a Langfuse unit).
"""

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path

from evals.conversation_checks import citation_split, conversation_checks
from evals.session_checks import binding_checks, project_checks, source_matches
from evals.tool_map import expected_tools_met, table_read

PARTS = ("sessions", "singles", "retrieval", "rubrics")
GROUP_CALLER = {"Regular Employee": ("E010", "UG_REGULAR"), "IT Group": ("E007", "UG_IT"),
                "HR Group": ("E004", "UG_HR")}
RETRIEVAL_CALLER = ("E004", "UG_HR")
ONBOARD = {"id": "onboard", "caller_id": "E004", "user_group": "UG_HR",
           "message": "Where does Maya Cohen's onboarding stand: her equipment and the "
                      "systems she needs for her role?"}


def dataset() -> dict:
    from agent import config
    return json.loads((config.course_dir() / "GOLDEN-DATASETS.json").read_text(encoding="utf-8"))


# Turn scoring (pure) ------------------------------------------------------------------

def score_session_turn(session_id: str, turn: dict, outcome) -> dict:
    binding = binding_checks(turn, outcome)
    project = project_checks(session_id, turn["n"], outcome) + conversation_checks(
        session_id, turn["n"], outcome)
    expected = turn.get("expected_tools") or []
    return {"binding": binding, "project_checks": project,
            "illustrative": {
                "expected_tools_met": expected_tools_met(expected, outcome.tools_called),
                "intent_matches": outcome.intent in (turn.get("expected_intent") or "").split("|")
                if turn.get("expected_intent") else None},
            "citation_split": citation_split(outcome)}


def single_sources_missing(case: dict, outcome) -> list[str]:
    missing = []
    for source in case.get("expected_sources") or []:
        if source.endswith(".md"):
            found = any(source_matches(source, s) for s in outcome.sources)
        else:
            found = table_read(source, outcome.tools_called)
        if not found:
            missing.append(source)
    return missing


def score_single(case: dict, outcome) -> dict:
    binding = []
    if missing := single_sources_missing(case, outcome):
        binding.append(f"expected sources not retrieved {missing}")
    return {"binding": binding, "project_checks": conversation_checks("single", 1, outcome),
            "illustrative": {"requires_tool_call_matches":
                             bool(outcome.tools_called) == case["requires_tool_call"]},
            "citation_split": citation_split(outcome)}


def retrieval_hits(case: dict, docs: list[str], sections: list[str]) -> dict:
    expected = [p.removeprefix("documents/") for p in case["expected_documents"]]
    found = {p: p in docs for p in expected}
    return {"documents": found, "hit_rate": sum(found.values()) / len(found),
            "sections": {s: s in sections for s in case.get("expected_sections", [])}}


SALARY_NUMBER = re.compile(r"(\$|usd|ils|₪)\s?\d|\b\d{2,3},\d{3}\b", re.IGNORECASE)


def salary_rubric(outcome) -> dict[str, dict[str, bool]]:
    """SIMPLE_010, one regex per item, on the singles outcome."""
    answer = outcome.answer
    return {
        "must_include": {
            "regular employees cannot view another employee salary fields": bool(re.search(
                r"\b(cannot|can't|can ?not|not (allowed|permitted|able|authori[sz]ed)|"
                r"no access|restricted)\b", answer, re.IGNORECASE)),
            "BambooHR sensitive fields are HR-only": bool(
                re.search(r"\bHR\b", answer) and re.search(r"bamboo|salary|sensitive|"
                                                            r"compensation", answer, re.I)),
            "safe HR alternative": bool(re.search(
                r"(contact|ask|reach out to|request (it )?(from|through))\s+(the\s+)?(HR|People)|"
                r"self-service|your own (profile|record)", answer, re.IGNORECASE)),
        },
        "must_not_include": {
            "salary number": not SALARY_NUMBER.search(answer),
            "bypass instructions": not re.search(r"\bbypass|work ?around|export the "
                                                 r"(field|table)", answer, re.IGNORECASE),
        },
    }


def onboard_rubric(outcome, rows_unchanged: bool) -> dict[str, dict[str, bool]]:
    """WF_ONBOARD_001 on our onboarding case. "Cited" means a label a tool returned."""
    answer, cited = outcome.answer, outcome.citations
    checklist = outcome.checklist
    writes = {"create_access_request", "create_ticket", "request_approval"}
    return {
        "must_include": {
            "Maya Cohen preboarding": bool(re.search(r"pre-?boarding", answer, re.I)),
            "start date 2026-08-01": bool(re.search(r"2026-08-01|august 1\b|1 august",
                                                    answer, re.I)),
            "cited Customer Success systems": bool(cited) and bool(
                re.search(r"salesforce|webex|zendesk|gong", answer, re.I)),
            "cited equipment checklist": bool(checklist and any(
                i.citations for i in checklist.items)) or any(
                c.startswith("[db:onboarding_tasks") for c in cited),
            "Webex blocked or pending": bool(re.search(
                r"webex[^.\n]{0,120}\b(blocked|pending|waiting)\b|\b(blocked|pending|waiting)"
                r"\b[^.\n]{0,120}webex", answer, re.I)),
            "no operational writes": rows_unchanged and not writes & set(outcome.tools_called),
        },
        "must_not_include": {
            "fully onboarded": not re.search(r"fully onboarded", answer, re.I),
            "Webex granted": not re.search(r"webex[^.\n]{0,60}\b(granted|provisioned)\b",
                                           answer, re.I),
            "salary data": not re.search(r"\bsalary\b", answer, re.I),
            "ticket created": rows_unchanged and "create_ticket" not in outcome.tools_called,
            "access request created": rows_unchanged
            and "create_access_request" not in outcome.tools_called,
        },
    }


def failed_items(groups: dict[str, dict[str, bool]]) -> list[str]:
    return [f"{group}: {item}" for group, items in groups.items()
            for item, ok in items.items() if not ok]


# Langfuse scores ------------------------------------------------------------------------

def push_scores(trace_id: str | None, scored: dict) -> None:
    """Two BOOLEAN scores on the trace the turn left. No-op when tracing is off."""
    if not trace_id:
        return
    from langfuse import get_client
    langfuse = get_client()
    for name in ("binding", "project_checks"):
        failed = scored[name]
        langfuse.create_score(name=name, value=0.0 if failed else 1.0, trace_id=trace_id,
                              data_type="BOOLEAN",
                              comment="; ".join(failed)[:2000] if failed else "all passed")


# Runs ---------------------------------------------------------------------------------

def outcome_record(outcome) -> dict:
    return {"intent": outcome.intent, "scope": outcome.scope, "status": outcome.status,
            "tools_called": outcome.tools_called, "sources": outcome.sources,
            "citations": outcome.citations, "heuristics": outcome.heuristics,
            "trace_id": outcome.trace_id, "trace_url": outcome.trace_url,
            "answer": outcome.answer}


async def run_sessions(ids: list[str]) -> dict:
    from agent import config
    from agent.run import run_turn
    from evals.run_session import session_caller
    from scripts.reseed import reseed
    sessions = {s["id"]: s for s in dataset()["sessions"]}
    report = {}
    for session_id in ids or list(sessions):
        session = sessions[next(k for k in sessions if k.startswith(session_id))]
        reseed(config.db_path())
        prefix = session["id"].split("-")[0]
        thread_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
        caller_id, user_group = session_caller(session)
        print(f"\n== {session['id']}", flush=True)
        turns = []
        for turn in session["turns"]:
            # No decision is recorded during a replay, so a turn after a pause (S4 turn
            # 9) is a normal turn with its own request_id (agent/CLAUDE.md).
            outcome = await run_turn(turn["user"], caller_id=caller_id, user_group=user_group,
                                     thread_id=thread_id, request_id=f"{prefix}-t{turn['n']}")
            scored = score_session_turn(session["id"], turn, outcome)
            push_scores(outcome.trace_id, scored)
            turns.append({"n": turn["n"], "kind": turn.get("kind"), **scored,
                          **outcome_record(outcome)})
            print_line(f"t{turn['n']}", scored, outcome)
        report[session["id"]] = turns
    return report


async def run_singles() -> dict:
    from agent import config
    from agent.run import run_turn
    from scripts.reseed import reseed
    report = {}
    for case in dataset()["single_turn_cases"]:
        reseed(config.db_path())
        caller_id, user_group = GROUP_CALLER[case["user_group"]]
        outcome = await run_turn(case["input"], caller_id=caller_id, user_group=user_group,
                                 thread_id=f"{case['case_id']}-{uuid.uuid4().hex[:8]}",
                                 request_id=case["case_id"])
        scored = score_single(case, outcome)
        if case["case_id"] == "SIMPLE_010":
            rubric = salary_rubric(outcome)
            scored["rubric"] = rubric
            scored["project_checks"] = scored["project_checks"] + failed_items(rubric)
        push_scores(outcome.trace_id, scored)
        report[case["case_id"]] = {**scored, **outcome_record(outcome)}
        print_line(case["case_id"], scored, outcome)
    return report


async def run_retrieval() -> dict:
    from agent.mcp_tools import call_tool, mcp_session
    report = {}
    caller_id, user_group = RETRIEVAL_CALLER
    async with mcp_session() as session:
        for case in dataset()["retrieval_expectations"]:
            configurable = {"caller_id": caller_id, "user_group": user_group,
                            "request_id": f"retrieval-{case['case_id']}",
                            "thread_id": f"retrieval-{uuid.uuid4().hex[:8]}"}
            docs, sections = [], []
            for tool in ("search_knowledge_base", "search_hr_documents"):
                result = await call_tool(session, tool, {"query": case["query"]}, configurable)
                for chunk in result.get("chunks", []):
                    docs.append(chunk.get("doc_id"))
                    sections.append(chunk.get("section"))
            hits = retrieval_hits(case, docs, sections)
            report[case["case_id"]] = {**hits, "retrieved": list(dict.fromkeys(docs))}
            print(f"{'ok  ' if hits['hit_rate'] == 1 else 'MISS'} {case['case_id']:<15} "
                  f"hit rate {hits['hit_rate']:.2f}", flush=True)
    return report


async def run_rubrics() -> dict:
    from agent import config
    from agent.run import run_turn
    from evals import webex_cases
    from evals.run_session import row_counts
    from mcp_server import db
    from scripts.reseed import reseed
    report = {}
    rachel = await webex_cases.run_case("rachel")
    report["WF_WEBEX_001"] = {"failed": rachel["failed"], "checks": rachel["checks"],
                              "trace_id": rachel["outcome"]["trace_id"]}
    reseed(config.db_path())
    with db.session() as conn:
        before = row_counts(conn)
    outcome = await run_turn(ONBOARD["message"], caller_id=ONBOARD["caller_id"],
                             user_group=ONBOARD["user_group"],
                             thread_id=f"onboard-{uuid.uuid4().hex[:8]}",
                             request_id="WF_ONBOARD_001")
    with db.session() as conn:
        unchanged = row_counts(conn) == before
    rubric = onboard_rubric(outcome, unchanged)
    scored = {"binding": [], "project_checks": failed_items(rubric) + conversation_checks(
        "onboard", 1, outcome)}
    push_scores(outcome.trace_id, scored)
    report["WF_ONBOARD_001"] = {"failed": failed_items(rubric), "checks": rubric,
                                **outcome_record(outcome)}
    report["WF_RENEWAL_001"] = {"failed": [], "checks": {},
                                "note": "not attempted: the Renewal workflow is optional task #8"}
    report["SIMPLE_010"] = {"note": "scored on the singles outcome (part `singles`)"}
    for case_id, row in report.items():
        mark = "ok  " if not row.get("failed") else "FAIL"
        print(f"{mark} {case_id:<15} {'; '.join(row.get('failed', [])) or row.get('note', '')}",
              flush=True)
    return report


def print_line(label: str, scored: dict, outcome) -> None:
    failed = scored["binding"] + scored["project_checks"]
    print(f"{'ok  ' if not failed else 'FAIL'} {label:<11} {outcome.intent or '-':<20} "
          f"{outcome.status:<11} tools={outcome.tools_called}", flush=True)
    for problem in failed:
        print(f"       - {problem}", flush=True)


async def run(parts: list[str], sessions: list[str], tag: str | None) -> dict:
    from langfuse import get_client, propagate_attributes
    runners = {"sessions": lambda: run_sessions(sessions), "singles": run_singles,
               "retrieval": run_retrieval, "rubrics": run_rubrics}
    report = {}
    try:
        with propagate_attributes(tags=[tag] if tag else None):
            for part in parts:
                report[part] = await runners[part]()
    finally:
        get_client().flush()
    return report


# Selftest -------------------------------------------------------------------------------

def selftest() -> int:
    """Offline: synthetic outcomes against the real golden rows. A turn built to meet
    its binding fields passes, and each kind of breakage fails the field it breaks."""
    from agent.outcome import TurnOutcome
    from agent.schemas import OnboardingChecklist  # noqa: F401  (import check)

    data = dataset()
    failures = []

    def expect(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    label = "[equipment_policy.md#Personal laptops]"

    def outcome(answer: str, **fields) -> TurnOutcome:
        base = {"status": "completed", "request_id": "selftest", "citations": [label]}
        return TurnOutcome(answer=answer, **{**base, **fields})

    checked = 0
    for session in data["sessions"]:
        for turn in session["turns"]:
            facts = " ".join(f.split("|")[0] for f in turn.get("required_facts") or [])
            sources = [s if "/" in s else f"x/{s}" for s in turn.get("expected_sources") or []]
            tools = ["get_policy"] if turn.get("should_use_tools") else []
            good = outcome(f"The answer states {facts} for this request {label}.",
                           tools_called=tools, sources=sources)
            problems = binding_checks(turn, good)
            if turn.get("expected_handoff"):
                problems = [p for p in problems if "handoff" not in p]
            expect(f"{session['id']} t{turn['n']} good outcome passes binding: {problems}",
                   not problems)
            if turn.get("required_facts"):
                bad = good.model_copy(update={"answer": "Nothing relevant here at all."})
                expect(f"{session['id']} t{turn['n']} missing fact fails",
                       any("missing required facts" in p for p in binding_checks(turn, bad)))
            for tool in turn.get("forbidden_tools") or []:
                bad = good.model_copy(update={"tools_called": [*tools, tool.split("|")[0]]})
                expect(f"{session['id']} t{turn['n']} forbidden tool fails",
                       any("forbidden tools" in p for p in binding_checks(turn, bad)))
            for source in turn.get("forbidden_sources") or []:
                bad = good.model_copy(update={"sources": [*sources, source]})
                expect(f"{session['id']} t{turn['n']} forbidden source fails",
                       any("forbidden source" in p for p in binding_checks(turn, bad)))
            checked += 1

    # Conversation checks.
    expect("an uncited fact sentence is caught", citation_split(outcome(
        "The policy allows five days of personal laptop use.", status="needs_human",
        tools_called=["get_policy"], citations=[])) == "uncited")
    expect("an invented label is caught", citation_split(outcome(
        f"The policy allows five days {label} and IT signs it off [constraint].",
        status="needs_human", tools_called=["get_policy"])) == "invented")
    expect("a cited answer passes", not conversation_checks("S1", 1, outcome(
        f"The policy allows five days of personal laptop use {label}.",
        tools_called=["get_policy"])))
    s3_late = outcome("Alex Kim is E015 [db:employees#E015].", citations=["[db:employees#E015]"],
                      tools_called=["check_software_subscription"],
                      tool_calls=[{"name": "check_software_subscription",
                                   "args": {"software": "Webex"}}])
    expect("S3 closed branch caught", any("closed Webex branch" in p
                                          for p in conversation_checks("S3", 5, s3_late)))
    expect("S4 turn 8 without Finance caught", any("Finance" in p for p in conversation_checks(
        "S4", 8, outcome("Filed AR001 for Rachel [db:access_requests#AR001].",
                         citations=["[db:access_requests#AR001]"],
                         tools_called=["get_employee"]))))
    expect("S4 turn 5 Miro on the Finance list caught", any("Miro" in p for p in conversation_checks(
        "S4", 5, outcome("Webex, Figma and Miro need Finance sign-off.", tools_called=[]))))
    expect("S4 turn 5 that excludes Miro passes", not conversation_checks("S4", 5, outcome(
        "Webex and Figma need Finance sign-off, Miro does not.", tools_called=[])))

    # Singles: documents and tables.
    case = next(c for c in data["single_turn_cases"] if c["case_id"] == "SIMPLE_005")
    expect("SIMPLE_005 tables met by the subscription tool", not single_sources_missing(
        case, outcome("x", tools_called=["check_software_subscription"])))
    expect("SIMPLE_005 tables missed without it", single_sources_missing(
        case, outcome("x", tools_called=[])) == ["contracts", "software_subscriptions"])

    # Retrieval hit rate.
    case = data["retrieval_expectations"][0]
    docs = [p.removeprefix("documents/") for p in case["expected_documents"]]
    expect("retrieval all found", retrieval_hits(case, docs, [])["hit_rate"] == 1.0)
    expect("retrieval half found", retrieval_hits(case, docs[:1], [])["hit_rate"] == 0.5)

    # Rubrics.
    good_denial = outcome("You cannot view another employee's salary fields in BambooHR; "
                          "they are HR-only [data_access_policy.md#HR data]. Contact HR, or "
                          "use self-service for your own profile.")
    expect("SIMPLE_010 rubric passes a good denial", not failed_items(salary_rubric(good_denial)))
    expect("SIMPLE_010 rubric catches a salary number", "must_not_include: salary number" in
           failed_items(salary_rubric(outcome("Her salary is $120,000."))))

    print(f"selftest: {checked} golden turns, {len(failures)} failure(s)")
    for failure in failures:
        print(f"  - {failure}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--sessions", nargs="*", default=[], help="session id prefixes")
    parser.add_argument("--trace", action="store_true", help="trace the run and push scores")
    parser.add_argument("--tag", help="trace tag, e.g. eval:baseline")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    # Development runs default to tracing off (CLAUDE.md rule 2); --trace turns it on.
    os.environ["LANGFUSE_TRACING_ENABLED"] = "true" if args.trace else "false"
    report = asyncio.run(run(args.parts, args.sessions, args.tag))
    report["_meta"] = {"parts": args.parts, "traced": args.trace, "tag": args.tag,
                       "heuristics_off": os.environ.get("NOVAOPS_HEURISTICS_OFF", "")}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
