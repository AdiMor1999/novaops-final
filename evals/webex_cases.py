"""Our Webex cases, run through run_turn() with deterministic checks.

The Rachel scenario is our own case, built from the example in PROJECT-DESCRIPTION.md
§9: Rachel Stein (E010, UG_REGULAR) asks for Webex access for herself. It is checked
against the golden `WF_WEBEX_001` rubric and retrieval expectation, read from
GOLDEN-DATASETS.json:

- every `must_include` item maps to one check on the AccessDecision or the outcome;
- every `must_not_include` item maps to a check on the answer and the actions;
- every `expected_documents` path must be among the documents that reached the model;
- every expected document that reached the model is cited in the answer (task #6).

The team seat question (spec §13) is a second case: a manager asks which of her team
holds a Webex seat. The database records seat counts, not holders, so the answer must
say that seat assignments are not recorded, from the tool result that says so.

    python -m scripts.reseed && python -m mcp_server &
    python -m evals.webex_cases rachel team --out report.json
    python -m evals.webex_cases rachel --repeat 3 --trace --tag eval:rachel --out r.json

Exit code 1 if any check fails. The database is reseeded before each case.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import uuid

from agent import config
from agent.run import run_turn
from agent.schemas import FORBIDDEN_CLAIMS
from evals.run_session import row_counts
from evals.session_checks import source_matches
from mcp_server import db
from scripts.reseed import reseed

RACHEL = {
    "id": "rachel",
    "caller_id": "E010", "user_group": "UG_REGULAR",
    "message": ("I joined Customer Success and need Webex access. Webex says my account is "
                "not licensed."),
}
TEAM = {
    "id": "team",
    "caller_id": "E018", "user_group": "UG_REGULAR",
    "message": "Which people on my team currently hold a Webex seat?",
}
RUBRIC = "WF_WEBEX_001"


def golden(section: str, case_id: str) -> dict:
    data = json.loads((config.course_dir() / "GOLDEN-DATASETS.json").read_text(encoding="utf-8"))
    return next(row for row in data[section] if row["case_id"] == case_id)


def fact_text(decision) -> str:
    return " ".join(f.statement for f in decision.observed_facts).lower()


def must_include(outcome) -> dict[str, bool]:
    """One check per rubric item, keyed by the rubric's own wording."""
    d = outcome.access_decision
    if d is None:  # no decision record: every item fails
        return dict.fromkeys(golden("golden_rubrics", RUBRIC)["must_include"], False)
    actions = {(a.tool, a.record_id): a for a in d.actions_taken}
    reused = {r.record_id for r in d.reused_records}
    expansion = [a for a in d.actions_taken
                 if a.tool == "request_approval" and "subscription_expansion" in a.detail]
    return {
        "employee is eligible for Webex": "eligib" in fact_text(d),
        "subscription is over seat limit": bool(
            d.blocking_reason and re.search(r"42 active seats against a 40-seat limit",
                                            d.blocking_reason)),
        "contract is close to renewal": bool(
            re.search(r"2026-07-20|july 20|20 july|renew", fact_text(d))),
        "do not claim access granted": d.access_granted is False
        and not FORBIDDEN_CLAIMS.search(outcome.answer),
        "reuse existing ticket T001 and access request AR001": {"T001", "AR001"} <= reused,
        "durable approval for expansion or renewal": outcome.status == "pending"
        and ("request_approval", "AP001") in actions and bool(expansion),
    }


def must_not_include(outcome) -> dict[str, bool]:
    """True means the forbidden thing is absent."""
    d = outcome.access_decision
    answer = outcome.answer.lower()
    created = [a for a in (d.actions_taken if d else []) if not a.reused]
    return {
        "access already granted": "already granted" not in answer
        and "access has been granted" not in answer,
        "only a password reset": "only a password reset" not in answer,
        "no approval needed": not re.search(r"no approval (is )?(needed|required)", answer),
        "a duplicate ticket or access request was created": not created,
    }


def expected_documents(outcome) -> dict[str, bool]:
    paths = golden("retrieval_expectations", RUBRIC)["expected_documents"]
    return {path: any(source_matches(path.removeprefix("documents/"), s)
                      for s in outcome.sources) for path in paths}


def document_citations(outcome) -> dict[str, bool]:
    """Task #6: every expected document that reached the model is cited by at least one
    `[<file>#<section>]` label in the answer. A document the case searches returned is
    evidence the decision rests on, so it is treated as used. Whether the cited section
    supports its claim is the citation-quality judge's question, not this check's."""
    paths = golden("retrieval_expectations", RUBRIC)["expected_documents"]
    reached = [p.removeprefix("documents/") for p in paths
               if any(source_matches(p.removeprefix("documents/"), s) for s in outcome.sources)]
    cited = {path: any(c.startswith(f"[{path.rsplit('/', 1)[-1]}#") for c in outcome.citations)
             for path in reached}
    steps = outcome.access_decision.recommended_next_steps if outcome.access_decision else []
    cited["a next step gives an option from a document"] = any(".md#" in s for s in steps)
    return cited


def rachel_checks(outcome) -> dict[str, dict[str, bool]]:
    rubric = golden("golden_rubrics", RUBRIC)
    include, exclude = must_include(outcome), must_not_include(outcome)
    if set(include) != set(rubric["must_include"]) \
            or set(exclude) != set(rubric["must_not_include"]):
        raise ValueError(f"{RUBRIC} items changed; update the checks in evals/webex_cases.py")
    return {"must_include": include, "must_not_include": exclude,
            "expected_documents": expected_documents(outcome),
            "document_citations": document_citations(outcome)}


def team_checks(outcome) -> dict[str, dict[str, bool]]:
    answer = outcome.answer.lower()
    return {"team": {
        "looked up the team by manager_id": "get_team" in outcome.tools_called,
        "read the seat gap from the subscription": "check_software_subscription"
        in outcome.tools_called,
        "says seat assignments are not recorded": bool(
            re.search(r"not (recorded|tracked)|no record of which|does not record", answer)),
        "does not claim access was granted": not FORBIDDEN_CLAIMS.search(outcome.answer),
    }}


CASES = {"rachel": (RACHEL, rachel_checks), "team": (TEAM, team_checks)}


async def run_case(name: str) -> dict:
    case, checks = CASES[name]
    reseed(config.db_path())
    with db.session() as conn:
        before = row_counts(conn)
    outcome = await run_turn(case["message"], caller_id=case["caller_id"],
                             user_group=case["user_group"],
                             thread_id=f"{name}-{uuid.uuid4().hex[:8]}",
                             request_id=f"webex-{name}-{uuid.uuid4().hex[:6]}")
    results = checks(outcome)
    with db.session() as conn:
        results["rows"] = {"no ticket, access request or approval was created":
                           row_counts(conn) == before}
    results["run"] = {"ended pending" if name == "rachel" else "ended completed":
                      outcome.status == ("pending" if name == "rachel" else "completed")}
    failed = [f"{group}: {item}" for group, items in results.items()
              for item, ok in items.items() if not ok]
    print(f"\n== {name}: {outcome.status}  tools={outcome.tools_called}", flush=True)
    for line in failed:
        print(f"FAIL {line}", flush=True)
    print(f"trace {outcome.trace_url}", flush=True)
    return {"case": name, "status": outcome.status, "checks": results, "failed": failed,
            "outcome": outcome.model_dump()}


async def main(names: list[str], out: str | None, *, repeat: int = 1,
               tag: str | None = None) -> int:
    from langfuse import get_client, propagate_attributes
    try:
        with propagate_attributes(tags=[tag] if tag else None):
            report = [await run_case(name) for _ in range(repeat) for name in names]
    finally:
        get_client().flush()
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
    failed = sum(len(r["failed"]) for r in report)
    print(f"\n{failed} check(s) failed over {len(report)} run(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="+", choices=sorted(CASES))
    parser.add_argument("--out")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--trace", action="store_true",
                        help="trace the runs; off by default (CLAUDE.md rule 2)")
    parser.add_argument("--tag", help="trace tag, e.g. eval:rachel")
    args = parser.parse_args()
    os.environ["LANGFUSE_TRACING_ENABLED"] = "true" if args.trace else "false"
    sys.exit(asyncio.run(main(args.cases, args.out, repeat=args.repeat, tag=args.tag)))
