"""Replay a golden session turn by turn on one thread, through run_turn().

Starting code: Lesson 10 `evals/runner.py` (`run_session`: one stable thread id, the
turns in order, one record per turn). Changes: the graph is not passed in, because
run_turn() is the only way in; the database is reseeded once before the session (a
session is one case); each turn's request_id is `<session>-t<n>` so traces match turns;
and the record keeps what the checks need: tools, sources, citations, status, the
checklist, the handoff and the trace URL.

    python -m scripts.reseed && python -m mcp_server &      # the server, on MCP_SERVER_URL
    python -m evals.run_session S2-onboarding-maya S9-maya-own-account --out report.json

Exit code 1 if any turn has a problem.
"""

import argparse
import asyncio
import json
import sys
import uuid

from agent import config
from agent.run import run_turn
from evals.session_checks import binding_checks, project_checks
from evals.tool_matrix import golden_sessions
from scripts.reseed import reseed


def turn_record(session_id: str, turn: dict, outcome) -> dict:
    problems = binding_checks(turn, outcome) + project_checks(session_id, turn["n"], outcome)
    return {
        "n": turn["n"],
        "kind": turn.get("kind"),
        "intent": outcome.intent,
        "expected_intent": turn.get("expected_intent"),
        "scope": outcome.scope,
        "status": outcome.status,
        "tools_called": outcome.tools_called,
        "sources": outcome.sources,
        "citations": outcome.citations,
        "checklist": outcome.checklist.model_dump() if outcome.checklist else None,
        "handoff": outcome.handoff,
        "approval_id": outcome.approval_id,
        "answer": outcome.answer,
        "trace_url": outcome.trace_url,
        "problems": problems,
    }


async def run_session(session: dict, *, thread_id: str | None = None) -> list[dict]:
    thread_id = thread_id or f"{session['id'].split('-')[0]}-{uuid.uuid4().hex[:8]}"
    records = []
    for turn in session["turns"]:
        outcome = await run_turn(turn["user"], caller_id=session["caller_id"],
                                 user_group=session["caller_group"], thread_id=thread_id,
                                 request_id=f"{session['id'].split('-')[0]}-t{turn['n']}")
        record = turn_record(session["id"], turn, outcome)
        record["thread_id"] = thread_id
        records.append(record)
        print_turn(record)
    return records


def print_turn(record: dict) -> None:
    mark = "ok  " if not record["problems"] else "FAIL"
    print(f"{mark} t{record['n']:<2} {record['intent'] or '-':<20} {record['status']:<11} "
          f"tools={record['tools_called']}", flush=True)
    for problem in record["problems"]:
        print(f"       - {problem}", flush=True)


async def main(ids: list[str], out: str | None) -> int:
    sessions = {s["id"]: s for s in golden_sessions(tuple(ids))}
    report = {}
    for session_id in ids:
        reseed(config.db_path())
        print(f"\n== {session_id}", flush=True)
        report[session_id] = await run_session(sessions[session_id])
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
    failed = sum(bool(r["problems"]) for records in report.values() for r in records)
    print(f"\n{failed} turn(s) with problems")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.sessions, args.out)))
