"""Build evals/results/results.md from the raw run files in evals/results/raw/.

    python -m evals.report              # tables from whatever raw files exist
    python -m evals.report --units      # also read this month's Langfuse units

Every number is a count or a rate over repeated runs; nothing comes from a single turn.
The raw files are the evidence, committed next to the table:

| File | Written by |
| --- | --- |
| `baseline-<k>.json` | `evals.score --trace --tag eval:baseline` (sessions, plus the other parts on replay 1) |
| `ablation-<heuristic>-<k>.json` | `evals.ablate` (sessions, tracing off) |
| `rachel-before.json`, `rachel-after.json` | `evals.webex_cases rachel --repeat 3 --trace --tag eval:rachel` |
| `paused-s4-{before,after}-<k>.json` | `evals.score --parts sessions --sessions S4`, before and after the paused-thread fix |
| `trace-tree.json` | observations per turn before and after the tracing change (evals/trace_tree.py) |

Starting code: Lesson 10 `evals/compare.py` (the S6 to S8 noise curve, per-variant
deltas). Changes: binding pass rates instead of weighted quality, and our own sections.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from agent.heuristics import NAMES

RESULTS = Path(__file__).resolve().parent / "results"
RAW = RESULTS / "raw"
NOISE = {"S6": "clean", "S7": "noisy", "S8": "heavy noise"}


def load(pattern: str) -> list[tuple[str, dict | list]]:
    return [(p.stem, json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(RAW.glob(pattern))]


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f}% ({part:g}/{whole:g})" if whole else "n/a"


def session_turns(report: dict) -> dict[str, list[dict]]:
    return report.get("sessions", {})


def binding_rate(turns: list[dict]) -> tuple[int, int]:
    return sum(not t["binding"] for t in turns), len(turns)


# Sections -------------------------------------------------------------------------------

def noise_curve(baselines: list) -> list[str]:
    lines = ["## S6 to S8: robustness to how the request is written", "",
             "Binding pass rate = turns with no failed binding field / turns, over the "
             f"{len(baselines)} baseline replays. The three sessions have identical "
             "expectations, so the delta is the cost of noise.", "",
             "| Session | Noise | Binding pass rate | Project checks pass rate | Delta vs S6 |",
             "| --- | --- | --- | --- | --- |"]
    rates = {}
    for prefix, noise in NOISE.items():
        turns = [t for _, r in baselines for sid, ts in session_turns(r).items()
                 if sid.startswith(prefix) for t in ts]
        ok, n = binding_rate(turns)
        proj = sum(not t["project_checks"] for t in turns)
        rates[prefix] = ok / n if n else None
        delta = "" if prefix == "S6" or rates["S6"] is None or rates[prefix] is None else \
            f"{100 * (rates[prefix] - rates['S6']):+.0f} points"
        lines.append(f"| {prefix} | {noise} | {pct(ok, n)} | {pct(proj, n)} | {delta} |")
    return lines + [""]


def per_session(baselines: list) -> list[str]:
    lines = ["## Every session, binding and project checks", "",
             "| Session | Turns per replay | Binding pass rate | Project checks pass rate |",
             "| --- | --- | --- | --- |"]
    ids = sorted({sid for _, r in baselines for sid in session_turns(r)})
    for sid in ids:
        turns = [t for _, r in baselines for t in session_turns(r).get(sid, [])]
        ok, n = binding_rate(turns)
        proj = sum(not t["project_checks"] for t in turns)
        per = len(session_turns(baselines[0][1]).get(sid, []))
        lines.append(f"| {sid} | {per} | {pct(ok, n)} | {pct(proj, n)} |")
    failures = Counter(p.split(" [")[0].split(" {")[0] for _, r in baselines
                       for ts in session_turns(r).values() for t in ts for p in t["binding"])
    if failures:
        lines += ["", "Most frequent binding failures:", ""]
        lines += [f"- {count}x {kind}" for kind, count in failures.most_common(8)]
    return lines + [""]


def s9_turn_one(baselines: list) -> list[str]:
    runs = [next((t for t in session_turns(r).get(sid, []) if t["n"] == 1), None)
            for _, r in baselines for sid in session_turns(r) if sid.startswith("S9")]
    runs = [t for t in runs if t]
    ok = sum(not any("required facts" in p for p in t["binding"]) for t in runs)
    return ["## S9 turn 1, required facts", "",
            f"Required facts present: {pct(ok, len(runs))} of baseline replays.", ""]


def retrieval(baselines: list) -> list[str]:
    cases = [c for _, r in baselines for c in r.get("retrieval", {}).values()]
    if not cases:
        return []
    found = sum(sum(c["documents"].values()) for c in cases)
    expected = sum(len(c["documents"]) for c in cases)
    full = sum(c["hit_rate"] == 1 for c in cases)
    lines = ["## `retrieval_expectations` hit rate", "",
             f"Expected documents retrieved: {pct(found, expected)}. Cases with every "
             f"expected document: {pct(full, len(cases))}.", "",
             "| Case | Hit rate per run |", "| --- | --- |"]
    by_case: dict[str, list[float]] = {}
    for _, r in baselines:
        for case_id, c in r.get("retrieval", {}).items():
            by_case.setdefault(case_id, []).append(c["hit_rate"])
    lines += [f"| {k} | {', '.join(f'{v:.2f}' for v in vs)} |" for k, vs in by_case.items()]
    return lines + [""]


def citation_split(baselines: list) -> list[str]:
    turns = [t for _, r in baselines for ts in session_turns(r).values() for t in ts]
    needs_human = [t for t in turns if t["status"] == "needs_human"]
    split = Counter(t.get("citation_split") or "other reason" for t in needs_human)
    lines = ["## Turns ending `needs_human`, and why", "",
             f"{pct(len(needs_human), len(turns))} of session turns ended `needs_human`.", "",
             "| Reason | Turns | Share of all turns |", "| --- | --- | --- |"]
    labels = {"uncited": "citation rule: uncited fact sentences",
              "invented": "citation rule: labels no tool returned",
              "both": "citation rule: both", "other reason": "another reason (gate, lookup)"}
    lines += [f"| {labels[k]} | {v} | {pct(v, len(turns))} |" for k, v in split.most_common()]
    return lines + [""]


def heuristics(baselines: list) -> list[str]:
    turns = [t for _, r in baselines for ts in session_turns(r).values() for t in ts]
    base_ok, base_n = binding_rate(turns)
    lines = ["## Heuristics: how often each fires, and what changes without it", "",
             f"Firing rate: turns on which the rule fired / session turns, over the "
             f"{len(baselines)} baseline replays (all heuristics on). Off: three replays "
             "with only that heuristic switched off (`NOVAOPS_HEURISTICS_OFF`), tracing "
             f"off. Baseline binding pass rate: {pct(base_ok, base_n)}.", "",
             "| Heuristic | Fires on | Binding pass rate with it off | Change | Turns that "
             "flip (on → off) |", "| --- | --- | --- | --- | --- |"]
    on_by_turn = turn_pass_rates(baselines)
    for name in NAMES:
        fired = sum(name in t.get("heuristics", []) for t in turns)
        runs = load(f"ablation-{name}-*.json")
        off_turns = [t for _, r in runs for ts in session_turns(r).values() for t in ts]
        if not off_turns:
            lines.append(f"| `{name}` | {pct(fired, len(turns))} | not run | | |")
            continue
        ok, n = binding_rate(off_turns)
        change = 100 * (ok / n - base_ok / base_n) if base_n and n else 0
        flips = flipped(on_by_turn, turn_pass_rates(runs))
        lines.append(f"| `{name}` | {pct(fired, len(turns))} | {pct(ok, n)} | "
                     f"{change:+.0f} points | {', '.join(flips) or 'none'} |")
    return lines + ["", "A turn flips when its binding pass rate moves by half or more "
                    "between the two sets of runs.", ""]


def turn_pass_rates(runs: list) -> dict[str, float]:
    seen: dict[str, list[bool]] = {}
    for _, r in runs:
        for sid, ts in session_turns(r).items():
            for t in ts:
                seen.setdefault(f"{sid.split('-')[0]} t{t['n']}", []).append(not t["binding"])
    return {k: sum(v) / len(v) for k, v in seen.items()}


def flipped(on: dict[str, float], off: dict[str, float]) -> list[str]:
    return [f"{k} {on[k]:.0%}→{off[k]:.0%}" for k in on if k in off and abs(on[k] - off[k]) >= 0.5]


def rachel() -> list[str]:
    lines = ["## The Rachel case before and after the Webex citation fix", "",
             "Three traced runs each (`eval:rachel`). Documents: expected `WF_WEBEX_001` "
             "documents cited / expected documents that reached the model.", "",
             "| Runs | Expected documents cited | Runs citing all | Runs with a next step from "
             "a document | Runs passing every rubric check |", "| --- | --- | --- | --- | --- |"]
    for stem, runs in load("rachel-*.json"):
        cited = reached = full = steps = clean = 0
        for run in runs:
            outcome = run["outcome"]
            docs = {k: v for k, v in run["checks"].get("document_citations", {}).items()
                    if k.endswith(".md")}
            if not docs:  # a before-run file written without the check: recompute it
                from evals.webex_cases import document_citations
                from agent.outcome import TurnOutcome
                docs = {k: v for k, v in document_citations(
                    TurnOutcome(**outcome)).items() if k.endswith(".md")}
            cited += sum(docs.values())
            reached += len(docs)
            full += all(docs.values())
            next_steps = (outcome.get("access_decision") or {}).get("recommended_next_steps", [])
            steps += any(".md#" in s for s in next_steps)
            clean += not [f for f in run["failed"] if not f.startswith("document_citations")]
        lines.append(f"| {stem.removeprefix('rachel-')} | {pct(cited, reached)} | "
                     f"{full}/{len(runs)} | {steps}/{len(runs)} | {clean}/{len(runs)} |")
    return lines + [""]


def paused_thread() -> list[str]:
    lines = ["## S4 before and after the paused-thread fix", "",
             "S4 turn 8 pauses on an approval; turn 9 is a wrap-up. Before the fix any "
             "message on a paused thread resumed the pause; after it, a message with no "
             "recorded decision is a normal turn. S4 is the only golden session with a turn "
             "after a pause (S2 turn 10 hands off without pausing; S6 to S8 pause on their "
             "last turn).", "",
             "| Runs | S4 turn 9 binding pass | S4 binding pass rate | S4 turn 9 tools_called |",
             "| --- | --- | --- | --- |"]
    for when in ("before", "after"):
        runs = load(f"paused-s4-{when}-*.json")
        if not runs:
            continue
        turns = [t for _, r in runs for ts in session_turns(r).values() for t in ts]
        t9 = [t for t in turns if t["n"] == 9]
        ok, n = binding_rate(turns)
        tools = "; ".join(str(t["tools_called"]) for t in t9)
        lines.append(f"| {when} | {sum(not t['binding'] for t in t9)}/{len(t9)} | "
                     f"{pct(ok, n)} | {tools} |")
    return lines + [""]


def trace_tree() -> list[str]:
    path = RAW / "trace-tree.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    lines = ["## Observations per turn, before and after the tracing change", "",
             "| Turn | Before (trace) | After (trace) |", "| --- | --- | --- |"]
    lines += [f"| {r['turn']} | {r['before']} (`{r['before_trace']}`) | {r['after']} "
              f"(`{r['after_trace']}`) |" for r in rows]
    return lines + [""]


def units() -> list[str]:
    from evals.usage import usage
    found = usage()
    return ["## Langfuse units this billing period", "",
            f"Since {found['from']}: {found['observations']} observations, "
            f"{found['traces']} traces, {found['scores']} scores = **{found['units']}** "
            "units (Hobby limit 50,000).", ""]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", action="store_true")
    args = parser.parse_args()
    baselines = load("baseline-*.json")
    out = ["# Task #6 results", "",
           "Generated by `python -m evals.report` from the files in `raw/`. Interpretation, "
           "the judge-versus-check disagreement and the cost note are in `README.md` here.",
           ""]
    out += trace_tree()
    if baselines:
        out += noise_curve(baselines) + per_session(baselines) + s9_turn_one(baselines)
        out += retrieval(baselines) + citation_split(baselines) + heuristics(baselines)
    out += rachel() + paused_thread()
    if args.units:
        out += units()
    (RESULTS / "results.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {RESULTS / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
