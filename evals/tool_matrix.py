"""The tool-matrix checker (Lesson 10 homework, milestone 3).

For every S2 and S9 turn that should use tools, and for every intent the golden row
accepts, each expected tool must be reachable from that intent's loadout. This catches a
policy that makes a dataset expectation impossible even when the classification is right.
It also checks that no loadout, and not the rearm widening either, contains a write tool
or `request_approval`.

    python -m evals.tool_matrix     # prints one line per problem; exit 1 if any
"""

import json
import sys

from agent import config
from agent.policy import INTENTS, LOADOUTS, NOT_FOR_THE_MODEL, rearm_tools
from evals.tool_map import ours

MAYA_SESSIONS = ("S2-onboarding-maya", "S9-maya-own-account")


def golden_sessions(ids: tuple[str, ...] = MAYA_SESSIONS) -> list[dict]:
    data = json.loads((config.course_dir() / "GOLDEN-DATASETS.json").read_text(encoding="utf-8"))
    return [s for s in data["sessions"] if s["id"] in ids]


def loadout(intent: str, catalogue: list[str]) -> set[str]:
    chosen = LOADOUTS[intent]
    return set(rearm_tools(catalogue)) if chosen is None else set(chosen)


def reachability_problems(sessions: list[dict], catalogue: list[str]) -> list[str]:
    problems = []
    for session in sessions:
        for turn in session["turns"]:
            if not turn["should_use_tools"] or not turn["expected_intent"]:
                continue
            for intent in turn["expected_intent"].split("|"):
                tools = loadout(intent, catalogue)
                for expected in turn["expected_tools"]:
                    if not ours(expected) & tools:
                        problems.append(f"{session['id']} turn {turn['n']}: {expected} is not "
                                        f"reachable from the {intent} loadout")
    return problems


def write_problems(catalogue: list[str]) -> list[str]:
    problems = [f"loadout {intent} contains {sorted(set(LOADOUTS[intent]) & NOT_FOR_THE_MODEL)}"
                for intent in INTENTS
                if LOADOUTS[intent] and set(LOADOUTS[intent]) & NOT_FOR_THE_MODEL]
    if leaked := set(rearm_tools(catalogue)) & NOT_FOR_THE_MODEL:
        problems.append(f"rearm contains {sorted(leaked)}")
    if leaked := set(loadout("other", catalogue)) & NOT_FOR_THE_MODEL:
        problems.append(f"the 'other' loadout contains {sorted(leaked)}")
    return problems


def server_catalogue() -> list[str]:
    """Every tool the MCP server serves, writes included, as list_tools() reports them."""
    import asyncio

    from mcp_server.server import mcp
    return sorted(t.name for t in asyncio.run(mcp.list_tools()))


def check(catalogue: list[str] | None = None) -> list[str]:
    catalogue = catalogue if catalogue is not None else server_catalogue()
    return reachability_problems(golden_sessions(), catalogue) + write_problems(catalogue)


if __name__ == "__main__":
    found = check()
    for problem in found:
        print(problem)
    print(f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
