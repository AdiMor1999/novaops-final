"""Check that a Langfuse trace has the readable shape (task #6, decisions 12 and 14).

    turn (AGENT)
      <node> (SPAN, one of the graph's nodes)
        model:<role> (GENERATION, with the model id and token usage)
        tool:<name> (TOOL)
          server:<name> (SPAN)
            rerank (GENERATION, retrieval's reranker)

Every observation is named after a node, a tool or a model call. Nothing is a LangChain
runnable (`RunnableSequence`, `ChatBedrockConverse`, `LangGraph`, `after_*`). A resumed
turn adds a second `turn` root to the same trace; `approval_decision` is the root of its
own trace (agent/approvals.py).

The trace is read through the v2 observations API (`api.observations.get_many`); this
project's Langfuse org answers the legacy trace API with 410.

    python -m evals.trace_tree <trace_id> [<trace_id> ...]

Exit code 1 if any trace has a problem.
"""

import argparse
import re
import sys
import time
from collections import Counter

from langfuse import get_client

ROOTS = {"turn": "AGENT", "approval_decision": "SPAN"}
ANONYMOUS = re.compile(r"^(Runnable|ChatBedrock|LangGraph|__start__|after_|_Repairing)")
FIELDS = "core,basic,usage,model"


def graph_nodes() -> frozenset[str]:
    """The node names the graph registers, read from a graph built with no session."""
    from agent.graph import build_graph
    graph = build_graph(None, [], None, model=None, planner=None)
    return frozenset(n for n in graph.get_graph().nodes if not n.startswith("__"))


def fetch(trace_id: str, *, wait_for: int = 1, attempts: int = 6, fields: str = FIELDS) -> list:
    """All observations of one trace. Ingestion lags export by a few seconds, so this
    retries until at least `wait_for` observations are there."""
    api = get_client().api
    for attempt in range(attempts):
        found, cursor = [], None
        while True:
            page = api.observations.get_many(trace_id=trace_id, limit=1000, cursor=cursor,
                                             fields=fields,
                                             request_options={"timeout_in_seconds": 60})
            found.extend(page.data)
            cursor = getattr(page.meta, "cursor", None) if page.meta else None
            if not cursor:
                break
        if len(found) >= wait_for:
            return found
        time.sleep(5 * (attempt + 1))
    return found


def resumed_root(o) -> bool:
    """The `turn` of a resumed run. agent/run.py opens it with `trace_context` holding
    the paused trace's id, and the SDK then gives it a random parent span id
    (`Langfuse._create_remote_parent_span`) that no observation has. Langfuse shows
    such an observation at the top of the trace, next to the paused `turn`."""
    return o.name == "turn" and o.type == "AGENT"


def problems(observations: list, nodes: frozenset[str], *, live: bool = True) -> list[str]:
    """What breaks the shape. `live=False` skips the usage check for stubbed models."""
    by_id = {o.id: o for o in observations}
    out = []

    def parent(o):
        return by_id.get(o.parent_observation_id) if o.parent_observation_id else None

    for o in observations:
        up = parent(o)
        kind, name = o.type, o.name or ""
        if ANONYMOUS.match(name):
            out.append(f"anonymous runnable {kind} {name!r}")
        elif o.parent_observation_id and up is None and not resumed_root(o):
            out.append(f"{kind} {name!r} points at a parent that is not in the trace")
        elif up is None:
            if ROOTS.get(name) != kind:
                out.append(f"root is {kind} {name!r}, not turn or approval_decision")
        elif name in nodes and kind == "SPAN":
            if up.name not in ROOTS:
                out.append(f"node span {name!r} sits under {up.name!r}, not a root")
        elif name.startswith("model:") and kind == "GENERATION":
            if up.name not in nodes:
                out.append(f"{name!r} sits under {up.name!r}, not a node span")
            if live and not o.model:
                out.append(f"{name!r} has no model id")
            if live and not (o.usage_details or {}).get("input"):
                out.append(f"{name!r} has no token usage")
        elif name.startswith("tool:") and kind == "TOOL":
            if up.name not in nodes and up.name not in ROOTS:
                out.append(f"{name!r} sits under {up.name!r}, not a node span")
        elif name.startswith("server:") and kind == "SPAN":
            if up.name != "tool:" + name.removeprefix("server:"):
                out.append(f"{name!r} sits under {up.name!r}, not its tool span")
        elif name == "rerank" and kind == "GENERATION":
            if not (up.name or "").startswith("server:"):
                out.append(f"rerank sits under {up.name!r}, not a server span")
        else:
            out.append(f"unexpected {kind} {name!r} under {up.name!r}")
    return out


def tree_lines(observations: list) -> list[str]:
    children: dict = {}
    for o in observations:
        children.setdefault(o.parent_observation_id, []).append(o)
    lines = []

    ids = {o.id for o in observations}
    for o in observations:  # a resumed turn's parent id is not in the trace
        if o.parent_observation_id and o.parent_observation_id not in ids:
            children.setdefault(None, []).append(o)

    def walk(parent_id, depth):
        for o in sorted(children.get(parent_id, []), key=lambda o: str(o.start_time)):
            lines.append(f"{'  ' * depth}{o.type} {o.name}")
            walk(o.id, depth + 1)
    walk(None, 0)
    return lines


def check(trace_id: str, nodes: frozenset[str], *, live: bool = True,
          wait_for: int = 1) -> dict:
    observations = fetch(trace_id, wait_for=wait_for)
    return {"trace_id": trace_id, "observations": len(observations),
            "turns": sum(o.name == "turn" for o in observations),
            "by_type": dict(Counter(o.type for o in observations)),
            "problems": problems(observations, nodes, live=live) if observations
            else ["no observations found"],
            "tree": tree_lines(observations)}


def main(trace_ids: list[str], show: bool, live: bool = True) -> int:
    from agent import config  # noqa: F401  (loads .env before the client starts)
    nodes = graph_nodes()
    failed = 0
    for trace_id in trace_ids:
        result = check(trace_id, nodes, live=live)
        mark = "ok  " if not result["problems"] else "FAIL"
        print(f"{mark} {trace_id}  {result['observations']} observations, "
              f"{result['turns']} turn(s), {result['by_type']}")
        for problem in result["problems"]:
            print(f"       - {problem}")
        if show:
            print("\n".join(f"       {line}" for line in result["tree"]))
        failed += bool(result["problems"])
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace_ids", nargs="+")
    parser.add_argument("--show", action="store_true", help="print the tree")
    parser.add_argument("--stubbed", action="store_true",
                        help="scripted models (evals/durability.py): skip the usage check")
    args = parser.parse_args()
    sys.exit(main(args.trace_ids, args.show, live=not args.stubbed))
