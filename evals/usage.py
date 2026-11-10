"""This billing period's Langfuse units, from the v2 metrics API.

    python -m evals.usage                   # since the 2026-11-10 reset
    python -m evals.usage --from 2026-11-10T00:00:00Z

A unit is one trace, observation or score. Hobby includes 50,000 a month with no
overage, so this is read before and after every traced measurement.
"""

import argparse
import json

from langfuse import get_client

RESET = "2026-11-10T00:00:00Z"
UNTIL = "2026-12-10T00:00:00Z"


def _query(view: str, metrics: list[dict], start: str) -> dict:
    query = {"view": view, "dimensions": [], "metrics": metrics, "filters": [],
             "fromTimestamp": start, "toTimestamp": UNTIL}
    rows = get_client().api.metrics.metrics(query=json.dumps(query)).data
    return rows[0] if rows else {}


def usage(start: str = RESET) -> dict:
    observed = _query("observations", [{"measure": "count", "aggregation": "count"},
                                       {"measure": "traceId", "aggregation": "uniq"}], start)
    scores = sum(_query(view, [{"measure": "count", "aggregation": "count"}], start)
                 .get("count_count", 0) for view in ("scores-numeric", "scores-categorical"))
    observations = observed.get("count_count", 0)
    traces = observed.get("uniq_traceId", 0)
    return {"from": start, "observations": observations, "traces": traces, "scores": scores,
            "units": observations + traces + scores}


if __name__ == "__main__":
    from agent import config  # noqa: F401  (loads .env before the client starts)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", default=RESET)
    print(json.dumps(usage(parser.parse_args().start), indent=2))
