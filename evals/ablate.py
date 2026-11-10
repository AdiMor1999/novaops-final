"""Heuristic ablation: full golden replays with one heuristic switched off.

    python -m evals.ablate --repeat 3                      # all five heuristics, in parallel
    python -m evals.ablate fuzzy_name_lookup --repeat 1

For each heuristic in agent/heuristics.py, one worker runs its own MCP server on its own
port, with its own database and checkpoint file, and `NOVAOPS_HEURISTICS_OFF` set to that
heuristic in both the server and the agent (two of the five live in the server). The
worker then replays every golden session `--repeat` times through `evals.score --parts
sessions`. Tracing is off: these runs go only to `evals/results/raw/`, never to Langfuse.
The heuristics-on reference is the baseline replays (evals/results/raw/baseline-*.json).

Workers run in parallel because they share nothing but Bedrock and OpenSearch. Each
worker starts a new Python process per replay, so `--code` points them at a frozen copy
of the commit under test.
"""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent.heuristics import NAMES

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "evals" / "results" / "raw"
FIRST_PORT = 9890


def wait_for_port(port: int, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"MCP server on port {port} did not start")


def worker_env(heuristic: str, port: int, scratch: Path) -> dict:
    env = os.environ.copy()
    env.update(NOVAOPS_HEURISTICS_OFF=heuristic, LANGFUSE_TRACING_ENABLED="false",
               MCP_SERVER_URL=f"http://127.0.0.1:{port}/mcp",
               NOVAOPS_DB_PATH=str(scratch / "novaops.db"),
               NOVAOPS_CHECKPOINT_DB=str(scratch / "checkpoints.db"))
    return env


def start(heuristic: str, port: int, replays: range, logs: Path, code: Path) -> subprocess.Popen:
    """One worker: for each replay, start the server, replay every session, stop the
    server. A server per replay means one lost server costs one replay, not the rest."""
    scratch = Path(tempfile.mkdtemp(prefix=f"ablate-{heuristic}-"))
    env = worker_env(heuristic, port, scratch)
    py = sys.executable
    server_log = logs / f"{heuristic}-server.log"
    steps = [f"{py} -m scripts.reseed >/dev/null"]
    for k in replays:
        steps.append(
            f"({py} -m mcp_server >> {server_log} 2>&1 & echo $! > {scratch / 'pid'}) && "
            f"{py} -c 'from evals.ablate import wait_for_port; wait_for_port({port})' && "
            f"{py} -m evals.score --parts sessions "
            f"--out {RAW / f'ablation-{heuristic}-{k}.json'}; "
            f"kill $(cat {scratch / 'pid'}); sleep 2")
    return subprocess.Popen(["/bin/sh", "-c", " && ".join(steps)], cwd=code, env=env,
                            stdout=open(logs / f"{heuristic}.log", "a"),
                            stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("heuristics", nargs="*", default=list(NAMES))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--first", type=int, default=1,
                        help="number of the first replay, to continue an interrupted run")
    parser.add_argument("--logs", default=str(RAW / "ablation-logs"))
    parser.add_argument("--code", default=str(REPO),
                        help="run the workers from this copy of the repo (e.g. `git archive "
                             "HEAD` unpacked), so edits made meanwhile do not leak in. "
                             "Settings come from this process's environment.")
    args = parser.parse_args()
    if unknown := sorted(set(args.heuristics) - set(NAMES)):
        parser.error(f"unknown heuristics {unknown}")
    RAW.mkdir(parents=True, exist_ok=True)
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    replays = range(args.first, args.first + args.repeat)
    workers = {h: start(h, FIRST_PORT + i, replays, logs, Path(args.code))
               for i, h in enumerate(args.heuristics)}
    failed = 0
    for heuristic, proc in workers.items():
        code = proc.wait()
        print(f"{heuristic}: exit {code}", flush=True)
        failed += code != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
