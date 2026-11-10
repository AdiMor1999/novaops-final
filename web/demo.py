"""One command for a live demo: the MCP server and the web chat, with tracing on.

    python -m web.demo            # start both; Ctrl-C stops both
    python -m web.demo --reseed   # restore the database from the course seed first

Both run as child processes of this one, with `LANGFUSE_TRACING_ENABLED=true`, so every
answer in the page carries its Langfuse trace link. Tests and development keep tracing
off; this launcher is the only thing that turns it on.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

from web import settings

MCP_START_TIMEOUT = 30.0


def _env() -> dict:
    return {**os.environ, "LANGFUSE_TRACING_ENABLED": "true"}


def _wait_for_mcp(server: subprocess.Popen) -> None:
    deadline = time.monotonic() + MCP_START_TIMEOUT
    while not settings.mcp_reachable():
        if server.poll() is not None:
            raise SystemExit(f"The MCP server exited with code {server.returncode}.")
        if time.monotonic() > deadline:
            raise SystemExit(f"The MCP server did not open {settings.mcp_address()} "
                             f"within {MCP_START_TIMEOUT:.0f}s.")
        time.sleep(0.3)


def _stop(children: list[subprocess.Popen]) -> None:
    # SIGTERM, not SIGINT: a shell that starts this in the background sets SIGINT to
    # ignored, and the children inherit that. uvicorn shuts down on either, so both
    # servers still run their shutdown and flush Langfuse.
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main() -> None:
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reseed", action="store_true",
                        help="restore the database from the course seed before starting")
    args = parser.parse_args()
    if settings.mcp_reachable():
        raise SystemExit(f"Something already listens on {settings.mcp_address()}. Stop it "
                         "first, so the demo runs against a server with tracing on.")
    if args.reseed:
        subprocess.run([sys.executable, "-m", "scripts.reseed"], check=True)

    children: list[subprocess.Popen] = []
    try:
        children.append(subprocess.Popen([sys.executable, "-m", "mcp_server"], env=_env()))
        _wait_for_mcp(children[0])
        children.append(subprocess.Popen([sys.executable, "-m", "web"], env=_env()))
        print(f"\nOpen http://{settings.host()}:{settings.port()}  (Ctrl-C stops both)\n",
              flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _stop(children)


if __name__ == "__main__":
    main()
