"""Where the web chat listens, and where it finds the database and the MCP server.

Host and port are the only settings with defaults: a local demo binds to 127.0.0.1 and
nothing else, and task #10 changes them through `WEB_HOST` and `WEB_PORT` when it
deploys. The database path and the MCP URL come from the same variables the agent
reads, through `agent.config`, so the chat and the agent always see the same files.
"""

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

from agent.config import db_path, mcp_server_url

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8790  # clear of the MCP server (9880) and the lesson servers (9876 to 9878)


def host() -> str:
    return os.environ.get("WEB_HOST", "").strip() or DEFAULT_HOST


def port() -> int:
    raw = os.environ.get("WEB_PORT", "").strip()
    return int(raw) if raw else DEFAULT_PORT


def database() -> Path:
    return db_path()


def mcp_address() -> tuple[str, int]:
    url = urlparse(mcp_server_url())
    return url.hostname or "127.0.0.1", url.port or 9880


def mcp_reachable(timeout: float = 0.5) -> bool:
    """True when something accepts connections on the MCP server's port. A turn
    without the server would fail deep inside the MCP client's task group, so the
    app checks this first and tells the page in plain words."""
    try:
        with socket.create_connection(mcp_address(), timeout=timeout):
            return True
    except OSError:
        return False
