"""The one config loader. Every module reads settings through these functions.

`find_dotenv()` walks up from the working directory, so the `.env` at the Daedalus
workspace root is found from any worktree. A missing variable raises `MissingConfig`
with its name. Nothing here falls back to a default: a guessed model id or index name
runs fine and quietly does the wrong thing, which is worse than stopping.

Each getter reads the environment when called, not at import, so tests can point
`NOVAOPS_DB_PATH` at a temporary file with `monkeypatch.setenv`.
"""

import os
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Lesson 7 hard-codes this name. It holds deliberately poisoned documents, so it is
# refused by name even if someone puts it in OPENSEARCH_INDEX.
FORBIDDEN_INDEX = "novaops-kb"

EMBED_DIM = 1024  # Titan Text Embeddings V2 width; the index mapping uses the same number.


class MissingConfig(RuntimeError):
    """A required environment variable is unset or empty. Ask for it; do not default it."""


def require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingConfig(
            f"{name} is not set. Add it to the workspace .env (see .env.example)."
        )
    return value


def optional(name: str) -> str | None:
    """Only for variables `.env.example` marks optional."""
    return os.environ.get(name, "").strip() or None


def aws_region() -> str:
    return require("AWS_REGION")


def bedrock_model_id() -> str:
    return require("BEDROCK_MODEL_ID")


def embedding_model_id() -> str:
    return require("BEDROCK_EMBEDDING_MODEL_ID")


def opensearch_collection() -> str:
    return require("OPENSEARCH_COLLECTION")


def opensearch_index() -> str:
    return require("OPENSEARCH_INDEX")


def mcp_server_url() -> str:
    return require("MCP_SERVER_URL")


def course_dir() -> Path:
    return Path(require("NOVAOPS_COURSE_DIR"))


def dataset_dir() -> Path:
    return course_dir() / "novaops-enterprise-agent-dataset"


def documents_dir() -> Path:
    return dataset_dir() / "documents"


def db_path() -> Path:
    """The NovaOps SQLite file. The parent folder is created if missing."""
    path = Path(require("NOVAOPS_DB_PATH"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_db() -> Path:
    path = Path(require("NOVAOPS_CHECKPOINT_DB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def today() -> date:
    """The pinned clock (2026-07-01). The only source of "today" in this repo."""
    raw = require("NOVAOPS_TODAY")
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise MissingConfig(f"NOVAOPS_TODAY must be YYYY-MM-DD, got {raw!r}.") from exc


def now_iso() -> str:
    """A timestamp on the pinned date. The time of day is the real UTC time, so rows
    written in one run still sort in the order they were written."""
    clock = datetime.now(timezone.utc).time().replace(microsecond=0)
    return f"{today().isoformat()}T{clock.isoformat()}Z"


def heuristics_off() -> str:
    """Comma list of agent/heuristics.py names switched off for an ablation run. Empty,
    the default, means every heuristic is on. Optional in `.env.example`."""
    return os.environ.get("NOVAOPS_HEURISTICS_OFF", "").strip()


def judge_model_id() -> str:
    """The judges' model (Amazon Nova Pro, `us.amazon.nova-pro-v1:0`). Langfuse calls it
    through its own LLM connection; the repo uses it for the connection test only."""
    return require("JUDGE_MODEL_ID")
