"""Delete and rebuild the NovaOps database at NOVAOPS_DB_PATH, then print row counts.

    python -m scripts.reseed

Run it before each independent case (spec §13). It loads the course `schema.sql` and
`seed.sql` unchanged, then adds this project's side tables (`approval_controls`,
`idempotency_keys`) from `mcp_server/writes.py`.
"""

import sqlite3
import sys
from pathlib import Path

from agent import config
from mcp_server import db
from mcp_server.writes import CONTROL_SCHEMA


def reseed(path: Path | None = None, dataset: Path | None = None) -> dict[str, int]:
    path = path or config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    database_dir = (dataset or config.dataset_dir()) / "database"
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript((database_dir / db.SCHEMA_FILE).read_text(encoding="utf-8"))
        conn.executescript((database_dir / db.SEED_FILE).read_text(encoding="utf-8"))
        conn.executescript(CONTROL_SCHEMA)
        conn.commit()
        conn.row_factory = sqlite3.Row
        return db.table_counts(conn)
    finally:
        conn.close()


def main() -> None:
    path = config.db_path()
    counts = reseed(path)
    print(f"Reseeded {path}")
    for table, count in counts.items():
        print(f"  {table:24} {count}")
    if counts.get("employees", 0) == 0:
        sys.exit("Seed loaded no employees.")


if __name__ == "__main__":
    main()
