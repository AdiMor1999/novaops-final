"""Grep the repo's code and config for banned calls. Docs that state the ban are not
code. The patterns are built from pieces so this file does not match itself."""

from tests.conftest import REPO

BANNED = ["InMemory" + "Saver", "Memory" + "Saver", "date" + ".today()"]
SKIP = {".venv", ".git", "__pycache__", ".pytest_cache"}


def source_files():
    for path in REPO.rglob("*"):
        if path.is_file() and not SKIP & set(path.relative_to(REPO).parts) \
                and path.suffix in {".py", ".txt", ".ini", ".toml", ".cfg"}:
            yield path


def test_no_banned_code_anywhere():
    hits = [f"{p.relative_to(REPO)}: {word}" for p in source_files()
            for word in BANNED if word in p.read_text(encoding="utf-8", errors="ignore")]
    assert hits == []
