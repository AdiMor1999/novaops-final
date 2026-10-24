"""Tests build their state at run time: a reseeded database in tmp_path, a stub
embedder where OpenSearch is not under test. Nothing is read from tests/fixtures/."""

import os
from pathlib import Path

import pytest

from mcp_server.context import CallerContext
from scripts.reseed import reseed

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    if not os.environ.get("NOVAOPS_COURSE_DIR"):
        pytest.skip("NOVAOPS_COURSE_DIR is not set; the tests need the course dataset")
    monkeypatch.setenv("NOVAOPS_DB_PATH", str(tmp_path / "novaops.db"))
    monkeypatch.setenv("NOVAOPS_CHECKPOINT_DB", str(tmp_path / "checkpoints.db"))
    monkeypatch.setenv("NOVAOPS_TODAY", "2026-07-01")
    monkeypatch.setenv("OPENSEARCH_INDEX", "novaops-final-kb")
    return tmp_path


@pytest.fixture
def seeded(env):
    reseed(env / "novaops.db")
    return env / "novaops.db"


@pytest.fixture
def conn(seeded):
    from mcp_server import db
    connection = db.connect(seeded)
    yield connection
    connection.close()


def caller(caller_id="E007", group="UG_IT", request_id="REQ-T1") -> CallerContext:
    return CallerContext(caller_id=caller_id, user_group=group, request_id=request_id)


def stub_embed(_text: str) -> list[float]:
    return [0.0] * 1024
