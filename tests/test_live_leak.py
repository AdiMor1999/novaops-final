"""Live check against novaops-final-kb: a non-manager never gets a manager_playbook chunk."""

import os

import pytest

from mcp_server.retrieval import HR_CORPORA, RerankParseFailed, search

pytestmark = pytest.mark.integration

QUERY = "How do I handle a promotion decision for someone on my team?"


def no_rerank(_query, _candidates):
    raise RerankParseFailed("rerank skipped in this test; vector order is enough")


@pytest.fixture
def live(seeded):
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("OPENSEARCH_COLLECTION")):
        pytest.skip("no AWS or OpenSearch credentials")
    from mcp_server import opensearch
    if not opensearch.index_exists():
        pytest.skip("novaops-final-kb is not created; run scripts.create_index and scripts.ingest")


def test_non_manager_gets_no_manager_playbook_chunk(live):
    result = search(QUERY, HR_CORPORA, "E001", top_k=20, rerank=no_rerank)
    assert result["caller_is_manager"] is False
    assert result["count"] > 0
    assert all(c["corpus"] != "manager_playbook" and c.get("audience") != "manager"
               for c in result["chunks"])


def test_manager_can_get_manager_playbook_chunk(live):
    result = search(QUERY, HR_CORPORA, "E018", top_k=20, rerank=no_rerank)
    assert any(c["corpus"] == "manager_playbook" for c in result["chunks"])
