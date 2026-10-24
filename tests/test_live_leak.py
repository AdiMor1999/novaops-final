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


MAYA_QUERY = "Maya Cohen offer letter: required systems, start date and onboarding summary"
MAYA_SOURCES = {"maya_cohen_offer_letter.md", "maya_cohen_onboarding_summary.md"}


def test_unrelated_regular_caller_gets_no_maya_employment_chunk(live):
    # E010 is UG_REGULAR, manages nobody and is not Maya.
    result = search(MAYA_QUERY, HR_CORPORA, "E010", top_k=20, rerank=no_rerank)
    assert result["access"]["employment_subjects"] == ["E010"]
    assert not MAYA_SOURCES & {c["source"] for c in result["chunks"]}
    assert all(c.get("subject_employee_id") in (None, "E010") for c in result["chunks"])
    assert not any("E001" == c.get("subject_employee_id") for c in result["chunks"])


@pytest.mark.parametrize("caller", ["E004", "E018", "E001"])
def test_hr_manager_and_subject_get_maya_employment_chunks(live, caller):
    result = search(MAYA_QUERY, HR_CORPORA, caller, top_k=20, rerank=no_rerank)
    assert MAYA_SOURCES & {c["source"] for c in result["chunks"]}


def test_contract_question_returns_a_contracts_chunk(live):
    from mcp_server.retrieval import KB_CORPORA
    result = search("Webex contract seat limit and annual cost", KB_CORPORA, "E007",
                    top_k=5, rerank=no_rerank)
    assert any(c["corpus"] == "contracts" for c in result["chunks"])
    assert "webex_vendor_agreement.md" in {c["source"] for c in result["chunks"]}
