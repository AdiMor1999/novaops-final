"""The access clauses must sit inside knn.<field>.filter (pre-filter). Assert on the
request body, not on an answer."""

from mcp_server import retrieval
from mcp_server.db import caller_access
from mcp_server.retrieval import (HR_CORPORA, KB_CORPORA, PLAYBOOK_NOTE, RerankParseFailed,
                                  body_for_caller, search)
from tests.conftest import stub_embed

PLAYBOOK = [{"term": {"audience": "manager"}}, {"term": {"corpus": "manager_playbook"}}]


def knn_filter(body):
    (field,) = body["query"]["knn"].keys()
    return body["query"]["knn"][field]["filter"]


def employment_clause(body):
    clauses = [c for c in knn_filter(body)["bool"].get("must_not", []) if "bool" in c]
    return clauses[0]["bool"] if clauses else None


def test_non_manager_body_has_must_not_inside_knn_filter(seeded):
    body = body_for_caller("promotion process", HR_CORPORA, "E001", embed=stub_embed)
    assert knn_filter(body)["bool"]["must_not"][:2] == PLAYBOOK
    assert "post_filter" not in body and "filter" not in body["query"]


def test_manager_body_keeps_the_playbook_open(seeded):
    body = body_for_caller("promotion process", HR_CORPORA, "E018", embed=stub_embed)
    assert not any(c in knn_filter(body)["bool"].get("must_not", []) for c in PLAYBOOK)


def test_corpus_scope_is_inside_the_same_filter(seeded):
    body = body_for_caller("vpn", ("it_kb",), "E001", embed=stub_embed)
    assert knn_filter(body)["bool"]["must"] == [{"terms": {"corpus": ["it_kb"]}}]


def test_unrelated_regular_caller_reads_only_their_own_employment_documents(seeded):
    # E010 (Rachel Stein) manages nobody and is UG_REGULAR: Maya's offer letter
    # (subject E001) is excluded inside the k-NN filter, before ranking.
    clause = employment_clause(body_for_caller("Maya Cohen offer letter", HR_CORPORA, "E010",
                                               embed=stub_embed))
    assert clause["filter"] == [{"term": {"corpus": "employment"}},
                                {"exists": {"field": "subject_employee_id"}}]
    assert clause["must_not"] == [{"terms": {"subject_employee_id": ["E010"]}}]


def test_subject_reads_her_own_employment_documents(seeded):
    clause = employment_clause(body_for_caller("my offer letter", HR_CORPORA, "E001",
                                               embed=stub_embed))
    assert clause["must_not"] == [{"terms": {"subject_employee_id": ["E001"]}}]


def test_manager_reads_her_reports_employment_documents(seeded):
    clause = employment_clause(body_for_caller("Maya Cohen offer letter", HR_CORPORA, "E018",
                                               embed=stub_embed))
    subjects = clause["must_not"][0]["terms"]["subject_employee_id"]
    assert "E018" in subjects and "E001" in subjects and "E010" in subjects


def test_hr_caller_has_no_employment_clause(seeded):
    # UG_HR comes from the employee row, not from _meta.
    assert caller_access("E004", seeded).hr is True
    assert employment_clause(body_for_caller("Maya Cohen offer letter", HR_CORPORA, "E004",
                                             embed=stub_embed)) is None


def test_unknown_caller_reads_no_one_elses_employment_documents(seeded):
    clause = employment_clause(body_for_caller("offer letter", HR_CORPORA, "E999",
                                               embed=stub_embed))
    assert clause["must_not"] == [{"terms": {"subject_employee_id": ["E999"]}}]


def test_knowledge_base_search_includes_contracts():
    assert "contracts" in KB_CORPORA and "employment" not in KB_CORPORA


def _no_rerank(_query, _candidates):
    raise RerankParseFailed("not under test")


def test_playbook_note_is_identical_whatever_the_query(seeded, monkeypatch):
    """The note must not depend on whether a playbook chunk would have matched, or it
    tells the caller what the playbook covers. One query 'matches' a handbook chunk,
    the other matches nothing; the note is the same."""
    def fake_search(body):
        text = body["query"]["knn"]["vector"]["vector"]
        return [] if text[0] == 1.0 else [{"_score": 0.9, "_source": {
            "text": "Promotions happen on April 1 and October 1.", "corpus": "handbook",
            "source": "making-a-career.md", "doc_id": "handbook/making-a-career.md",
            "citation": "[making-a-career.md#Promotions]"}}]

    monkeypatch.setattr(retrieval.opensearch, "search", fake_search)
    playbook_topic = search("manager guide to feedback in the first 90 days", HR_CORPORA,
                            "E001", embed=lambda _q: [1.0] * 1024, rerank=_no_rerank)
    other_topic = search("how do I get promoted", HR_CORPORA, "E001",
                         embed=lambda _q: [0.0] * 1024, rerank=_no_rerank)
    assert playbook_topic["access_note"] == other_topic["access_note"] == PLAYBOOK_NOTE
    assert playbook_topic["access"] == other_topic["access"]
    manager = search("how do I get promoted", HR_CORPORA, "E018",
                     embed=lambda _q: [0.0] * 1024, rerank=_no_rerank)
    assert "access_note" not in manager


def test_precedence_note_names_the_newer_memo():
    chunks = [
        {"doc_type": "policy", "source": "saas_procurement_policy.md", "last_updated": "2026-01-01"},
        {"doc_type": "memo", "source": "saas_renewal_freeze_q3.md", "last_updated": "2026-06-25"},
    ]
    note = retrieval.precedence_note(chunks)
    assert note.startswith("saas_renewal_freeze_q3.md (2026-06-25) is newer than "
                           "saas_procurement_policy.md")
    assert retrieval.precedence_note(chunks[:1]) is None


def test_rerank_keeps_at_most_two_chunks_per_document():
    hits = [{"_source": {"doc_id": d}} for d in ["a", "a", "a", "b", "a", "c", "d"]]
    assert retrieval.diverse(list(range(7)), hits, 5) == [0, 1, 3, 5, 6]
    only_one = [{"_source": {"doc_id": "a"}}] * 4
    assert retrieval.diverse(list(range(4)), only_one, 3) == [0, 1, 2]


def test_header_chunk_gives_way_to_a_content_section_of_its_document():
    header = {"doc_id": "it_kb/vpn_access.md", "section": "VPN Access",
              "text": "VPN Access / VPN Access\n\n# VPN Access\n\nKB article ID: IT-KB-108\n"
                      "Last updated: 2026-06-20\nOwner team: IT"}
    steps = {"doc_id": "it_kb/vpn_access.md", "section": "First-line Troubleshooting",
             "text": "VPN Access / First-line Troubleshooting\n\nConfirm target actually "
                     "requires VPN, verify MFA and device trust."}
    lone = {"doc_id": "it_kb/mfa_reset.md", "text": "# MFA Reset\n\nLast updated: 2026-06-01"}
    hits = [{"_source": h} for h in (header, lone, steps)]
    assert retrieval.header_only(header) and not retrieval.header_only(steps)
    assert retrieval.diverse([0, 1, 2], hits, 5) == [1, 2]  # a lone header stays


def test_get_policy_on_a_handbook_page_says_where_it_lives(env):
    from mcp_server.server import _not_a_policy
    assert _not_a_policy("titles-for-support").startswith(
        "titles-for-support.md is a handbook document, not a policy")
    assert _not_a_policy("promotion").startswith("No policy named")  # playbook stays unnamed
