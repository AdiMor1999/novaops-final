from agent import config
from mcp_server.documents import build_chunks, chunk_document, parse_document


def test_policy_metadata_from_inline_lines(env):
    doc = parse_document(config.documents_dir() / "policies" / "equipment_policy.md", "policies")
    assert doc.corpus == "policies"
    assert doc.last_updated == "2026-06-01"
    assert doc.doc_type == "policy"
    assert doc.audience is None
    chunks = chunk_document(doc)
    labels = [c.citation for c in chunks]
    assert "[equipment_policy.md#Personal laptops]" in labels
    record = chunks[0].record("titan")
    assert "audience" not in record and record["doc_id"] == "policies/equipment_policy.md"
    assert record["chunk_id"] == "policies/equipment_policy.md#0"


def test_handbook_front_matter_parsed_and_stripped(env):
    doc = parse_document(config.documents_dir() / "handbook" / "benefits-and-perks.md", "handbook")
    assert doc.audience == "all"
    assert doc.last_updated == "2025-03-07"
    assert not doc.body.startswith("---")
    assert all("audience:" not in c.text and "last_updated:" not in c.text
               for c in chunk_document(doc))


def test_memo_date_and_doc_type(env):
    doc = parse_document(config.documents_dir() / "internal_memos" / "saas_renewal_freeze_q3.md",
                         "internal_memos")
    assert doc.doc_type == "memo" and doc.last_updated


def test_employment_subject_matched_by_name_only(seeded):
    from mcp_server.db import employee_names
    chunks = build_chunks(config.documents_dir(), employee_names(seeded))
    subjects = {c.doc.source: c.subject_employee_id for c in chunks if c.doc.corpus == "employment"}
    assert subjects["rachel_stein_customer_success_access_profile.md"] == "E010"
    assert subjects["sara_ben_david_hr_access_profile.md"] == "E004"
    assert subjects["contractor_access_addendum.md"] is None
    assert len({c.doc.doc_id for c in chunks}) == 102
