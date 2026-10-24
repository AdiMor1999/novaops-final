"""The access clause must sit inside knn.<field>.filter (pre-filter). Assert on the
request body, not on an answer."""

from mcp_server.retrieval import HR_CORPORA, body_for_caller
from tests.conftest import stub_embed

RESTRICTED = [{"term": {"audience": "manager"}}, {"term": {"corpus": "manager_playbook"}}]


def knn_filter(body):
    (field,) = body["query"]["knn"].keys()
    return body["query"]["knn"][field]["filter"]


def test_non_manager_body_has_must_not_inside_knn_filter(seeded):
    body = body_for_caller("promotion process", HR_CORPORA, "E001", embed=stub_embed)
    assert knn_filter(body)["bool"]["must_not"] == RESTRICTED
    assert "post_filter" not in body and "filter" not in body["query"]


def test_manager_body_has_no_must_not(seeded):
    body = body_for_caller("promotion process", HR_CORPORA, "E018", embed=stub_embed)
    assert "must_not" not in knn_filter(body)["bool"]


def test_corpus_scope_is_inside_the_same_filter(seeded):
    body = body_for_caller("vpn", ("it_kb",), "E001", embed=stub_embed)
    assert knn_filter(body)["bool"]["must"] == [{"terms": {"corpus": ["it_kb"]}}]
