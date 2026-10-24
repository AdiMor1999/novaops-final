import pytest

from mcp_server import opensearch
from mcp_server.opensearch import IndexRefused
from scripts import ingest


class ExplodingClient:
    """Any attribute access means the guard let a call through."""
    def __getattr__(self, name):
        raise AssertionError(f"OpenSearch client touched ({name}) for a refused index")


@pytest.mark.parametrize("name", ["novaops-kb", "some-other-index"])
def test_create_delete_ingest_refuse_other_names(env, name):
    fake = ExplodingClient()
    with pytest.raises(IndexRefused):
        opensearch.create_index(name, os_client=fake)
    with pytest.raises(IndexRefused):
        opensearch.delete_index(name, os_client=fake)
    with pytest.raises(IndexRefused):
        ingest.ingest(name, os_client=fake)
    with pytest.raises(IndexRefused):
        ingest.replace_documents([], name, os_client=fake)


def test_poisoned_index_refused_even_when_configured(env, monkeypatch):
    monkeypatch.setenv("OPENSEARCH_INDEX", "novaops-kb")
    fake = ExplodingClient()
    with pytest.raises(IndexRefused):
        opensearch.checked_index()
    with pytest.raises(IndexRefused):
        opensearch.create_index(os_client=fake)
    with pytest.raises(IndexRefused):
        ingest.ingest(os_client=fake)


def test_configured_name_passes(env):
    assert opensearch.checked_index("novaops-final-kb") == "novaops-final-kb"
