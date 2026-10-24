"""OpenSearch Serverless client and every index-management call.

Starting code: Lesson 7 `01-opensearch-setup/{client,create_index}.py`. Changes: the
index name comes from `OPENSEARCH_INDEX` instead of the hard-coded `novaops-kb`, and
every function that names an index goes through `checked_index()`. There is no
delete-all helper and no function that takes a bare index name without the guard.

The collection `novaops-rag` also holds the lesson indexes. `novaops-kb` contains
deliberately poisoned documents, so it is refused by name even if config says so.
"""

import functools
import time

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection, helpers
from opensearchpy.exceptions import ConnectionError as OSConnectionError
from opensearchpy.exceptions import TransportError

from agent import config


class IndexRefused(RuntimeError):
    """An index name other than the configured one, or the poisoned lesson index."""


def checked_index(name: str | None = None) -> str:
    """The one guard. Returns the configured index name, or raises IndexRefused."""
    configured = config.opensearch_index()
    if configured == config.FORBIDDEN_INDEX:
        raise IndexRefused(
            f"OPENSEARCH_INDEX is {configured!r}, the Lesson 7 index with poisoned "
            "documents. This project uses its own index (novaops-final-kb)."
        )
    if name is not None and name != configured:
        raise IndexRefused(
            f"Refused to touch index {name!r}: only {configured!r} (OPENSEARCH_INDEX) is allowed."
        )
    return configured


# Same mapping as Lesson 7 plus the metadata this project filters and cites on. NextGen
# Serverless picks the k-NN engine itself and rejects an explicit "engine" field.
# innerproduct on unit-normalized Titan vectors ranks the same as cosine.
INDEX_BODY = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "vector": {
                "type": "knn_vector",
                "dimension": config.EMBED_DIM,
                "method": {"name": "hnsw", "space_type": "innerproduct"},
            },
            "text": {"type": "text"},
            "title": {"type": "text"},
            "audience": {"type": "keyword"},
            "corpus": {"type": "keyword"},
            "source": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "section": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "citation": {"type": "keyword"},
            "subject_employee_id": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "last_updated": {"type": "date", "format": "yyyy-MM-dd"},
        }
    },
}


def _aoss_session() -> boto3.Session:
    """OpenSearch may live in a different AWS account from Bedrock (Lesson 7). Use the
    OPENSEARCH_AWS_* pair when both are set, else the default credentials."""
    key = config.optional("OPENSEARCH_AWS_ACCESS_KEY_ID")
    secret = config.optional("OPENSEARCH_AWS_SECRET_ACCESS_KEY")
    if bool(key) != bool(secret):
        raise config.MissingConfig("Set both OPENSEARCH_AWS_* variables or neither.")
    if key:
        return boto3.Session(aws_access_key_id=key, aws_secret_access_key=secret,
                             region_name=config.aws_region())
    return boto3.Session(region_name=config.aws_region())


def resolve_endpoint() -> str:
    override = config.optional("OPENSEARCH_ENDPOINT")
    if override:
        return override
    name = config.opensearch_collection()
    aoss = _aoss_session().client("opensearchserverless")
    details = aoss.batch_get_collection(names=[name]).get("collectionDetails", [])
    if not details:
        raise RuntimeError(f"OpenSearch collection {name!r} not found.")
    endpoint = details[0].get("collectionEndpoint")
    if not endpoint:
        raise RuntimeError(f"Collection {name!r} is {details[0].get('status')}; wait for ACTIVE.")
    return endpoint


@functools.lru_cache(maxsize=1)
def client() -> OpenSearch:
    host = resolve_endpoint().replace("https://", "").replace("http://", "").rstrip("/")
    auth = AWSV4SignerAuth(_aoss_session().get_credentials(), config.aws_region(), "aoss")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
        # NextGen Serverless scales to zero; the first request after idle can take ~10 s.
        timeout=120,
        max_retries=3,
        retry_on_timeout=True,
    )


def index_exists(name: str | None = None, os_client=None) -> bool:
    return (os_client or client()).indices.exists(index=checked_index(name))


def create_index(name: str | None = None, os_client=None) -> str:
    index = checked_index(name)
    (os_client or client()).indices.create(index=index, body=INDEX_BODY)
    return index


def delete_index(name: str | None = None, os_client=None) -> str:
    index = checked_index(name)
    (os_client or client()).indices.delete(index=index)
    return index


def count(query: dict | None = None, name: str | None = None, os_client=None) -> int:
    body = {"query": query} if query else None
    return (os_client or client()).count(index=checked_index(name), body=body)["count"]


def search(body: dict, name: str | None = None, os_client=None) -> list[dict]:
    return (os_client or client()).search(index=checked_index(name), body=body)["hits"]["hits"]


def ids_for_doc(doc_id: str, name: str | None = None, os_client=None) -> list[str]:
    body = {"size": 500, "query": {"term": {"doc_id": doc_id}}, "_source": False}
    return [hit["_id"] for hit in search(body, name, os_client)]


def bulk_delete(ids: list[str], name: str | None = None, os_client=None) -> None:
    """Delete these document ids from the configured index. Not a delete-all: it only
    takes ids, which ingest finds by doc_id."""
    if not ids:
        return
    index = checked_index(name)
    helpers.bulk(os_client or client(),
                 [{"_op_type": "delete", "_index": index, "_id": i} for i in ids])


def bulk_index(records: list[dict], name: str | None = None, os_client=None) -> None:
    """Serverless vector collections assign document ids themselves and reject a
    caller-supplied _id (Lesson 7), so stable ids live in the doc_id and chunk_id fields."""
    index = checked_index(name)
    helpers.bulk(os_client or client(), [{"_index": index, "_source": r} for r in records])


def wait_for_count(expected: int, timeout: int = 90, name: str | None = None, os_client=None) -> int:
    """Serverless has no manual refresh; poll until the count settles on `expected`."""
    deadline = time.time() + timeout
    seen = -1
    while time.time() < deadline:
        try:
            seen = count(name=name, os_client=os_client)
        except (TransportError, OSConnectionError):
            seen = -1  # a cold collection can fail the first count; keep polling
        if seen == expected:
            return seen
        time.sleep(3)
    return seen
