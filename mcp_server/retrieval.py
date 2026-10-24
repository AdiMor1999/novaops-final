"""Access-filtered retrieval with a listwise rerank. Runs inside the MCP server.

Starting code: Lesson 7 `02-filtering/retrieval.py` (`build_filter`, the filter inside the
k-NN block) and `03-reranking/{retrieval,reranker}.py` (retrieve wide, rerank narrow).

What changed from Lesson 7:

- Access is decided by manager status read from `employees.manager_id`, not by a role
  string. Lesson 7 kept `audience: all` for employees, which would also hide the 70
  documents that have no `audience` at all. Here a non-manager gets a `must_not` on
  `audience: manager` (and on the `manager_playbook` corpus, in case a chunk lost its
  audience field), and unlabelled documents stay readable (spec §12).
- The clause is a pre-filter: it sits inside `knn.vector.filter`, so the engine applies
  it during the HNSW walk. A restricted chunk never enters the candidate list, so it can
  never reach the reranker or the answer model. A post-filter would already have leaked.
"""

import logging
from collections.abc import Callable

from agent import config
from agent.model import call_model, embed_text, forced_tool_input
from mcp_server import opensearch
from mcp_server.db import is_manager

log = logging.getLogger(__name__)

CANDIDATES = 20  # retrieve wide
TOP_K = 5  # rerank narrow
RESTRICTED_AUDIENCE = "manager"
RESTRICTED_CORPUS = "manager_playbook"

KB_CORPORA = ("it_kb", "policies", "internal_memos")
HR_CORPORA = ("handbook", "employment", "policies", "internal_memos", "manager_playbook")

SOURCE_FIELDS = ["text", "source", "section", "doc_id", "chunk_id", "corpus", "audience",
                 "last_updated", "doc_type", "citation", "title", "subject_employee_id"]


class RerankParseFailed(ValueError):
    """The rerank call returned no usable scores. Retrieval falls back to vector order
    and marks the result `reranked: false`; it does not fail silently."""


def access_clauses(manager: bool) -> list[dict]:
    if manager:
        return []
    return [{"term": {"audience": RESTRICTED_AUDIENCE}}, {"term": {"corpus": RESTRICTED_CORPUS}}]


def build_filter(corpora: tuple[str, ...], manager: bool) -> dict:
    clause: dict = {"must": [{"terms": {"corpus": list(corpora)}}]}
    must_not = access_clauses(manager)
    if must_not:
        clause["must_not"] = must_not
    return {"bool": clause}


def build_search_body(vector: list[float], corpora: tuple[str, ...], manager: bool,
                      k: int = CANDIDATES) -> dict:
    """The OpenSearch request body. Pure, so tests assert on it without a cluster."""
    return {
        "size": k,
        "query": {"knn": {"vector": {
            "vector": vector,
            "k": k,
            "filter": build_filter(corpora, manager),
        }}},
        "_source": SOURCE_FIELDS,
    }


def body_for_caller(query: str, corpora: tuple[str, ...], caller_id: str,
                    embed: Callable[[str], list[float]] = embed_text) -> dict:
    return build_search_body(embed(query), corpora, is_manager(caller_id))


_RERANK_TOOL = {
    "toolSpec": {
        "name": "rank",
        "description": "Score every candidate passage for relevance to the query.",
        "inputSchema": {"json": {
            "type": "object", "required": ["scores"],
            "properties": {"scores": {
                "type": "array",
                "description": "One entry per candidate, referenced by the [index] shown.",
                "items": {
                    "type": "object", "required": ["index", "score"],
                    "properties": {
                        "index": {"type": "integer"},
                        "score": {"type": "number", "description": "Relevance 0.0 to 1.0."},
                    },
                },
            }},
        }},
    }
}


def rerank_scores(query: str, candidates: list[dict]) -> dict[int, float]:
    """One forced `rank` call scoring every candidate (Lesson 7 reranker). The whole
    chunk is sent: scoring a truncated fragment would rank on text the answer never sees."""
    listing = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(candidates))
    response = call_model(
        [{"role": "user", "content": [{"text":
            f"Query: {query}\n\nScore each candidate passage from 0.0 to 1.0 for how well "
            f"it helps answer the query. Use the exact [index] shown.\n\n{listing}"}]}],
        tools=[_RERANK_TOOL], tool_choice="rank", max_tokens=1500, name="rerank",
    )
    payload = forced_tool_input(response, "rank")
    if not payload or not isinstance(payload.get("scores"), list):
        raise RerankParseFailed("rank tool was not called or returned no scores")
    scores = {}
    for entry in payload["scores"]:
        index, score = entry.get("index"), entry.get("score")
        if isinstance(index, int) and 0 <= index < len(candidates) and isinstance(score, (int, float)):
            scores[index] = max(0.0, min(1.0, float(score)))
    if not scores:
        raise RerankParseFailed("no score referenced a valid candidate index")
    return scores


def _chunk_result(hit: dict, vector_score: float, rerank_score: float | None) -> dict:
    src = hit["_source"]
    return {
        "citation": src.get("citation"),
        "source": src.get("source"),
        "section": src.get("section"),
        "doc_id": src.get("doc_id"),
        "chunk_id": src.get("chunk_id"),
        "corpus": src.get("corpus"),
        "doc_type": src.get("doc_type"),
        "last_updated": src.get("last_updated"),
        "audience": src.get("audience"),
        "subject_employee_id": src.get("subject_employee_id"),
        "vector_score": round(vector_score, 4),
        "rerank_score": rerank_score,
        "text": src.get("text"),
    }


def search(query: str, corpora: tuple[str, ...], caller_id: str, top_k: int = TOP_K,
           embed: Callable[[str], list[float]] = embed_text,
           rerank: Callable[[str, list[dict]], dict[int, float]] = rerank_scores) -> dict:
    manager = is_manager(caller_id)
    body = build_search_body(embed(query), corpora, manager)
    hits = opensearch.search(body)
    result = {
        "query": query,
        "index": config.opensearch_index(),
        "corpora": list(corpora),
        "caller_is_manager": manager,
        "candidates": len(hits),
    }
    if not hits:
        return {**result, "reranked": False, "count": 0, "chunks": [],
                "note": "No document matched this query in the searched collections."}

    candidates = [h["_source"] for h in hits]
    try:
        scores = rerank(query, candidates)
        order = sorted(range(len(hits)), key=lambda i: scores.get(i, 0.0), reverse=True)
        reranked, fallback = True, None
    except RerankParseFailed as exc:
        log.warning("rerank fell back to vector order: %s", exc)
        scores, order = {}, list(range(len(hits)))
        reranked, fallback = False, str(exc)

    chunks = [_chunk_result(hits[i], hits[i]["_score"], scores.get(i)) for i in order[:top_k]]
    result.update(reranked=reranked, count=len(chunks), chunks=chunks)
    if fallback:
        result["rerank_fallback_reason"] = fallback
    return result
