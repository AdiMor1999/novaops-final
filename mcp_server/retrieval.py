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
- A second clause in the same filter guards employment documents (task #4). An
  `employment` chunk with a `subject_employee_id` is readable only by that subject, by
  a UG_HR caller and by the subject's manager. Employment chunks with no subject (the
  equipment responsibility form) stay open. Spec §12 leaves unlabelled collections
  open to everyone; this is a deliberate deviation, because a regular employee must not
  read another person's offer letter (Lesson 10 homework, milestone 2).
- A non-manager's HR search always carries the same `access_note`, whatever the query
  and whether or not a playbook chunk would have matched, so the note cannot reveal
  what the playbook contains.
- `contracts` is searchable through the knowledge base: the Webex contract is the
  evidence for the 42-of-40 seat blocker.
- When a result holds both a memo and a policy, a `precedence_note` says which is
  newer. Newer or more specific guidance wins (the 7,500 memo over the 5,000 policy).
- Task #5: a document's header chunk (its title and `Key: value` lines, nothing else)
  gives up its place when a content section of the same document is a candidate. For
  "can you find the VPN steps?" (S8 turn 1) the header of vpn_access.md ranked first,
  and Nova Lite answered from it ("the steps are in the article, last updated
  2026-06-20") on four runs out of five. The header's facts are already fields on
  every chunk (`last_updated`, `source`).
"""

import logging
import re
from collections.abc import Callable

from agent import config, heuristics
from agent.model import call_model, embed_text, forced_tool_input
from mcp_server import opensearch
from mcp_server.db import CallerAccess, caller_access

log = logging.getLogger(__name__)

CANDIDATES = 20  # retrieve wide
TOP_K = 5  # rerank narrow
# At most this many of the TOP_K chunks from one document, so a long document cannot
# fill every slot: S9 turn 1 got five chunks from two handbook files and missed
# titles-for-support.md, the page about the caller's own role family.
MAX_PER_DOC = 2
RESTRICTED_AUDIENCE = "manager"
RESTRICTED_CORPUS = "manager_playbook"
EMPLOYMENT_CORPUS = "employment"
PLAYBOOK_NOTE = "The manager playbook is not searchable for this caller."

KB_CORPORA = ("it_kb", "policies", "internal_memos", "contracts")
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


def employment_clause(subjects: tuple[str, ...] | None) -> dict | None:
    """Excludes employment chunks about someone the caller may not read. None for HR
    (every subject is readable). Chunks without a subject never match, so they stay
    open. An empty tuple (unknown caller) excludes every subject's documents."""
    if subjects is None:
        return None
    clause: dict = {"filter": [{"term": {"corpus": EMPLOYMENT_CORPUS}},
                               {"exists": {"field": "subject_employee_id"}}]}
    if subjects:
        clause["must_not"] = [{"terms": {"subject_employee_id": list(subjects)}}]
    return {"bool": clause}


def build_filter(corpora: tuple[str, ...], access: CallerAccess) -> dict:
    clause: dict = {"must": [{"terms": {"corpus": list(corpora)}}]}
    must_not = access_clauses(access.manager)
    employment = employment_clause(access.employment_subjects)
    if employment and EMPLOYMENT_CORPUS in corpora:
        must_not.append(employment)
    if must_not:
        clause["must_not"] = must_not
    return {"bool": clause}


def build_search_body(vector: list[float], corpora: tuple[str, ...], access: CallerAccess,
                      k: int = CANDIDATES) -> dict:
    """The OpenSearch request body. Pure, so tests assert on it without a cluster."""
    return {
        "size": k,
        "query": {"knn": {"vector": {
            "vector": vector,
            "k": k,
            "filter": build_filter(corpora, access),
        }}},
        "_source": SOURCE_FIELDS,
    }


def body_for_caller(query: str, corpora: tuple[str, ...], caller_id: str,
                    embed: Callable[[str], list[float]] = embed_text) -> dict:
    return build_search_body(embed(query), corpora, caller_access(caller_id))


def access_summary(corpora: tuple[str, ...], access: CallerAccess) -> dict:
    """What the filter excluded, in the tool result, so the trace shows it per call."""
    subjects = access.employment_subjects
    summary = {
        "caller_is_manager": access.manager,
        "caller_is_hr": access.hr,
        "manager_playbook": "searchable" if access.manager else "excluded by pre-filter",
    }
    if EMPLOYMENT_CORPUS in corpora:
        summary["employment_subjects"] = "all" if subjects is None else list(subjects)
    return summary


def precedence_note(chunks: list[dict]) -> str | None:
    """Code states which guidance is newer, so the answer does not have to infer it
    from two dates (S2 turn 9: the 2026-06-25 freeze memo over the procurement policy)."""
    memos = [c for c in chunks if c.get("doc_type") == "memo" and c.get("last_updated")]
    policies = [c for c in chunks if c.get("doc_type") == "policy"]
    if not memos or not policies:
        return None
    newest = max(memos, key=lambda c: c["last_updated"])
    older = sorted({p["source"] for p in policies
                    if (p.get("last_updated") or "") < newest["last_updated"]})
    if not older:
        return None
    return (f"{newest['source']} ({newest['last_updated']}) is newer than "
            f"{', '.join(older)}. Where they disagree, the newer, more specific memo wins.")


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


_TITLE_LINE = re.compile(r"^#\s")
_META_LINE = re.compile(r"^[A-Z][A-Za-z ]{1,40}:\s")


def header_only(source: dict) -> bool:
    """True for a chunk that holds only the document's title and `Key: value` lines
    ('KB article ID: IT-KB-108', 'Last updated: 2026-06-20'). The first line of every
    stored chunk is its 'Title / Section' breadcrumb (mcp_server/documents.py), not
    content."""
    lines = [line.strip() for line in (source.get("text") or "").splitlines() if line.strip()]
    section = source.get("section")
    if lines and section and lines[0].endswith(f" / {section}"):
        lines = lines[1:]
    return bool(lines) and all(_TITLE_LINE.match(l) or _META_LINE.match(l) for l in lines)


def diverse(order: list[int], hits: list[dict], top_k: int,
            per_doc: int = MAX_PER_DOC, demote: bool = True) -> list[int]:
    """The first top_k of `order`, at most `per_doc` per document; if fewer documents
    than that matched, the remaining slots go to the next chunks in order. With
    `demote` (the `title_chunk_demotion` heuristic), a header-only chunk is skipped
    when a content chunk of its document is among the candidates."""
    if demote:
        with_content = {h["_source"].get("doc_id") for h in hits
                        if not header_only(h["_source"])}
        order = [i for i in order if not (header_only(hits[i]["_source"])
                                          and hits[i]["_source"].get("doc_id") in with_content)]
    picked, per = [], {}
    for i in order:
        doc = hits[i]["_source"].get("doc_id")
        if per.get(doc, 0) < per_doc:
            picked.append(i)
            per[doc] = per.get(doc, 0) + 1
        if len(picked) == top_k:
            return picked
    rest = [i for i in order if i not in picked]
    return picked + rest[:top_k - len(picked)]


def search(query: str, corpora: tuple[str, ...], caller_id: str, top_k: int = TOP_K,
           embed: Callable[[str], list[float]] = embed_text,
           rerank: Callable[[str, list[dict]], dict[int, float]] = rerank_scores) -> dict:
    access = caller_access(caller_id)
    body = build_search_body(embed(query), corpora, access)
    hits = opensearch.search(body)
    result = {
        "query": query,
        "index": config.opensearch_index(),
        "corpora": list(corpora),
        "caller_is_manager": access.manager,
        "access": access_summary(corpora, access),
        "candidates": len(hits),
    }
    if RESTRICTED_CORPUS in corpora and not access.manager:
        result["access_note"] = PLAYBOOK_NOTE
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

    demote = heuristics.enabled("title_chunk_demotion")
    picked = diverse(order, hits, top_k, demote=demote)
    if demote and picked != diverse(order, hits, top_k, demote=False):
        # This process is not the agent's; the name travels back in the result and
        # agent/mcp_tools.py records it on the turn (agent/heuristics.py).
        result[heuristics.SERVER_KEY] = ["title_chunk_demotion"]
    chunks = [_chunk_result(hits[i], hits[i]["_score"], scores.get(i)) for i in picked]
    result.update(reranked=reranked, count=len(chunks), chunks=chunks)
    if note := precedence_note(chunks):
        result["precedence_note"] = note
    if fallback:
        result["rerank_fallback_reason"] = fallback
    return result
