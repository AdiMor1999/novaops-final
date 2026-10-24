"""Chunk, embed and index every NovaOps document into the project index.

    python -m scripts.reseed        # ingest reads employee names from the database
    python -m scripts.create_index
    python -m scripts.ingest

Starting code: Lesson 7 `01-opensearch-setup/ingest.py` and `02-filtering/ingest.py`.
Changes: all seven collections, metadata from front matter or inline header lines
(`mcp_server/documents.py`), chunks on `##` sections, stable `doc_id` / `chunk_id`, the
embedding model id stored on every chunk, and `subject_employee_id` on employment files.

Re-runnable: Serverless vector collections reject caller-supplied `_id`s, so a document
cannot be overwritten in place. Each document's old chunks are found by `doc_id` and
deleted before its new chunks go in. A second run leaves the count unchanged.
"""

import logging
import sys
from collections import Counter

from agent import config
from agent.model import embed_text
from mcp_server import db, opensearch
from mcp_server.documents import CORPORA, Chunk, build_chunks


def embed_records(chunks: list[Chunk], embed=embed_text) -> list[dict]:
    model = config.embedding_model_id()
    records = []
    for i, chunk in enumerate(chunks, start=1):
        record = chunk.record(model)
        record["vector"] = embed(chunk.text)
        records.append(record)
        if i % 50 == 0 or i == len(chunks):
            print(f"  embedded {i}/{len(chunks)}")
    return records


def replace_documents(records: list[dict], index: str | None = None, os_client=None) -> None:
    """Delete each document's previous chunks, then bulk-index the new ones."""
    index = opensearch.checked_index(index)
    stale = []
    for doc_id in sorted({r["doc_id"] for r in records}):
        stale += opensearch.ids_for_doc(doc_id, index, os_client)
    opensearch.bulk_delete(stale, index, os_client)
    print(f"  removed {len(stale)} chunks from a previous run")
    opensearch.bulk_index(records, index, os_client)


def ingest(index: str | None = None, os_client=None, embed=embed_text) -> Counter:
    index = opensearch.checked_index(index)  # refuse before any embedding is paid for
    if not opensearch.index_exists(index, os_client):
        sys.exit(f"Index {index!r} does not exist. Run `python -m scripts.create_index`.")
    chunks = build_chunks(config.documents_dir(), db.employee_names())
    files = Counter(c.doc.corpus for c in {c.doc.doc_id: c for c in chunks}.values())
    print(f"Chunked {sum(files.values())} files into {len(chunks)} chunks. Embedding...")
    records = embed_records(chunks, embed)
    replace_documents(records, index, os_client)
    settled = opensearch.wait_for_count(len(records), name=index, os_client=os_client)
    per_corpus = Counter(r["corpus"] for r in records)
    print(f"\n{'corpus':18} {'files':>5} {'chunks':>6}")
    for corpus in CORPORA:
        print(f"{corpus:18} {files[corpus]:>5} {per_corpus[corpus]:>6}")
    print(f"{'total':18} {sum(files.values()):>5} {len(records):>6}")
    print(f"Index {index!r} now reports {settled} chunks.")
    return per_corpus


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="  %(levelname)s %(message)s")
    ingest()


if __name__ == "__main__":
    main()
