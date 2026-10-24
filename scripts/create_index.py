"""Create the project index (HNSW, innerproduct, 1024 dimensions) in `novaops-rag`.

    python -m scripts.create_index              # create if missing
    python -m scripts.create_index --recreate   # drop and recreate the project index

Starting code: Lesson 7 `01-opensearch-setup/create_index.py`. Changes: the name comes
from OPENSEARCH_INDEX through `checked_index()`, the mapping carries this project's
metadata fields, and an existing index is kept unless `--recreate` is passed.
"""

import argparse

from mcp_server import opensearch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recreate", action="store_true",
                        help="delete and recreate the configured index")
    args = parser.parse_args()

    index = opensearch.checked_index()
    if opensearch.index_exists():
        if not args.recreate:
            print(f"Index {index!r} exists ({opensearch.count()} chunks). Nothing to do.")
            return
        opensearch.delete_index()
        print(f"Deleted {index!r}.")
    opensearch.create_index()
    print(f"Created {index!r} (HNSW, innerproduct, 1024-dim).")


if __name__ == "__main__":
    main()
