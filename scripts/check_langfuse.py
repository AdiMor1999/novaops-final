"""Check that this machine can reach the Langfuse project before a run records nothing.

    python -m scripts.check_langfuse

Starting code: Lesson 11 `01-langfuse-setup/01-connect-langfuse.py`. Both failure modes
are silent at run time: with no keys the client is a no-op, and with wrong keys the
exporter logs a 401 on a background thread and drops the batch.
"""

import os
import sys

from langfuse import get_client

from agent import config

REQUIRED = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")


def main() -> None:
    for name in REQUIRED:
        config.require(name)  # raises MissingConfig with the variable's name
    # auth_check() raises on rejected credentials instead of returning False.
    try:
        ok = get_client().auth_check()
    except Exception as exc:  # the SDK raises several unrelated types; we only report it
        sys.exit(f"Langfuse rejected these keys ({type(exc).__name__}). Check that the key pair "
                 f"is from one project and that LANGFUSE_BASE_URL is that project's region.")
    if not ok:
        sys.exit("Langfuse auth_check returned False.")
    print(f"[connected] {os.environ['LANGFUSE_BASE_URL']}")
    print("[safe] credentials were read from .env and never printed")


if __name__ == "__main__":
    main()
