# CLAUDE.md for novaops-final

NovaOps' company brain: one conversational agent with two scopes. Maya answers HR and
onboarding questions and never writes. Webex handles IT operations and files writes only against
a recorded human approval. The spec is `PROJECT-DESCRIPTION.md` in the course repo. Graders read
this repository and the Langfuse traces and never run the code, so every README claim names a
file and every run leaves a trace.

Read the nested `CLAUDE.md` in `mcp_server/`, `agent/` and `evals/` before editing those
packages.

## Stack

- Python 3.11+. Dependencies are pinned in `requirements.txt`. Adding one means a line there
  plus a reason in the commit message.
- Amazon Bedrock through the Converse API. The model is Nova 2 Lite, and its id comes from
  `BEDROCK_MODEL_ID` (the `us.` inference profile). Embeddings use Titan Text Embeddings V2
  (`BEDROCK_EMBEDDING_MODEL_ID`, 1024 dimensions).
- LangGraph with `AsyncSqliteSaver` from `langgraph-checkpoint-sqlite`. The graph runs async,
  so the checkpointer has to be the async one.
- MCP server on FastMCP 1.x with `mcp<2`. `langchain-mcp-adapters` does not support MCP 2.0,
  and an unpinned install pulls 2.x and breaks every import. The server listens on port 9880,
  which does not collide with the lesson servers on 9876 to 9878.
- OpenSearch Serverless: collection `novaops-rag`, index `novaops-final-kb` (HNSW,
  innerproduct, 1024 dimensions).
- Langfuse Python SDK pinned like Lesson 11 (`langfuse>=4.10,<5`), using `get_client`,
  `propagate_attributes` and `langfuse.langchain.CallbackHandler`.

## Python environment

- Use the shared venv at the Daedalus workspace root:
  `/Users/adimor/.daedalus/workspaces/novaops/.venv` (Python 3.11+). Every worktree runs
  `python`, `pip` and `pytest` from it, for example
  `/Users/adimor/.daedalus/workspaces/novaops/.venv/bin/python -m pytest -q`.
- Never search the disk for another venv, and do not create one per worktree.
- Whenever `requirements.txt` changes, run
  `/Users/adimor/.daedalus/workspaces/novaops/.venv/bin/pip install -r requirements.txt`
  from the worktree that changed it.

## Layout

| Path | Contents |
| --- | --- |
| `mcp_server/` | The MCP server: SQLite access, retrieval over OpenSearch, every tool. The agent reaches it only over MCP. |
| `agent/` | The LangGraph graph, `run_turn()`, `call_model()` and `get_model()`, the typed returns. |
| `evals/` | Golden-session runner, deterministic checks, LLM judges, `targets.py`. |
| `tests/` | pytest suite. Fixtures in `tests/fixtures/` are protected (see Hooks). |
| `scripts/` | `ingest` (documents into `novaops-final-kb`) and `reseed` (restore the SQLite DB). |
| `vendor/`, `renewal/`, `web/` | Optional workflows, added only when their tasks start. `vendor/` imports nothing from `agent/`. |

## Where code comes from

Every module extends lesson lab code. Before building a module:

1. Find the module in the "Starting code" table of the workspace `BRIEF.md`.
2. Read the lesson files it names under `lesson-code/`.
3. Copy them into this repo, then change the copy.
4. Add a row to the README "Starting code" table: our file, the lesson file it came from, and
   what changed.

Never import from `lesson-code/` and never add it to `sys.path`. The repo has to run on its own.

## Hard rules

1. Every model call goes through `call_model()` (or `get_model()` for LangGraph) in
   `agent/model.py`. Never create a Bedrock runtime client anywhere else. boto retries are off,
   and one `should_retry` decides what is transient. Unknown error codes are permanent.
2. Every entry point runs under Langfuse. Each trace carries `session_id` = thread id and
   `user_id` = caller id, with `request_id` and the terminal `status` in metadata. Spans are
   named `classify`, `scope`, `tool:<name>`, `write_gate`, `answer`. Never remove or bypass the
   Langfuse callback on the compiled graph, and call `flush()` before the process exits. No
   test catches a missing callback, which is why this rule lives here.
3. Authorization, approvals and idempotency are enforced inside the MCP server, never by a
   prompt and never by a graph node alone. Gated writes require a recorded approval;
   `request_approval` only creates a pending row and is idempotent (rules in
   `mcp_server/CLAUDE.md`). The model never sees a tool that records an approval decision.
   `record_approval_decision` is called by application code only.
4. The checkpointer is durable. `InMemorySaver` and `MemorySaver` are banned everywhere, tests
   included. Tests use `AsyncSqliteSaver` on a temporary file.
5. The entry point is
   `run_turn(message, *, caller_id, user_group, thread_id, request_id=None)`. `caller_id`,
   `user_group` and `request_id` travel in graph config, and the tool wrapper injects them into
   MCP calls. They are never parsed from message text and never appear in a tool schema the
   model sees. `run_turn()` creates `request_id` when the caller passes none, and a resume
   reuses the one stored in the checkpoint (details in `agent/CLAUDE.md`).
6. Develop against `GOLDEN-DATASETS.json` only. Do not open, grep, copy or summarize
   `EVALUATION-INPUTS.yaml` until task #11. It runs once, at the end, and nothing is tuned
   against it. A deny rule and the hook enforce this.
7. `repos/novaops-course` and `lesson-code/` are read-only.
8. OpenSearch: read and write only the index named by `OPENSEARCH_INDEX` (`novaops-final-kb`)
   in collection `novaops-rag`. Never read, write, reuse or delete `novaops-kb` or any other
   lesson index; `novaops-kb` holds deliberately poisoned documents. Index-management code
   refuses any other name.
9. Never commit `.env`, AWS or Langfuse keys, approval tokens, or restricted source documents.
   Never copy any `lesson-code/**/.env` or the token inside the course README's clone URL.
10. Any UI or CLI calls `run_turn()`. There is no second path into the workflow.

## Status

Every run ends in exactly one `status`: `completed`, `blocked`, `pending`, `needs_human` or
`failed`. Code sets it from what the tools returned, never from model text. The answer text must
agree with it: a `pending` run never says access was granted, and "actions taken" lists only
tool results that returned a row id. What happened to the business request (granted, refused,
recommended) goes in the typed return object, not in `status`.

## Config

- Load config with `load_dotenv(find_dotenv())`. `find_dotenv()` walks up the directory tree,
  so the `.env` at the workspace root is found from any worktree. Never pass a relative path to
  `.env` and never hard-code a model id, region, index name or key.
- If a required variable is missing, stop and ask for it. Do not guess or default it.
- `.env.example` lists every variable with no values. A new variable goes there in the same
  commit that first reads it.
- The clock is pinned to 2026-07-01 through `NOVAOPS_TODAY`. Code reads the date from config,
  never from `date.today()`. (The dataset's own `as_of_date` is 2026-07-02. The spec still
  says 2026-07-01.)
- Run `python -m scripts.reseed` before each independent case. A multi-turn session is one
  thread on one database and is not reseeded between turns.

## Definition of done

A change is done when `python -m pytest -q` is green and the golden-session check for the
behaviour you touched passes (`S1` is the smoke test). A transcript that looks right is not
evidence. Say which check you ran and paste its result in the journal entry or PR.

## Hooks and permissions

`.claude/settings.json` holds the allow and deny lists and wires
`.claude/hooks/guard_protected_paths.py` as a `PreToolUse` hook. The hook refuses:

- edits to `evals/targets.py`, `tests/fixtures/**`, `.claude/**` and the dataset files;
- edits to anything under `novaops-course/` or `lesson-code/`;
- any read of `EVALUATION-INPUTS.yaml`, including through Bash or Grep.

If the hook blocks you, do not work around it. Stop and say which file you need and why. The
hook is pattern matching and can be bypassed; the deny rules and diff review are the layers
that hold.

## Style

- Comment the non-obvious AI and SDK decisions: why these retry codes, why a pre-filter, what
  a repair attempt costs. Do not comment code that explains itself.
- One small function per job. A bare `except:` or `except Exception:` that returns a default
  is not allowed. If you cannot name the exception, you have not handled it.
- A parse failure is never silent. Repair once, then raise or record a named failure.
- Commit messages and docs follow the same plain style as this file.
