# novaops-final

NovaOps' company brain: one conversational agent with two scopes. Maya answers HR and
onboarding questions from the NovaOps documents and database, cites every claim and never
writes. Webex handles IT operations: it diagnoses access blockers, reuses existing tickets and
requests, and files a write only against a recorded human approval.

This README is a skeleton. Each row in the tables below gets filled in by the task that builds
the file. A claim without a file is not made.

## Claims and where to check them

| Claim | File | Status |
| --- | --- | --- |
| Project conventions and hard rules | `CLAUDE.md`, `mcp_server/CLAUDE.md`, `agent/CLAUDE.md`, `evals/CLAUDE.md` | done (task #1) |
| Agent edits to eval targets, fixtures and the held-out inputs are refused | `.claude/settings.json`, `.claude/hooks/guard_protected_paths.py` | done (task #1) |
| One function calls Bedrock, with bounded retries that fail closed | `agent/model.py`, `tests/test_retry.py` | done (task #2) |
| Documents ingested with `audience`, `corpus`, `source`, `last_updated`, `doc_type`, `section` | `scripts/ingest.py`, `mcp_server/documents.py`, `tests/test_ingest_metadata.py` | done (task #2) |
| Database is a file and reseeds before each case | `scripts/reseed.py`, `mcp_server/db.py` | done (task #2) |
| Tools sit behind an MCP server; the agent never imports them | `mcp_server/server.py`, `agent/mcp_tools.py` | done (task #2) |
| Caller id, group and `request_id` travel in MCP `_meta`, never in a tool schema | `agent/mcp_tools.py`, `mcp_server/context.py`, `tests/test_boundary.py` | done (task #2) |
| The model has no tool that records an approval decision | `mcp_server/writes.py` (`record_approval_decision`), `tests/test_boundary.py` | done (task #2) |
| Gated writes are refused without a recorded approval; a replay returns the existing row | `mcp_server/writes.py`, `tests/test_writes.py` | done (task #2) |
| Index code refuses every index but `OPENSEARCH_INDEX`, and `novaops-kb` by name | `mcp_server/opensearch.py`, `tests/test_index_guard.py` | done (task #2) |
| Every trace carries `session_id`, `user_id`, `request_id` and `status` | `agent/run.py`, `agent/mcp_tools.py` | done (task #2) |
| Classify is the router; code derives the scope (`maya`/`webex`) from the intent | `agent/graph.py`, `agent/policy.py` (`SCOPE_OF`), `tests/test_policy.py` | done (task #3) |
| Each turn binds only its intent's tools; no loadout holds a write or `request_approval` | `agent/policy.py` (`LOADOUTS`, `visible_tools`), `tests/test_policy.py`, `tests/test_graph.py` | done (task #3) |
| A write becomes visible only through `release_write()`, which opens on an approved, unused, unexpired approval row | `agent/gate.py`, `tests/test_approval.py` | done (task #3) |
| Tool calls are capped at 6 per turn; one `rearm` reopens reads only; a tool not visible this turn is not run | `agent/graph.py` (`run_tools`, `rearm`), `tests/test_graph.py` | done (task #3) |
| A planner parse failure is repaired once, then raises `ParseFailed`; no default plan | `agent/structured.py`, `tests/test_graph.py` | done (task #3) |
| Durable checkpointer injected from outside the graph; a paused approval survives a killed process and resumes in another with the same `request_id` and trace | `agent/run.py` (`open_checkpointer`), `tests/test_approval.py`, `tests/approval_process.py`, `tests/test_banned.py` | done (task #3) |
| Every graph tool call carries `caller_id`, `user_group` and `request_id` in MCP `_meta` | `agent/graph.py`, `agent/mcp_tools.py`, `tests/test_graph.py` | done (task #3) |
| `run_turn()` returns one `TurnOutcome` shape; `status` is set by code | `agent/outcome.py`, `agent/run.py`, `tests/test_run_status.py` | done (task #3) |
| Golden illustrative tool names are mapped to ours in one module | `evals/tool_map.py`, `tests/test_s1_live.py` | done (task #3) |
| `S1-quick-policy` passes end to end (intent, tools, facts, turn-1 citation) | `tests/test_s1_live.py` | done (task #3) |
| `manager_playbook` is unreachable for non-managers (pre-filter) | `mcp_server/retrieval.py`, `tests/test_prefilter.py`, `tests/test_live_leak.py` | filter done (task #2); Maya cases in task #4 |
| Maya returns `OnboardingChecklist` and a typed `Handoff` | `agent/models.py` | planned (task #4) |
| Webex reuses `T001` and `AR001` and pauses for approval | `mcp_server/server.py`, `agent/graph.py` | planned (task #5) |
| A write is released only against a recorded approval, exactly once | `mcp_server/writes.py`, `tests/` | planned (task #5) |
| Deterministic checks and LLM judges score the golden sessions | `evals/` | planned (task #6) |
| Evaluation run and submission | `SUBMISSION.md` | planned (task #11) |

## Stack

- Python 3.11+
- Amazon Bedrock Converse: Nova 2 Lite for the agent, Titan Text Embeddings V2 for retrieval.
  Model ids come from the environment.
- LangGraph with `AsyncSqliteSaver` (`langgraph-checkpoint-sqlite`)
- MCP server on FastMCP 1.x (`mcp<2`), port 9880, loaded with `langchain-mcp-adapters`
- OpenSearch Serverless: collection `novaops-rag`, index `novaops-final-kb`
- Langfuse Python SDK (`langfuse>=4.10,<5`, as in Lesson 11)

## Running

Configuration variables are listed in `.env.example`. The real `.env` sits at the workspace
root and is found with `find_dotenv()`.

```sh
python -m pip install -r requirements.txt
python -m scripts.check_langfuse      # keys and connectivity, before anything records nothing
python -m scripts.reseed              # rebuild NOVAOPS_DB_PATH from the course seed; prints row counts
python -m scripts.create_index        # novaops-final-kb, once (--recreate to rebuild it)
python -m scripts.ingest              # 102 files, 7 collections; prints chunks per corpus
python -m mcp_server                  # MCP server on MCP_SERVER_URL (port 9880)
python -m agent.ask "What's the NovaOps policy on using a personal laptop for work?" \
    --caller E010 --group UG_REGULAR --thread t1     # prints scope, visible tools and the trace URL
python -m pytest -q                   # integration tests skip without AWS credentials
```

`agent.ask` calls `run_turn()`; there is no other way into the workflow.

## Starting code

Every module started as a copy of lesson lab code and was changed here. Nothing imports from
the lesson folders. Rows are added as each module is copied.

| Our file | Copied from (`lesson-code/`) | What changed |
| --- | --- | --- |
| `.claude/hooks/guard_protected_paths.py` | `lesson-12-eval-loop-engineering/code/02-coding-loop/.claude/hooks/block_test_edits.py` | Protects `evals/targets.py`, `tests/fixtures/`, `.claude/`, the dataset files and the reference repos instead of every test file. Also blocks reads of `EVALUATION-INPUTS.yaml`. |
| `.claude/settings.json` | `lesson-12-eval-loop-engineering/code/03-eval-loop/.claude/settings.json` | Allow list for headless project work, deny rules for `.env`, the held-out inputs and protected paths. No activity log. |
| `CLAUDE.md` and nested `CLAUDE.md` files | `lesson-12-eval-loop-engineering/code/02-coding-loop/CLAUDE.md`, `03-eval-loop/CLAUDE.md`, `lesson-code/*-sdd/CLAUDE.md-example.html` | Rewritten for this project's stack, layout and hard rules. |
| `agent/model.py` | `lesson-12-eval-loop-engineering/samples/02-coding-loop/agent-hardened/agent.py` (`get_model`, `should_retry`, `call_with_retry`); `lesson-07-rag-advanced/code/01-opensearch-setup/client.py` (`embed_text`) | Adds `call_model()` over Converse with a Langfuse generation per call. boto retries off, one tenacity layer over `should_retry`, `maxTokens` always set. `embed_text` moved here so the server has no Bedrock client of its own. |
| `agent/config.py` | none (new) | One loader. A missing variable raises `MissingConfig` with its name. `today()` reads `NOVAOPS_TODAY`. |
| `mcp_server/db.py` | `lesson-10-context-engineering/code/server/db.py` | File DB instead of `:memory:`, a connection per call, dates from `NOVAOPS_TODAY`, seat-assignment gap stated on every subscription row. New reads: `get_access_requests`, `get_approvals`, `get_team`, `is_manager`. Writes moved to `writes.py`. |
| `mcp_server/server.py` | `lesson-10-context-engineering/code/server/server.py` | Port 9880, caller context from `_meta` on every tool, dict results that say when they are empty, docstrings rewritten for this corpus. Adds `create_ticket`, `get_access_requests`, `get_approvals`, `request_approval`, `get_team`. |
| `mcp_server/writes.py` (`create_ticket`) | `lesson-08-mcp/homework/README.md` task 2; row shape from Lesson 10 `db.create_access_request` | Approval-gated and idempotent. Ids continue the seed scheme (`T007`), not `TCK-0001`. |
| `mcp_server/writes.py` (gate) | `lesson-13-ai-security-and-guardrails/SDD/shared/harness/enforcement.py` | `filing_grants` becomes `approval_controls`, tied to a row in `approvals` and scoped to employee, system, caller and workflow request. Idempotency keys added. `confirm()` becomes `record_approval_decision()`, still not an MCP tool, and only the named approver may decide. |
| `mcp_server/context.py` | none (new) | Reads caller context from MCP request `_meta`. |
| `mcp_server/opensearch.py`, `scripts/create_index.py` | `lesson-07-rag-advanced/code/01-opensearch-setup/{client,create_index}.py` | Index name from `OPENSEARCH_INDEX` through one guard that refuses `novaops-kb`. Mapping adds `doc_id`, `chunk_id`, `section`, `doc_type`, `citation`, `subject_employee_id`, `embedding_model` and a date `last_updated`. An existing index is kept unless `--recreate`. |
| `mcp_server/documents.py`, `scripts/ingest.py` | `lesson-07-rag-advanced/code/01-opensearch-setup/ingest.py`, `02-filtering/ingest.py` (`parse_frontmatter`, `chunk_document`) | All seven collections. Metadata from front matter or inline header lines. Chunks on `##` sections (the sliding window only inside long sections). Stable ids. Re-runnable: old chunks found by `doc_id` and deleted first, because Serverless rejects caller-set `_id`s. No subject tagging. |
| `mcp_server/retrieval.py` | `lesson-07-rag-advanced/code/02-filtering/retrieval.py` (`build_filter`), `03-reranking/{retrieval,reranker}.py` | Access clause from manager status (`manager_id`), as `must_not` inside the k-NN filter. Corpus scope per tool. 20 candidates reranked to 5; a reranker parse failure falls back to vector order with `reranked: false`. Lesson 10's FAISS `rag.py` is not copied. |
| `agent/mcp_tools.py` | `lesson-12-eval-loop-engineering/samples/02-coding-loop/agent-hardened/agent.py` (`load_tools`, `safe_tool_result`) | Sessions from `langchain-mcp-adapters`; our own `call_tool()` because the adapter does not forward `_meta`. One `tool:<name>` span per call. |
| `agent/run.py`, `agent/ask.py` | `lesson-11-observability-and-evals/code/02-instrument-agent/run.py`; resume loop from `lesson-09-agentic-frameworks/code/03-human-in-the-loop/graph.py` (`run`) | `run_turn()` with the final signature: root span per turn, `propagate_attributes(session_id, user_id, metadata)`, flush in `finally`. Opens an injected `AsyncSqliteSaver` and the MCP session, builds the graph, and resumes a paused thread with `Command(resume=...)` instead of asking a terminal. A resume keeps the checkpoint's `request_id` (a different one raises `RequestIdMismatch`) and opens its root span in the paused trace. Returns `TurnOutcome`. |
| `agent/graph.py` | `lesson-10-context-engineering/code/02-dynamic-tool-loadout/graph.py`, `03-history-distillation/graph.py`; `lesson-09-agentic-frameworks/code/03-human-in-the-loop/graph.py` (`human_review`); `lesson-11-observability-and-evals/code/02-instrument-agent/graph.py` (callback) | `plan`/`select`/`model` become `classify`/`scope`/`gather`. The checkpointer is a parameter. Our `tools` node calls `as_langchain_tools()` instead of `ToolNode`, caps calls at 6 and refuses tools not visible this turn. Approval path: `request_approval` (code calls the MCP tool) → `await_approval` (`interrupt()`) → `write_gate` (`release_write()`). Earlier turns reach the model as text, because Bedrock refuses `toolUse` blocks in a call that binds no tools. Distillation is task #4. |
| `agent/policy.py` | `lesson-10-context-engineering/code/02-dynamic-tool-loadout/stage02_policy.py` | Same eight intents. `TurnPlan` fields renamed (`intent`, `facts`, `active_constraints`, `needs_tools`, `action_confirmed`). Adds `SCOPE_OF`, loadouts for `get_team`, `get_access_requests`, `get_approvals`. `select_tools()` no longer adds `create_access_request` on `action_confirmed`. |
| `agent/structured.py` | `lesson-12-eval-loop-engineering/samples/02-coding-loop/agent-hardened/agent.py` (`parse_with_repair`) | Raises `ParseFailed` (named) instead of `StructuredOutputError`; replaces Lesson 10's silent `TurnPlan` fallback. |
| `agent/gate.py` | `lesson-10-context-engineering/code/02-dynamic-tool-loadout/stage02_policy.py` (`select_tools` write release) | The release reads the approval row over MCP (`get_approvals`) instead of trusting `action_confirmed`. Fails closed on a read error. |
| `agent/outcome.py` | `agent/run.py` from task #2 (`cited_labels`, `_status`) | `TurnOutcome`. Labels are checked against every tool result in the thread, so a follow-up can cite turn 1. Status adds `blocked` (rejected approval) and the shut-gate cases. |
| `evals/tool_map.py` | none (new) | Golden illustrative tool names to ours. Forbidden tools are matched by exact name only. |
| `scripts/check_langfuse.py` | `lesson-11-observability-and-evals/code/01-langfuse-setup/01-connect-langfuse.py` | Uses the shared config loader. |
| `scripts/reseed.py` | none (new) | Deletes and rebuilds the DB from the course `schema.sql` and `seed.sql`, then adds the side tables. |

Planned sources, from the workspace brief:

| Area | Start from |
| --- | --- |
| MCP server and the ten tools | `lesson-10-context-engineering/code/server/{server,db,rag}.py` |
| `create_ticket` | `lesson-08-mcp/homework/README.md` task 2, write shape from `db.create_access_request` |
| Approval-gated writes | `lesson-13-ai-security-and-guardrails/SDD/shared/harness/enforcement.py` |
| OpenSearch ingest and retrieval | `lesson-07-rag-advanced/code/01-opensearch-setup/`, `02-filtering/`, `03-reranking/` |
| Agent graph | `lesson-10-context-engineering/code/03-history-distillation/graph.py`, `stage03_policy.py`, `02-dynamic-tool-loadout/stage02_policy.py`, `00-agent-shared/agent.py` |
| Retry and structured-output hardening | `lesson-12-eval-loop-engineering/samples/02-coding-loop/agent-hardened/agent.py` |
| Interrupt and durable resume | `lesson-09-agentic-frameworks/code/03-human-in-the-loop/graph.py` |
| Maya specification | `lesson-10-context-engineering/homework/README.md` |
| Session replay and checks | `lesson-10-context-engineering/code/evals/` |
| Langfuse | `lesson-11-observability-and-evals/code/01-langfuse-setup/`, `02-instrument-agent/` |
| Scoring and judges | `lesson-11-observability-and-evals/code/03-score-the-run/`, `04-llm-judge/` |

## Decisions

Each decision names the lesson principle behind it. "Ours" marks a decision no lesson makes.

### Foundation

| Decision | Lesson principle |
| --- | --- |
| One `call_model()` / `get_model()` is the only Bedrock caller. Model id and region from env, version pinned, boto retries off, one `should_retry` that fails closed on unknown codes. | L5, L2, L3, L14 |
| Tools live behind an MCP server. The agent discovers them with `list_tools()`. Docstrings are the tool prompts. Bad arguments return `is_error` results. | L8 |
| Ingest parses front matter and stores `audience`, `corpus`, `source`, `last_updated` as filterable fields. Each file is chunked on its own. | L7, L6 |
| OpenSearch Serverless, collection `novaops-rag`, index `novaops-final-kb` from `OPENSEARCH_INDEX`. Index code refuses any other name and refuses `novaops-kb`. Retrieve wide, then listwise rerank. | L7 |
| The audience clause is a k-NN pre-filter. Non-managers get `must_not: {term: {audience: "manager"}}`. | L7, L13 |
| Retrieval runs inside the MCP server. `caller_id` and `user_group` are injected from graph config, not chosen by the model. | L8, L10, L13 |
| The database is a file with a reseed command. The clock is pinned to 2026-07-01. | Ours (spec §13) |
| `mcp<2`, FastMCP 1.x, `AsyncSqliteSaver`, Langfuse `get_client` / `propagate_attributes` / `CallbackHandler`, MCP port 9880. | L9, L11 |
| Caller context goes in the MCP request `_meta` (key `novaops`), not in a hidden parameter. `_meta` is protocol, so no input schema can carry it, and a call without it is refused. Business ids are named `access_request_id` and `record_id` so `request_id` only ever means the workflow request. | L8, L13; the mechanism is ours |
| `last_updated` comes from front matter, else `Effective date:` (policies), else `Date:` (memos), else `Last updated:` (IT KB). `doc_type` is `memo`, `policy` or the corpus name. `subject_employee_id` is set on employment files by exact full name only; two files match nobody and stay empty. | L7; the fallbacks are ours |
| Contracts are ingested (corpus `contracts`) but neither search tool reads them yet. `search_knowledge_base` covers IT KB, policies and memos; `search_hr_documents` covers handbook, employment, policies, memos and the filtered manager playbook. | Ours |
| The seat question (`W-I-02`) is answered by fields on `check_software_subscription`: `seat_assignments_recorded: false` and a note, next to the seat totals. One tool call gives both the count and the gap. | Ours (spec §13) |
| Approval scope lives in a side table, `approval_controls`: employee, system, requesting caller, workflow `request_id`, expiry (7 days from the pinned date), and the idempotency key that consumed it. A gated write needs an `approved` row of the matching type (`access_request` for `create_access_request`, `ticket` for `create_ticket`) with that scope. | L13 |
| Idempotency: a write's key is (action, employee, system or ticket topic, workflow `request_id`), stored in `idempotency_keys`. An open row for the same employee and system (`AR001`, `T001`) is also reused. A reuse writes nothing and does not consume the approval. | L13, L14, spec §6.6 |

### Agent

| Decision | Lesson principle |
| --- | --- |
| One graph: classify (`TurnPlan`) → scope → gather evidence → answer. Classify is the router. | L10, L9 |
| `LOADOUTS` keyed by intent. `None` exposes all read tools, `[]` none. Write tools never appear in it. | L10 |
| `release_write()` opens only on a persisted approval row for this action, caller and request. `action_confirmed` affects visibility only. | L10, L9, L13 |
| The approval-gated MCP write tools re-check the approval in one transaction. A graph guard is not authorization. | L13 |
| `interrupt()` plus an injected SQLite checkpointer. The resume is the same trace. | L9, L14 |
| Every write carries an idempotency key. A replay returns the existing row with `reused: true`. An approval is consumed once. | L13, L14 |
| Caller identity is a function input in graph config, never read from message text. | L13 |
| `TurnPlan.active_constraints` carries constraints across turns. Memory distillation past a threshold, with a safety check. History cut at turn boundaries. | L10 |
| At most one refined retrieval round. `rearm` reopens read tools only. A turn cap applies. | L10, L4, L13 |
| Typed returns by forced final tool call, validated locally, one repair, then `ParseFailed`. | L3, L4, L12, L14 |
| `status` is set by code from tool results. "Actions taken" lists only results with a row id. | L4, L8, L14 |
| Newer or more specific guidance wins over policy, with a deterministic check on the 7,500 threshold case. | L7 (metadata); the rule is ours |

### Maya

| Decision | Lesson principle |
| --- | --- |
| A caller is a manager when some employee's `manager_id` equals their id. Unknown status fails closed. | L7, L6, L13; the rule is ours (spec §12) |
| Chunks carry `[source#section]` labels and the answer cites them. A deterministic check rejects uncited claims. | L6, L11 |
| Maya's intents have no route to `release_write()`, and a test asserts it. | L10 |
| An explicit request to act produces a persisted `Handoff` and status `pending`. | Ours (spec §6.5) |

### Webex

| Decision | Lesson principle |
| --- | --- |
| Diagnose from the employee, subscription, tickets, access requests, approvals, then KB and policies. The seat overage is organizational. | L8, L7 |
| "Go ahead" persists an approval request, interrupts and ends `pending`. A separate test records a decision, resumes the thread and shows one write. | L13, spec §9 |
| `W-I-02` uses a team lookup by `manager_id` and states that seat assignments are not recorded. | Ours (spec §13) |
| New tools: `get_approvals`, `request_approval` (idempotent, creates a pending row, needs no prior approval; called by the graph's approval node, not the model), `record_approval_decision` (application code only). | Gap no lesson covers |

### Observability and evaluation

| Decision | Lesson principle |
| --- | --- |
| Langfuse callback on the compiled graph. Spans `classify`, `scope`, `tool:<name>`, `write_gate`, `answer`. `flush()` on exit. | L11, L14 |
| Deterministic checks first, judges second. Illustrative golden fields mapped to our tool names once, in `evals/tool_map.py`. | L11, L8 |
| Project checks under `evals/`: constraint carry, closed branch, citations, replay, durable pause, write gate. | L11, spec §12 |
| Develop on `GOLDEN-DATASETS.json`. `EVALUATION-INPUTS.yaml` runs once with a fresh seed per case. | L11, L12 |
| Root and nested `CLAUDE.md`, a hook protecting eval targets and fixtures. | L5, L12 |
