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
| Agent edits to eval targets, fixtures and the held-out inputs are refused, in Daedalus sessions too | `.claude/settings.json`, `.claude/hooks/guard_protected_paths.py`, `.claude/user-settings.snippet.json`, `tests/test_guard_hook.py` | done (task #1, fixed before task #5) |
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
| `manager_playbook` is unreachable for non-managers (pre-filter); a non-manager's HR search returns one fixed note whatever the query | `mcp_server/retrieval.py`, `tests/test_prefilter.py`, `tests/test_live_leak.py` | done (tasks #2, #4) |
| Another person's employment documents are filtered out before ranking; HR, the subject and the subject's manager can read them | `mcp_server/retrieval.py` (`employment_clause`), `mcp_server/db.py` (`caller_access`), `tests/test_prefilter.py`, `tests/test_live_leak.py` | done (task #4) |
| Contracts are searchable through `search_knowledge_base` | `mcp_server/retrieval.py` (`KB_CORPORA`), `tests/test_live_leak.py` | done (task #4) |
| Database rows carry `[db:<table>#<id>]` labels; every fact sentence on a tool turn is cited, with one repair call, then `needs_human` | `mcp_server/server.py` (`_rows`), `agent/outcome.py` (`uncited_sentences`), `agent/graph.py` (`cite`), `tests/test_maya.py` | done (task #4) |
| Maya returns a validated `OnboardingChecklist` on onboarding turns with checklist rows | `agent/schemas.py`, `agent/records.py`, `agent/graph.py` (`checklist`), `tests/test_maya.py` | done (task #4) |
| A confirmed access request for a preboarding subject is handed to Webex once, through a typed port, with no model tool call; a subject no lookup identifies ends `needs_human` | `agent/handoff.py`, `agent/ports.py`, `tests/test_maya.py` | done (task #4) |
| Long threads are distilled into `ConversationMemory`; a memory that drops a constraint is rejected | `agent/memory.py`, `agent/graph.py` (`distil`), `tests/test_maya.py` | done (task #4) |
| Every expected S2/S9 read tool is reachable from its intent's loadout; no loadout or rearm holds a write | `evals/tool_matrix.py`, `tests/test_maya.py` | done (task #4) |
| `S2-onboarding-maya` and `S9-maya-own-account` replay on one thread with binding and project checks | `evals/run_session.py`, `evals/session_checks.py` | done (task #4) |
| Webex reads the case from code (subscription, tickets, access requests, approvals), reuses `T001`, `AR001`, `AP001` and `AP002`, and pauses; replaying the Rachel scenario adds no ticket, access request or approval | `agent/graph.py` (`request_approval`), `agent/webex.py`, `tests/test_webex.py` | done (task #5) |
| A write is released only against a recorded approval, exactly once: the released write consumes AP001 even when it returns the existing AR001, a replay returns AR001 with `reused: true`, and pasted approval text opens nothing | `mcp_server/writes.py` (`_record_reuse`), `agent/gate.py`, `tests/test_writes.py`, `tests/test_webex.py` | done (task #5) |
| A process killed between the pause and the decision resumes the same thread after a restart, releases the write once and ends `blocked` by the seat limit | `tests/approval_process.py`, `tests/test_approval.py` | done (task #5) |
| Webex returns a validated `AccessDecision` (observed facts, reused records, actions taken, next steps, blocking reason, status) that cannot say access was granted | `agent/schemas.py`, `agent/records.py`, `agent/webex.py`, `agent/graph.py` (`decide`) | done (task #5) |
| A confirmed access request for someone the thread never looked up is looked up by code: one match routes, none or several end `needs_human` | `agent/handoff.py`, `agent/graph.py` (`lookup_subject`), `tests/test_maya.py` | done (task #5) |
| `S6`, `S7` and `S8` replay with binding checks plus the write-gate checks (turn 3 files nothing, turn 4 pauses on reused AP001 and AP002, no new row over the session) | `evals/run_session.py`, `evals/session_checks.py` | done (task #5) |
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
| `.claude/hooks/guard_protected_paths.py` | `lesson-12-eval-loop-engineering/code/02-coding-loop/.claude/hooks/block_test_edits.py` | Protects `evals/targets.py`, `tests/fixtures/`, `.claude/`, the dataset files and the reference repos instead of every test file. Also blocks reads of `EVALUATION-INPUTS.yaml` and the denied shell commands. Finds the checkout from the target path, so it works from a Daedalus session folder; `--workspace` limits it to this workspace when wired from user settings. |
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
| `mcp_server/retrieval.py` | `lesson-07-rag-advanced/code/02-filtering/retrieval.py` (`build_filter`), `03-reranking/{retrieval,reranker}.py` | Access clause from manager status (`manager_id`), as `must_not` inside the k-NN filter. Corpus scope per tool. 20 candidates reranked to 5; a reranker parse failure falls back to vector order with `reranked: false`. Lesson 10's FAISS `rag.py` is not copied. Task #4: the employment-subject clause, the fixed playbook `access_note`, an `access` summary per result, `contracts` in the KB, at most two chunks per document, and a `precedence_note`. |
| `agent/mcp_tools.py` | `lesson-12-eval-loop-engineering/samples/02-coding-loop/agent-hardened/agent.py` (`load_tools`, `safe_tool_result`) | Sessions from `langchain-mcp-adapters`; our own `call_tool()` because the adapter does not forward `_meta`. One `tool:<name>` span per call. |
| `agent/run.py`, `agent/ask.py` | `lesson-11-observability-and-evals/code/02-instrument-agent/run.py`; resume loop from `lesson-09-agentic-frameworks/code/03-human-in-the-loop/graph.py` (`run`) | `run_turn()` with the final signature: root span per turn, `propagate_attributes(session_id, user_id, metadata)`, flush in `finally`. Opens an injected `AsyncSqliteSaver` and the MCP session, builds the graph, and resumes a paused thread with `Command(resume=...)` instead of asking a terminal. A resume keeps the checkpoint's `request_id` (a different one raises `RequestIdMismatch`) and opens its root span in the paused trace. Returns `TurnOutcome`. |
| `agent/graph.py` | `lesson-10-context-engineering/code/02-dynamic-tool-loadout/graph.py`, `03-history-distillation/graph.py`; `lesson-09-agentic-frameworks/code/03-human-in-the-loop/graph.py` (`human_review`); `lesson-11-observability-and-evals/code/02-instrument-agent/graph.py` (callback) | `plan`/`select`/`model` become `classify`/`scope`/`gather`. The checkpointer is a parameter. Our `tools` node calls `as_langchain_tools()` instead of `ToolNode`, caps calls at 6 and refuses tools not visible this turn. Approval path: `request_approval` (code calls the MCP tool) → `await_approval` (`interrupt()`) → `write_gate` (`release_write()`). Earlier turns reach the model as text, because Bedrock refuses `toolUse` blocks in a call that binds no tools. Task #4 adds `distil` (Lesson 10 stage 3), `handoff`/`unverified`, `refine`, `cite` and `checklist`. |
| `agent/memory.py` | `lesson-10-context-engineering/code/03-history-distillation/stage03_policy.py` | `ConversationMemory`, thresholds and `memory_is_safe` unchanged. The token estimate is inlined. A distiller parse failure is repaired once, then the memory is rejected and the raw history kept. |
| `agent/webex.py` | `lesson-11-observability-and-evals/homework/README.md` milestone 2 (the Webex cases: seat-limit block, reuse of T001, approval not given yet, a refusal) | New module. Pure functions over tool results: which approvals to request, the status, the blocking reason, reused records, actions taken and the rendered answer. |
| `tests/approval_process.py` | `lesson-09-agentic-frameworks/homework/` task 3 (durable approval across a restart) | Three phases in separate processes: the Rachel scenario until the pause (then SIGKILL), the resume after `record_approval_decision`, and a replay of the same request. |
| `agent/schemas.py`, `agent/ports.py`, `agent/handoff.py` | `lesson-10-context-engineering/homework/README.md` milestones 3 and 4 (`CallerContext`, `EvidenceCitation`, `OnboardingChecklist`, `WebexHandoff`, `WebexPort` and fake) | Retrieval stays behind MCP (spec §6.1) instead of an in-graph `EvidenceRetriever`. The real port calls `request_approval` over MCP. Routing on the subject's record is ours. |
| `agent/records.py` | Lesson 3 tool forcing; `parse_with_repair` from `agent/structured.py` | `submit_checklist` as a forced tool; local Pydantic validation plus a check that every label came from a tool. |
| `evals/run_session.py`, `evals/session_checks.py` | `lesson-10-context-engineering/code/evals/{runner,checks,evalrules}.py` | Replays through `run_turn()` on one thread, reseeds once per session, request id `<session>-t<n>`. Only binding fields are enforced, plus project checks (turn-9 precedence, closed branch, handoff-only turn 10, no playbook for S9). |
| `evals/tool_matrix.py` | `lesson-10-context-engineering/homework/README.md` milestone 3 (matrix checker) | Checks every accepted intent of every S2/S9 turn, and that no loadout or rearm holds a write or `request_approval`. |
| `agent/policy.py` | `lesson-10-context-engineering/code/02-dynamic-tool-loadout/stage02_policy.py` | Same eight intents. `TurnPlan` fields renamed (`intent`, `facts`, `active_constraints`, `needs_tools`, `action_confirmed`), plus `subject_employee_id` and `system` (task #4). Adds `SCOPE_OF`, loadouts for `get_team`, `get_access_requests`, `get_approvals`. `select_tools()` no longer adds `create_access_request` on `action_confirmed`. Task #4: `subscription_review` gains `search_hr_documents`, `equipment_request` gains `list_onboarding_tasks` (found by the tool matrix); `checked_plan()` and `missing_checklist()`. |
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
| `search_knowledge_base` covers IT KB, policies, memos and contracts (task #4: the Webex contract is the evidence for the 42-of-40 blocker); `search_hr_documents` covers handbook, employment, policies, memos and the filtered manager playbook. | Ours |
| The seat question (`W-I-02`) is answered by fields on `check_software_subscription`: `seat_assignments_recorded: false` and a note, next to the seat totals. One tool call gives both the count and the gap. | Ours (spec §13) |
| Approval scope lives in a side table, `approval_controls`: employee, system, requesting caller, workflow `request_id`, expiry (7 days from the pinned date), and the idempotency key that consumed it. A gated write needs an `approved` row of the matching type (`access_request` for `create_access_request`, `ticket` for `create_ticket`) with that scope. | L13 |
| Idempotency: a write's key is (action, employee, system or ticket topic, workflow `request_id`), stored in `idempotency_keys`. An open row for the same employee and system (`AR001`, `T001`) is also reused. **Task #5:** returning the existing row is the release, so the first one stores the key, consumes the approval and audits `access_request_reused`. A replay with the same key returns the same row with `reused: true` and no error. Another request cannot use the consumed approval. | L13, L14, spec §6.6 |
| `request_approval` returns the row this workflow request already holds, whatever its status, and says so with `this_request`. A replay after the release finds the consumed AP001 again instead of opening a second approval; another request that finds AP001 held ends `needs_human`, and nothing is created. | L13, L14 |
| `get_employee` falls back to a typo of the whole full name when no name contains the query ('rrachel stein' in S8, 'rrachel steins' in S7), marked `match: similar`. Every name word pairs with its own query word, equal or one inserted or deleted letter apart (4 letters or more); a changed letter, a lone first name or an extra word matches nobody (`mcp_server/db.py` `similar_employees`, `tests/test_name_lookup.py`). | Ours |

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
| **Deviation from spec §12.** An `employment` chunk with a `subject_employee_id` is readable only by that subject, by UG_HR callers and by the subject's manager. The clause sits inside the k-NN `filter` like the audience clause. Employment chunks with no subject stay open. HR status comes from the caller's own employee row, not from `_meta`. Spec §12 leaves unlabelled collections open to everyone; a regular employee must not read another person's offer letter. | L7, L13, Lesson 10 homework milestone 2 |
| A non-manager's `search_hr_documents` result always carries the same `access_note` ("The manager playbook is not searchable for this caller."), whatever the query, so the note cannot reveal what the playbook covers. When the user asked about manager material and the answer left the note out, code adds it. | L13; ours |
| Chunks carry `[source#section]` labels, database rows `[db:<table>#<id>]`, and a whole result `[db:<table>]` (for "nothing in stock"). On a turn with tool results every fact sentence carries a label; one repair call on a miss, then `needs_human`. Facts the user stated are attributed ("as you said"), not labelled. Made-up labels such as `[constraint]` count as invented. | L6, L11; the per-sentence rule is ours |
| Maya's intents have no route to `release_write()` or the approval node, rearm included, and tests assert it. | L10 |
| A confirmed `access_request` is routed on the subject's record: the planner names the id, and code accepts it only if a `get_employee` row matches (task #5: when the thread has none, code looks the subject up first). `preboarding` (E001) goes to the Maya handoff, any other subject (E010) to the Webex approval node, no match to `needs_human`. S2 turn 10 and S6 turn 4 are the same request from the same caller (E004), so neither the intent nor the user group can tell them apart. | Ours (spec §6.5) |
| The handoff turn makes no model call. Code builds `WebexHandoff` (key `handoff:{thread}:{employee}:{system}`), calls `WebexPort` once and stores the result in graph state, so a replay of the turn makes no second call. `McpWebexPort` asks E006 (the Webex approver on AP001) through `request_approval`, which reuses an open row. The turn ends `pending`. | Lesson 10 homework milestone 4; L13, L14 |
| `OnboardingChecklist` comes from a forced `submit_checklist` call, validated locally (every item cited, blocked items give a reason, every label returned by a tool), one repair, then `ParseFailed`. | L3, L12 |
| Code corrects two planner claims it can disprove: `needs_tools=False` on a first turn (nothing to answer from) and on a request that asks for the current state ("right now", "still"). An equipment turn that never read the checklist gets one refined round with `list_onboarding_tasks`. | L10 (rearm, refined round); ours |
| The answer model is told who "I" is (caller id and group from graph config) and today's pinned date. Tools still authorize from `_meta`, so this grants nothing. | L13 (identity from the session) |
| Retrieval returns at most two chunks per document among the top five, and a `precedence_note` when a memo is newer than a policy in the same result. | L7; ours |

### Webex

| Decision | Lesson principle |
| --- | --- |
| Diagnose from the employee, subscription, tickets, access requests, approvals, then KB and policies. The seat overage is organizational. The model reads with the `access_request` loadout; the approval node then reads the subscription, tickets, access requests and approvals from code (reusing any identical read the model made), so the reuse decision never depends on what the model chose to look up. | L8, L7; ours |
| **"Go ahead" persists an approval request, interrupts and ends `pending`, with no write in that trace** (S6, S7, S8 turn 4, and the Rachel scenario). A separate test records the decision with `record_approval_decision`, resumes the same thread and shows the write released once. Reason: spec §9 releases a write only against a recorded approval, never against the model's reading of "go ahead"; a "go ahead" is the model-inferred `action_confirmed`, which only routes the turn. | L13, L9, spec §9 |
| The approval node asks for both approvals of AR001: `access_request` (AP001, E006) and `subscription_expansion` (AP002, E011). Both come back `reused: true`; no approval row is created. The gate opens on AP001 only. AP002 is requested because rubric `WF_WEBEX_001` asks for "durable approval for expansion or renewal" and AP002 is the row typed for the expansion; deciding it belongs to Renewal (task #8). | L13; rubric `WF_WEBEX_001` |
| When the gate opens, the next model call is forced to `create_access_request` (Converse `toolChoice: tool`), the only tool bound, with a note naming the employee and system. The released write returns AR001 (still `blocked` by the seat limit), so the resumed run ends `blocked` with the seat limit as `blocking_reason`, not `completed`. | L3 (tool forcing), L4 (code records the outcome) |
| `action_confirmed` also covers an employee asking for access for themselves ("I need Webex access"), so the Rachel scenario reaches the approval path in one turn. Only `access_request` reaches the approval node; a confirmed turn of another Webex intent has no approval type. | Ours (spec §9) |
| A confirmed access request names a subject the thread has not looked up: `lookup_subject` calls `get_employee` from code with the id or name the planner named (`TurnPlan.subject_name` is new). One match routes; none or several end `needs_human` with the names found. PR #4 ended every such turn `needs_human`. | L13 (identity from records, not text); ours |
| `AccessDecision`: observed facts and next steps come from a forced `submit_access_decision` call, validated locally (every fact cited, every label returned by a tool, and no text claiming access was granted, that no approval is needed, that it is only a password reset or that a new row was created), one repair, then `ParseFailed`. Code sets the status, blocking reason, reused records (T001, AR001) and actions taken (only write-path results with a row id), and writes the answer from the record. `access_granted` is the constant `false`. | L3, L4, L12; spec §6.5 |
| Calls made by code (the subject lookup, the case reads, `request_approval`) go into the thread as call and result pairs, so they are cited, traced as `tool:<name>` spans and listed in `tools_called`, but they do not count against the model's six calls per turn. | L11, L4 |
| `W-I-02` uses a team lookup by `manager_id` and states that seat assignments are not recorded. `subscription_review` also gets `list_onboarding_tasks`, so the answer can name the blocked provisioning task of a joiner on the team. | Ours (spec §13) |
| New tools: `get_approvals`, `request_approval` (idempotent, creates a pending row, needs no prior approval; called by the graph's approval node, not the model), `record_approval_decision` (application code only). | Gap no lesson covers |

### Observability and evaluation

| Decision | Lesson principle |
| --- | --- |
| Langfuse callback on the compiled graph. Spans `classify`, `scope`, `tool:<name>`, `write_gate`, `answer`. `flush()` on exit. | L11, L14 |
| Deterministic checks first, judges second. Illustrative golden fields mapped to our tool names once, in `evals/tool_map.py`. | L11, L8 |
| Project checks under `evals/`: constraint carry, closed branch, citations, replay, durable pause, write gate. | L11, spec §12 |
| Develop on `GOLDEN-DATASETS.json`. `EVALUATION-INPUTS.yaml` runs once with a fresh seed per case. | L11, L12 |
| Root and nested `CLAUDE.md`, a hook protecting eval targets and fixtures. | L5, L12 |
