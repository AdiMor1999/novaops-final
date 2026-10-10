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
| One function calls Bedrock, with bounded retries that fail closed | `agent/model.py` | planned (task #2) |
| Documents ingested with `audience`, `corpus`, `source`, `last_updated` | `scripts/ingest.py` | planned (task #2) |
| Database is a file and reseeds before each case | `scripts/reseed.py`, `mcp_server/db.py` | planned (task #2) |
| Tools sit behind an MCP server; the agent never imports them | `mcp_server/server.py` | planned (task #2) |
| Every trace carries `session_id`, `user_id`, `request_id` and `status` | `agent/run.py` | planned (task #2) |
| Classify is the router; writes never appear in a loadout | `agent/graph.py`, `agent/policy.py` | planned (task #3) |
| Durable checkpointer, no `InMemorySaver` anywhere | `agent/run.py`, `tests/` | planned (task #3) |
| `manager_playbook` is unreachable for non-managers (pre-filter) | `mcp_server/retrieval.py` | planned (task #4) |
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

Filled in by task #2. Configuration variables are listed in `.env.example`.

## Starting code

Every module started as a copy of lesson lab code and was changed here. Nothing imports from
the lesson folders. Rows are added as each module is copied.

| Our file | Copied from (`lesson-code/`) | What changed |
| --- | --- | --- |
| `.claude/hooks/guard_protected_paths.py` | `lesson-12-eval-loop-engineering/code/02-coding-loop/.claude/hooks/block_test_edits.py` | Protects `evals/targets.py`, `tests/fixtures/`, `.claude/`, the dataset files and the reference repos instead of every test file. Also blocks reads of `EVALUATION-INPUTS.yaml`. |
| `.claude/settings.json` | `lesson-12-eval-loop-engineering/code/03-eval-loop/.claude/settings.json` | Allow list for headless project work, deny rules for `.env`, the held-out inputs and protected paths. No activity log. |
| `CLAUDE.md` and nested `CLAUDE.md` files | `lesson-12-eval-loop-engineering/code/02-coding-loop/CLAUDE.md`, `03-eval-loop/CLAUDE.md`, `lesson-code/*-sdd/CLAUDE.md-example.html` | Rewritten for this project's stack, layout and hard rules. |

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

### Agent

| Decision | Lesson principle |
| --- | --- |
| One graph: classify (`TurnPlan`) → scope → gather evidence → answer. Classify is the router. | L10, L9 |
| `LOADOUTS` keyed by intent. `None` exposes all read tools, `[]` none. Write tools never appear in it. | L10 |
| `release_write()` opens only on a persisted approval row for this action, caller and request. `action_confirmed` affects visibility only. | L10, L9, L13 |
| The MCP write tools re-check the approval in one transaction. A graph guard is not authorization. | L13 |
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
| New tools: `get_approvals`, `request_approval` (behind the gate), `record_approval_decision` (application code only). | Gap no lesson covers |

### Observability and evaluation

| Decision | Lesson principle |
| --- | --- |
| Langfuse callback on the compiled graph. Spans `classify`, `scope`, `tool:<name>`, `write_gate`, `answer`. `flush()` on exit. | L11, L14 |
| Deterministic checks first, judges second. Illustrative golden fields mapped to our tool names in the scorer only. | L11, L8 |
| Project checks under `evals/`: constraint carry, closed branch, citations, replay, durable pause, write gate. | L11, spec §12 |
| Develop on `GOLDEN-DATASETS.json`. `EVALUATION-INPUTS.yaml` runs once with a fresh seed per case. | L11, L12 |
| Root and nested `CLAUDE.md`, a hook protecting eval targets and fixtures. | L5, L12 |
