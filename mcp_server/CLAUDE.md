# mcp_server/

The server boundary. Every read and write of NovaOps data goes through a tool defined here,
and this is where authorization, approvals and idempotency are enforced. Starting code:
`lesson-10-context-engineering/code/server/{server,db,rag}.py`, with the write pattern from
`lesson-13-ai-security-and-guardrails/SDD/shared/harness/enforcement.py`.

## Tools

- The docstring is the tool prompt. The model reads it to decide when to call the tool, so
  write it for the model: what the tool returns, when to use it, when not to. Keep it accurate
  when the code changes.
- The schema describes shape only. Check meaning inside the tool: the employee exists, the
  system id is known, the date parses, the caller may see this record.
- Raise, do not crash. Bad arguments and missing rows come back as an `is_error` result with a
  message the model can act on. An unexpected exception must not take the server down.
- Keep empty, truncated and error results distinguishable. "No rows" and "query failed" are
  different answers.
- `caller_id`, `user_group` and `request_id` arrive from the agent's tool wrapper. They are
  not in the schema the model sees, and the tool trusts only these injected values for identity.
- The mechanism is the MCP request `_meta` field, key `novaops`
  (`{"caller_id", "user_group", "request_id"}`, plus the Langfuse `trace_id` and
  `parent_observation_id`). `agent/mcp_tools.py` sets it on every `call_tool`;
  `mcp_server/context.py` reads it through FastMCP's `Context`. A call without it is
  refused. `_meta` is protocol, not a tool argument, so no input schema can carry it.
  `tests/test_boundary.py` checks that no schema names any of the three.
- Business record ids never use the name `request_id`, so it can only mean the
  workflow request: `access_request_id` (AR001) and `record_id` (whatever an approval is
  for). Every tool goes through `_serve()` in `server.py`.
- The server imports `call_model` and `embed_text` from `agent/model.py` for reranking
  and embeddings, so the one-Bedrock-function rule holds here too.

## Writes

There are three kinds of write, and each has its own rule. Every write runs in one
`BEGIN IMMEDIATE` transaction and carries an idempotency key. If any step fails, nothing is
written.

### Approval-gated writes

The tools in the agent's `WRITE_TOOLS`: `create_ticket` and `create_access_request`. Each one,
inside its transaction:

1. Checks the approval row: it exists, matches this action, target employee, system and
   request, is approved, is unused and is not expired.
2. Looks up the idempotency key. If a row already exists, return it with `reused: true` and
   write nothing.
3. Writes the row, marks the approval consumed, commits.

A replayed approval is consumed once. `create_access_request` records a request with status
`pending_approval`; it never provisions access.

### `request_approval`

Creates the approval row that a gated write later needs, so it cannot require an approval
itself. It also writes the row's scope to `approval_controls` (employee, system, caller,
workflow `request_id`, expiry), which is what the gated write matches against. A seed
approval such as `AP001` gets its scope the first time a workflow requests it. It is idempotent instead: the key is (request type, request id). If an open row exists
for that key, it returns that row with `reused: true` and writes nothing. It only ever creates
rows with status `pending`. It cannot approve, and nothing it writes opens a gated write.

### `record_approval_decision`

Sets an approval row to approved or rejected, with the actor id. Application code calls it at
resume time. It is never exposed to the model and never appears in any loadout.

## Data

- The database is a file at `NOVAOPS_DB_PATH`, never `:memory:`. `python -m scripts.reseed`
  restores it from the course `seed.sql`.
- Dates come from `NOVAOPS_TODAY` (2026-07-01), never from the system clock.
- Retrieval runs here. The audience clause sits inside the k-NN `filter` (pre-filter): a caller
  who manages nobody (no employee has `manager_id` equal to their id) gets
  `must_not: {term: {audience: "manager"}}`. Documents without `audience` stay readable. If
  manager status cannot be determined, exclude `manager` documents.
- Only the index named by `OPENSEARCH_INDEX`. Index-management code refuses any other name and
  refuses `novaops-kb` by name.
