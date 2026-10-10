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

## Writes

Every write tool, in one `BEGIN IMMEDIATE` transaction:

1. Checks the approval row: it exists, matches this action, target employee, system and
   request, is approved, is unused and is not expired.
2. Looks up the idempotency key. If a row already exists, return it with `reused: true` and
   write nothing.
3. Writes the row, marks the approval consumed, commits.

If any step fails, nothing is written. A replayed approval is consumed once.
`create_access_request` only proposes a row with status `pending_approval`.
`record_approval_decision` exists for application code and is never exposed to the model.

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
