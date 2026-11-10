"""A local web chat over Maya and Webex (task #12, ungraded).

Thin on purpose: every chat message goes through `agent.run.run_turn()`, the same entry
point the tests and `agent.ask` use, and approval decisions go through
`mcp_server.writes.record_approval_decision()`. The page holds no workflow logic.
`tests/test_web.py` checks that this package imports nothing else from `agent` or
`mcp_server`.
"""
