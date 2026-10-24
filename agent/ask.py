"""Ask one question from the command line. Calls run_turn(); there is no other path.

    python -m agent.ask "What's the NovaOps policy on using a personal laptop for work?" \
        --caller E010 --group UG_REGULAR --thread t1
"""

import argparse
import asyncio
import json

from agent.run import run_turn


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the NovaOps agent one question.")
    parser.add_argument("message")
    parser.add_argument("--caller", required=True, help="caller employee id, e.g. E010")
    parser.add_argument("--group", required=True, help="caller user group, e.g. UG_REGULAR")
    parser.add_argument("--thread", required=True, help="thread id (Langfuse session id)")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--json", action="store_true", help="print the whole result as JSON")
    args = parser.parse_args()

    result = asyncio.run(run_turn(args.message, caller_id=args.caller, user_group=args.group,
                                  thread_id=args.thread, request_id=args.request_id))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(result["answer"])
    print(f"\nstatus: {result['status']}   request_id: {result['request_id']}")
    print(f"citations: {', '.join(result['citations']) or '(none)'}")
    if result.get("trace_url"):
        print(f"trace: {result['trace_url']}")


if __name__ == "__main__":
    main()
