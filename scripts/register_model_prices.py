"""Register the agent model's price in Langfuse, so its generations show cost.

    python -m scripts.register_model_prices            # print what would be registered
    python -m scripts.register_model_prices --apply    # create the model definition

Langfuse prices a generation from its model id and token usage, but it has no price for
Nova 2 Lite, so without this the `model:<role>` generations show tokens and no cost.

The prices are the published on-demand prices for the `us.` cross-region inference
profile, per million tokens, as confirmed by the user on 2026-11-10: $0.30 input and
$2.50 output. They give an estimate only. The AWS bill is the source of truth for what a
run cost.
"""

import argparse
import re

from langfuse import get_client

from agent import config

PER_MILLION = {"input": 0.30, "output": 2.50}  # USD, published price, not the bill


def definition() -> dict:
    model_id = config.bedrock_model_id()
    return {"model_name": model_id, "match_pattern": f"(?i)^{re.escape(model_id)}$",
            "unit": "TOKENS", "input_price": PER_MILLION["input"] / 1_000_000,
            "output_price": PER_MILLION["output"] / 1_000_000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    wanted = definition()
    api = get_client().api
    existing = [m for m in api.models.list(limit=100).data
                if m.model_name == wanted["model_name"] and not m.is_langfuse_managed]
    if existing:
        print(f"[exists ] {wanted['model_name']} (id {existing[0].id}); delete it in Langfuse "
              "to change the price")
        return
    if not args.apply:
        print(f"[dry-run] would create {wanted}")
        return
    created = api.models.create(**wanted)
    print(f"[created] {created.model_name} input ${PER_MILLION['input']}/M, "
          f"output ${PER_MILLION['output']}/M (id {created.id})")


if __name__ == "__main__":
    main()
