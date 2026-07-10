from __future__ import annotations

import argparse
import json
from pathlib import Path

from ffxiv_prism.jsonl import write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select the broad or exact relation from a Huiji model graph."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--relation",
        choices=("same_model", "exact_model"),
        required=True,
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    field = (
        "same_model_equivalents"
        if args.relation == "same_model"
        else "exact_model_equivalents"
    )
    selected = dict(payload)
    selected["direct_equivalents"] = payload[field]
    selected["selected_relation"] = args.relation
    selected["default_relation"] = args.relation
    write_json(args.output, selected)
    print(
        f"relation={args.relation} "
        f"mapped={len(selected['direct_equivalents'])} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
