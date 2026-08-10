from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ffxiv_prism.jsonl import write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote already-consumed confirmation groups into training after "
            "their immutable blind reports have been written."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/glamour_split_v2.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    assignments = payload["assignments"]
    if any(role == "blind_test" for role in assignments.values()):
        raise ValueError("Sealed blind groups remain; evaluate them before promotion")
    promoted = sorted(
        glamour_id
        for glamour_id, role in assignments.items()
        if role == "confirmation_test"
    )
    if not promoted:
        raise ValueError("No consumed confirmation groups to promote")
    for glamour_id in promoted:
        assignments[glamour_id] = "train"
    payload["counts"] = dict(sorted(Counter(assignments.values()).items()))
    payload.setdefault("confirmation_promotions", []).append(
        {
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "group_count": len(promoted),
            "glamour_ids": promoted,
            "policy": (
                "Blind metrics remain immutable in their reports; groups may "
                "enter later training only after the sealed evaluation."
            ),
        }
    )
    write_json(args.manifest, payload)
    print(
        f"promoted={len(promoted)} counts={payload['counts']} "
        f"manifest={args.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
