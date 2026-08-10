from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write image-level examples belonging to one frozen split role."
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("data/processed/glamour_examples.jsonl"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/processed/glamour_split_v2.json"),
    )
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--review-output",
        type=Path,
        help="Optional all-passing placeholder manifest for visual contact sheets.",
    )
    args = parser.parse_args()

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    assignments = {
        str(glamour_id): str(role)
        for glamour_id, role in split["assignments"].items()
    }
    rows = [
        row
        for row in read_jsonl(args.examples)
        if assignments.get(str(row["glamour_id"])) == args.role
    ]
    write_jsonl(args.output, rows)
    if args.review_output is not None:
        write_jsonl(
            args.review_output,
            [
                {
                    "example_id": row["example_id"],
                    "source_image": row["local_path"],
                    "passed": True,
                    "quality_score": 1.0,
                    "rejection_reasons": [],
                }
                for row in rows
            ],
        )
    print(
        f"role={args.role} groups={len({str(row['glamour_id']) for row in rows})} "
        f"images={len(rows)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
