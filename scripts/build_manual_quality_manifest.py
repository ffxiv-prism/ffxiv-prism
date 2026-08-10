from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a visual quality manifest from accepted example ids."
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--accepts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accepted-examples-output",
        type=Path,
        help="Optional image-level examples file containing only accepted rows.",
    )
    args = parser.parse_args()

    accepted = {
        line.strip()
        for line in args.accepts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = []
    seen = set()
    examples = read_jsonl(args.examples)
    for example in examples:
        example_id = str(example["example_id"])
        passed = example_id in accepted
        if passed:
            seen.add(example_id)
        rows.append(
            {
                "example_id": example_id,
                "glamour_id": example["glamour_id"],
                "source_image": example["local_path"],
                "passed": passed,
                "quality_score": 1.0 if passed else 0.0,
                "rejection_reasons": (
                    [] if passed else ["visual_not_front_fullbody_high_quality"]
                ),
                "review_source": "complete_visual_review",
            }
        )
    missing = accepted - seen
    if missing:
        raise ValueError(f"Accepted ids not present in examples: {sorted(missing)}")
    write_jsonl(args.output, rows)
    if args.accepted_examples_output is not None:
        write_jsonl(
            args.accepted_examples_output,
            [
                example
                for example in examples
                if str(example["example_id"]) in accepted
            ],
        )
    print(
        f"images={len(rows)} accepted={len(accepted)} "
        f"groups={len({str(row['glamour_id']) for row in rows if row['passed']})} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
