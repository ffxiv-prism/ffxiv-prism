from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import portable_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply prediction-independent visual QA to a quality manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    reviews = {
        str(row["example_id"]): row
        for row in read_jsonl(args.reviews)
    }
    applied = 0
    reason_counts = Counter()
    for row in rows:
        review = reviews.get(str(row["example_id"]))
        if review is None:
            continue
        row["visual_review"] = review
        if review.get("decision") == "reject" and row.get("passed"):
            row["passed"] = False
            row.setdefault("rejection_reasons", []).append(
                "visual_review_reject"
            )
            applied += 1
            reason_counts[str(review.get("reason", "unspecified"))] += 1

    write_jsonl(args.output, rows)
    accepted = sum(bool(row.get("passed")) for row in rows)
    write_json(
        args.output.with_suffix(".summary.json"),
        {
            "source_manifest": portable_path(args.manifest),
            "visual_reviews": portable_path(args.reviews),
            "examples": len(rows),
            "accepted": accepted,
            "rejected": len(rows) - accepted,
            "coverage": accepted / len(rows) if rows else 0.0,
            "visual_rejections_applied": applied,
            "visual_rejection_reason_counts": dict(reason_counts),
            "output": portable_path(args.output),
        },
    )
    print(
        f"examples={len(rows)} accepted={accepted} applied={applied} "
        f"coverage={accepted / len(rows):.3f} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
