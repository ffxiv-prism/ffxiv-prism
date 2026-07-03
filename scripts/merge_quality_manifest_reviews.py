from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preserve reviewed quality decisions while merging a newer pose manifest."
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_manifest)
    candidate_rows = read_jsonl(args.candidate_manifest)
    review_rows = read_jsonl(args.reviews)
    base_by_id = {row["example_id"]: row for row in base_rows}
    reviews_by_id = {row["example_id"]: row for row in review_rows}

    merged: list[dict] = []
    new_examples = 0
    applied_reviews = 0
    for candidate in candidate_rows:
        example_id = candidate["example_id"]
        if example_id in base_by_id:
            merged.append(base_by_id[example_id])
            continue

        new_examples += 1
        row = candidate
        review = reviews_by_id.get(example_id)
        if review is not None:
            row = dict(candidate)
            row["visual_review"] = review
            if review["decision"] == "reject":
                row["passed"] = False
                reasons = list(row.get("rejection_reasons", []))
                if "visual_review_reject" not in reasons:
                    reasons.append("visual_review_reject")
                row["rejection_reasons"] = reasons
            elif review["decision"] == "accept":
                row["passed"] = True
                row["rejection_reasons"] = []
            else:
                raise ValueError(f"unsupported review decision: {review['decision']}")
            applied_reviews += 1
        merged.append(row)

    missing_reviews = sorted(set(reviews_by_id) - {row["example_id"] for row in candidate_rows})
    if missing_reviews:
        raise ValueError(f"reviews not found in candidate manifest: {missing_reviews}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    accepted = sum(bool(row.get("passed")) for row in merged)
    new_rows = [row for row in merged if row["example_id"] not in base_by_id]
    rejection_reasons = Counter(
        review["reason"]
        for review in review_rows
        if review.get("decision") == "reject"
    )
    summary = {
        "accepted": accepted,
        "coverage": accepted / len(merged) if merged else 0.0,
        "examples": len(merged),
        "new_accepted": sum(bool(row.get("passed")) for row in new_rows),
        "new_examples": new_examples,
        "new_visual_rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "new_visual_rejections_applied": sum(
            review.get("decision") == "reject" for review in review_rows
        ),
        "new_visual_reviews": args.reviews.as_posix(),
        "old_accepted": sum(bool(row.get("passed")) for row in base_rows),
        "old_examples": len(base_rows),
        "output": args.output.as_posix(),
        "source_manifest_new_pose": args.candidate_manifest.as_posix(),
        "source_manifest_old": args.base_manifest.as_posix(),
        "visual_reviews_applied": applied_reviews,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"examples={len(merged)} accepted={accepted} "
        f"new_examples={new_examples} new_accepted={summary['new_accepted']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
