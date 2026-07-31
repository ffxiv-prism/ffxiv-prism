from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402


GLOBAL_REJECT_ISSUES = {"non_game_image"}


def merge_manual_reviews(
    reviews: Sequence[Mapping[str, Any]],
    source_reviews: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decisions: dict[tuple[str, str], str] = {}
    sources: defaultdict[str, set[str]] = defaultdict(set)
    issues: defaultdict[str, set[str]] = defaultdict(set)
    duplicate_rows = 0
    source_counts: Counter[str] = Counter()
    source_decision_counts: Counter[str] = Counter()
    globally_rejected_ids: set[str] = set()
    for row in reviews:
        example_id = str(row["example_id"])
        slot = str(row["slot"]).upper()
        usability = str(row.get("usability") or "")
        if usability not in {"usable", "unusable"}:
            raise ValueError(
                f"invalid usability for {example_id}/{slot}: {usability!r}"
            )
        key = (example_id, slot)
        previous = decisions.get(key)
        if previous is not None:
            duplicate_rows += 1
            if previous != usability:
                raise ValueError(
                    f"conflicting reviews for {example_id}/{slot}: "
                    f"{previous} vs {usability}"
                )
        decisions[key] = usability
        review_source = str(row.get("review_source") or "unspecified")
        source_counts[review_source] += 1
        sources[example_id].add(review_source)
        issue = str(row.get("issue") or "none")
        if issue != "none":
            issues[example_id].add(issue)

    for row in source_reviews:
        example_id = str(row["example_id"])
        decision = str(row.get("decision") or "")
        if decision not in {"accept", "reject"}:
            raise ValueError(
                f"invalid source decision for {example_id}: {decision!r}"
            )
        source_decision_counts[decision] += 1
        if decision == "reject":
            globally_rejected_ids.add(example_id)
        review_source = str(row.get("review_source") or "unspecified")
        source_counts[review_source] += 1
        sources[example_id].add(review_source)
        issue = str(row.get("issue") or "none")
        if issue != "none":
            issues[example_id].add(issue)

    output = []
    reviewed_example_ids = {key[0] for key in decisions} | {
        str(row["example_id"]) for row in source_reviews
    }
    for example_id in sorted(reviewed_example_ids):
        slot_usability = {
            slot: usability
            for (candidate_id, slot), usability in sorted(decisions.items())
            if candidate_id == example_id
        }
        example_issues = sorted(issues[example_id])
        decision = (
            "reject"
            if example_id in globally_rejected_ids
            or GLOBAL_REJECT_ISSUES.intersection(example_issues)
            else "accept"
        )
        output.append(
            {
                "example_id": example_id,
                "decision": decision,
                "slot_usability": slot_usability,
                "issues": example_issues,
                "review_sources": sorted(sources[example_id]),
            }
        )

    unusable_by_slot = Counter(
        slot
        for (_, slot), usability in decisions.items()
        if usability == "unusable"
    )
    summary = {
        "input_rows": len(reviews),
        "source_input_rows": len(source_reviews),
        "unique_image_slot_reviews": len(decisions),
        "duplicate_rows": duplicate_rows,
        "reviewed_images": len(output),
        "globally_rejected_images": sum(
            row["decision"] == "reject" for row in output
        ),
        "unusable_image_slots": sum(unusable_by_slot.values()),
        "unusable_by_slot": dict(sorted(unusable_by_slot.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "source_decision_counts": dict(sorted(source_decision_counts.items())),
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge image-slot manual reviews into the review schema consumed "
            "by query-encoder training."
        )
    )
    parser.add_argument("--reviews", type=Path, action="append", required=True)
    parser.add_argument(
        "--source-reviews",
        type=Path,
        action="append",
        default=[],
        help="Optional whole-image source-domain accept/reject reviews.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for path in args.reviews for row in read_jsonl(path)]
    source_rows = [
        row for path in args.source_reviews for row in read_jsonl(path)
    ]
    output, summary = merge_manual_reviews(rows, source_rows)
    summary.update(
        {
            "version": 1,
            "review_files": [portable_path(path) for path in args.reviews],
            "source_review_files": [
                portable_path(path) for path in args.source_reviews
            ],
            "output": portable_path(args.output),
            "global_reject_issues": sorted(GLOBAL_REJECT_ISSUES),
        }
    )
    write_jsonl(args.output, output)
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
