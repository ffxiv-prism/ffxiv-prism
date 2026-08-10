from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import portable_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute a matching report on independently reviewed visible "
            "image-slot cases while preserving the original unfiltered metrics."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--quality-manifest", type=Path)
    parser.add_argument(
        "--split",
        choices=("blind", "validation", "test"),
        default="blind",
    )
    parser.add_argument(
        "--branch",
        choices=("raw", "foreground"),
        default="raw",
        help="Evaluation branch for query-encoder reports.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    reviews = (
        {
            str(row["example_id"]): row
            for row in read_jsonl(args.reviews)
        }
        if args.reviews is not None
        else {}
    )
    quality_allowlist = (
        {
            str(row["example_id"])
            for row in read_jsonl(args.quality_manifest)
            if row.get("passed")
        }
        if args.quality_manifest is not None
        else None
    )
    if args.split == "blind":
        summary = report
        cases = report.get("cases", [])
    else:
        evaluation = report["evaluation"]
        if args.split not in evaluation:
            evaluation = evaluation[args.branch]
        summary = evaluation[args.split]
        cases = evaluation[f"{args.split}_cases"]
    cases = [
        case
        for case in cases
        if case.get("status") == "eligible"
    ]
    retained = []
    excluded = []
    for case in cases:
        example_id = str(case["example_id"])
        slot = str(case["slot"])
        review = reviews.get(example_id)
        unusable_by_review = (
            review is not None
            and (
                review.get("decision") == "reject"
                or review.get("slot_usability", {}).get(slot) == "unusable"
            )
        )
        rejected_by_quality = (
            quality_allowlist is not None
            and example_id not in quality_allowlist
        )
        (excluded if unusable_by_review or rejected_by_quality else retained).append(
            case
        )

    filtered = summarize_cases(retained, int(summary.get("top_k", 10)))
    source_example_ids = {str(case["example_id"]) for case in cases}
    retained_example_ids = {str(case["example_id"]) for case in retained}
    payload = {
        "source_report": portable_path(args.report),
        "split": args.split,
        "branch": args.branch if args.split != "blind" else None,
        "visibility_reviews": portable_path(args.reviews),
        "quality_manifest": portable_path(args.quality_manifest),
        "review_policy": (
            "Independent visual visibility only; model predictions were not "
            "used to choose retained cases."
        ),
        "unfiltered": {
            "eligible_case_count": summary.get("eligible_case_count"),
            "appearance_recall_at_1": summary.get("appearance_recall_at_1"),
            "appearance_recall_at_k": summary.get("appearance_recall_at_k"),
            "appearance_mrr": summary.get("appearance_mrr"),
        },
        "visible_only": filtered,
        "excluded_case_count": len(excluded),
        "retained_case_count": len(retained),
        "source_image_count": len(source_example_ids),
        "retained_image_count": len(retained_example_ids),
        "image_coverage": (
            len(retained_example_ids) / len(source_example_ids)
            if source_example_ids
            else 0.0
        ),
    }
    write_json(args.output, payload)
    print(
        f"visible={len(retained)} excluded={len(excluded)} "
        f"appearance@1={filtered['appearance_recall_at_1']:.3f} "
        f"appearance@{filtered['top_k']}={filtered['appearance_recall_at_k']:.3f}"
    )
    return 0


def summarize_cases(cases: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    by_slot = defaultdict(list)
    for case in cases:
        by_slot[str(case["slot"])].append(case)
    output = summarize_one(cases, top_k)
    output["by_slot"] = {
        slot: summarize_one(rows, top_k)
        for slot, rows in sorted(by_slot.items())
    }
    return output


def summarize_one(cases: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    count = len(cases)
    hits_at_1 = sum(bool(case.get("appearance_hit_at_1")) for case in cases)
    hits_at_k = sum(bool(case.get("appearance_hit_at_k")) for case in cases)
    reciprocal_rank = sum(
        1.0 / int(case["appearance_rank"])
        for case in cases
        if case.get("appearance_rank")
    )
    return {
        "eligible_case_count": count,
        "appearance_hits_at_1": hits_at_1,
        "appearance_hits_at_k": hits_at_k,
        "appearance_recall_at_1": hits_at_1 / count if count else 0.0,
        "appearance_recall_at_k": hits_at_k / count if count else 0.0,
        "appearance_mrr": reciprocal_rank / count if count else 0.0,
        "top_k": top_k,
    }


if __name__ == "__main__":
    raise SystemExit(main())
