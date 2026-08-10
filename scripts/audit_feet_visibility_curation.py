from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import read_jsonl, write_json  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _unique_by_id(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in output:
            raise ValueError(f"duplicate {label} example_id: {example_id}")
        output[example_id] = row
    return output


def audit_feet_curation(
    features: Sequence[Mapping[str, Any]],
    queue: Sequence[Mapping[str, Any]],
    manual_reviews: Sequence[Mapping[str, Any]],
    negative_sample: Sequence[Mapping[str, Any]],
    training_reviews: Sequence[Mapping[str, Any]],
    *,
    min_negative_sample: int = 100,
    min_usable_rate: float = 0.95,
    min_wilson_low: float = 0.90,
) -> dict[str, Any]:
    features_by_id = _unique_by_id(features, label="feature")
    queue_by_id = _unique_by_id(queue, label="queue")
    negative_by_id = _unique_by_id(negative_sample, label="negative sample")
    training_by_id = _unique_by_id(training_reviews, label="training review")

    manual_by_id: dict[str, Mapping[str, Any]] = {}
    conflicts: list[str] = []
    for review in manual_reviews:
        if str(review.get("slot") or "").upper() != "FEET":
            continue
        example_id = str(review["example_id"])
        previous = manual_by_id.get(example_id)
        if previous is not None and previous.get("usability") != review.get("usability"):
            conflicts.append(example_id)
            continue
        manual_by_id[example_id] = review

    expected_queue_ids = {
        example_id
        for example_id, row in features_by_id.items()
        if row.get("triage_for_review") is True
    }
    queue_ids = set(queue_by_id)
    queue_missing = sorted(expected_queue_ids - queue_ids)
    queue_extra = sorted(queue_ids - expected_queue_ids)
    queue_unreviewed = sorted(queue_ids - set(manual_by_id))

    negative_ids = set(negative_by_id)
    negative_wrong_side = sorted(
        example_id
        for example_id in negative_ids
        if example_id not in features_by_id
        or features_by_id[example_id].get("triage_for_review") is not False
    )
    negative_unreviewed = sorted(negative_ids - set(manual_by_id))
    negative_glamours = {
        str(row["glamour_id"]) for row in negative_by_id.values()
    }
    negative_usable = sum(
        str(manual_by_id[example_id].get("usability")) == "usable"
        for example_id in negative_ids
        if example_id in manual_by_id
    )
    negative_reviewed = len(negative_ids) - len(negative_unreviewed)
    usable_rate = (
        negative_usable / negative_reviewed if negative_reviewed else 0.0
    )
    wilson_low, wilson_high = wilson_interval(negative_usable, negative_reviewed)

    known_unusable_ids = {
        example_id
        for example_id, review in manual_by_id.items()
        if example_id in features_by_id
        and str(review.get("usability")) == "unusable"
    }
    excluded_ids = {
        example_id
        for example_id, review in training_by_id.items()
        if review.get("decision") == "reject"
        or str((review.get("slot_usability") or {}).get("FEET")) == "unusable"
    }
    unapplied_unusable = sorted(known_unusable_ids - excluded_ids)
    rule_positive_unusable = len(known_unusable_ids & expected_queue_ids)
    rule_negative_unusable = len(known_unusable_ids - expected_queue_ids)

    checks = {
        "manual_review_conflicts_absent": not conflicts,
        "triage_queue_exact": not queue_missing and not queue_extra,
        "triage_queue_fully_reviewed": not queue_unreviewed,
        "negative_sample_large_enough": len(negative_ids) >= min_negative_sample,
        "negative_sample_one_per_glamour": len(negative_glamours) == len(negative_ids),
        "negative_sample_rule_negative_only": not negative_wrong_side,
        "negative_sample_fully_reviewed": not negative_unreviewed,
        "negative_sample_usable_rate": usable_rate >= min_usable_rate,
        "negative_sample_wilson_low": wilson_low >= min_wilson_low,
        "known_unusable_exclusions_applied": not unapplied_unusable,
    }
    return {
        "version": 1,
        "slot": "FEET",
        "status": "ready" if all(checks.values()) else "do_not_train",
        "checks": checks,
        "eligible_images": len(features_by_id),
        "eligible_glamours": len(
            {str(row["glamour_id"]) for row in features_by_id.values()}
        ),
        "triage": {
            "queued_images": len(queue_ids),
            "queued_glamours": len(
                {str(row["glamour_id"]) for row in queue_by_id.values()}
            ),
            "missing_queue_ids": queue_missing,
            "extra_queue_ids": queue_extra,
            "unreviewed_queue_ids": queue_unreviewed,
            "known_unusable": rule_positive_unusable,
        },
        "rule_negative_audit": {
            "sample_images": len(negative_ids),
            "sample_glamours": len(negative_glamours),
            "reviewed_images": negative_reviewed,
            "usable_images": negative_usable,
            "usable_rate": usable_rate,
            "wilson_95_low": wilson_low,
            "wilson_95_high": wilson_high,
            "wrong_side_ids": negative_wrong_side,
            "unreviewed_ids": negative_unreviewed,
        },
        "curation": {
            "manual_reviewed_images": len(manual_by_id),
            "known_unusable_images": len(known_unusable_ids),
            "rule_positive_unusable": rule_positive_unusable,
            "rule_negative_unusable": rule_negative_unusable,
            "excluded_by_training_reviews": len(known_unusable_ids & excluded_ids),
            "unapplied_unusable_ids": unapplied_unusable,
            "curated_images": len(features_by_id) - len(known_unusable_ids),
        },
        "conflicting_review_ids": sorted(set(conflicts)),
        "thresholds": {
            "min_negative_sample": min_negative_sample,
            "min_usable_rate": min_usable_rate,
            "min_wilson_low": min_wilson_low,
        },
        "interpretation_limit": (
            "This validates the FEET curation policy, not recognition accuracy. "
            "Parser predictions are used only to form a fully reviewed risk "
            "queue and a separately audited rule-negative pool."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the remediated FEET visibility curation policy."
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--triage-queue", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, action="append", required=True)
    parser.add_argument("--negative-audit-sample", type=Path, required=True)
    parser.add_argument("--training-reviews", type=Path, required=True)
    parser.add_argument("--min-negative-sample", type=int, default=100)
    parser.add_argument("--min-usable-rate", type=float, default=0.95)
    parser.add_argument("--min-wilson-low", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_feet_curation(
        read_jsonl(args.features),
        read_jsonl(args.triage_queue),
        [row for path in args.reviews for row in read_jsonl(path)],
        read_jsonl(args.negative_audit_sample),
        read_jsonl(args.training_reviews),
        min_negative_sample=args.min_negative_sample,
        min_usable_rate=args.min_usable_rate,
        min_wilson_low=args.min_wilson_low,
    )
    report["inputs"] = {
        "features": portable_path(args.features),
        "triage_queue": portable_path(args.triage_queue),
        "reviews": [portable_path(path) for path in args.reviews],
        "negative_audit_sample": portable_path(args.negative_audit_sample),
        "training_reviews": portable_path(args.training_reviews),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
