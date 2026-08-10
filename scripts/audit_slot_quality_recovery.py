from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402
from scripts.audit_glamour_training_readiness import (  # noqa: E402
    ENSEMBLE_INDEX_DIRS,
    INDEX_DIRS,
    SLOTS,
    build_identity_roots,
    identity_for,
    label_piece_ids,
    load_index_piece_ids,
    load_sqlite_piece_ids,
)


DEFAULT_EXAMPLES = Path(
    "data/processed/glamour_examples_huiji_models_closed.jsonl"
)
DEFAULT_QUALITY = Path("data/processed/front_fullbody_quality_final_v46.jsonl")
DEFAULT_QUEUE = Path(
    "data/outputs/slot-quality-manual-conversion-compact-20260801/"
    "review-set.jsonl"
)
DEFAULT_REVIEWS = Path(
    "data/processed/slot_quality_recovery_reviews_20260801.jsonl"
)
DEFAULT_VISIBILITY_REVIEWS = Path(
    "data/processed/training_visibility_reviews_20260801.jsonl"
)
DEFAULT_ELIGIBILITY = Path(
    "data/processed/slot_quality_eligibility_20260801.jsonl"
)
DEFAULT_SUMMARY = Path(
    "data/outputs/slot-quality-recovery-audit-20260801/summary.json"
)
DEFAULT_DB = Path("data/processed/eorzea.sqlite")
VALID_GLOBAL_DECISIONS = frozenset({"accept", "reject"})
VALID_SLOT_DECISIONS = frozenset({"usable", "unusable"})


def duplicate_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    counts = Counter(str(row.get("example_id") or "") for row in rows)
    return sorted(key for key, count in counts.items() if not key or count > 1)


def validate_reviews(
    queue_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    queue_by_id = {str(row.get("example_id")): row for row in queue_rows}
    review_by_id = {str(row.get("example_id")): row for row in review_rows}
    errors: dict[str, list[str]] = {
        "duplicate_queue_ids": duplicate_ids(queue_rows),
        "duplicate_review_ids": duplicate_ids(review_rows),
        "missing_review_ids": sorted(set(queue_by_id) - set(review_by_id)),
        "orphan_review_ids": sorted(set(review_by_id) - set(queue_by_id)),
        "invalid_global_decision_ids": [],
        "invalid_slot_key_ids": [],
        "invalid_slot_decision_ids": [],
        "rejected_with_usable_slot_ids": [],
    }
    accepted_slots: dict[str, set[str]] = {}
    for example_id in sorted(set(queue_by_id) & set(review_by_id)):
        queue = queue_by_id[example_id]
        review = review_by_id[example_id]
        global_decision = str(review.get("global_decision") or "")
        if global_decision not in VALID_GLOBAL_DECISIONS:
            errors["invalid_global_decision_ids"].append(example_id)
        expected_slots = set(queue.get("recoverable_slots") or [])
        decisions = review.get("slot_decisions") or {}
        if set(decisions) != expected_slots:
            errors["invalid_slot_key_ids"].append(example_id)
            continue
        if any(
            str(decisions[slot]) not in VALID_SLOT_DECISIONS
            for slot in expected_slots
        ):
            errors["invalid_slot_decision_ids"].append(example_id)
            continue
        usable = {
            slot for slot in expected_slots if decisions[slot] == "usable"
        }
        if global_decision == "reject" and usable:
            errors["rejected_with_usable_slot_ids"].append(example_id)
        if global_decision == "accept" and usable:
            accepted_slots[example_id] = usable
    return errors, accepted_slots


def build_eligibility_rows(
    examples: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    recovered_slots: Mapping[str, set[str]],
    visibility_reviews: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    quality_by_id = {
        str(row.get("example_id")): row for row in quality_rows
    }
    rows = []
    for example in examples:
        example_id = str(example.get("example_id"))
        quality_passed = bool(
            (quality_by_id.get(example_id) or {}).get("passed")
        )
        source = "global_quality_pass"
        if quality_passed:
            slots = [slot for slot in SLOTS if label_piece_ids(example, slot)]
        else:
            source = "manual_slot_recovery"
            slots = [
                slot
                for slot in SLOTS
                if slot in recovered_slots.get(example_id, set())
                and label_piece_ids(example, slot)
            ]
        review = (visibility_reviews or {}).get(example_id)
        if review is not None:
            if review.get("decision") == "reject":
                slots = []
            else:
                usability = review.get("slot_usability") or {}
                slots = [
                    slot
                    for slot in slots
                    if usability.get(slot, "usable") != "unusable"
                ]
        if not slots:
            continue
        rows.append(
            {
                "example_id": example_id,
                "glamour_id": str(example.get("glamour_id")),
                "eligible_slots": slots,
                "quality_source": source,
            }
        )
    return rows


def summarize_slot_coverage(
    examples: Sequence[Mapping[str, Any]],
    eligibility_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    examples_by_id = {
        str(example.get("example_id")): example for example in examples
    }
    eligible_by_slot = {
        slot: [
            examples_by_id[str(row["example_id"])]
            for row in eligibility_rows
            if slot in row.get("eligible_slots", [])
        ]
        for slot in SLOTS
    }
    output = {}
    for slot in SLOTS:
        roots = build_identity_roots(examples, slot)
        glamour_ids_by_identity: defaultdict[int, set[str]] = defaultdict(set)
        races = Counter()
        for example in eligible_by_slot[slot]:
            identity = identity_for(example, slot, roots)
            if identity is None:
                continue
            glamour_ids_by_identity[identity].add(str(example.get("glamour_id")))
            races[str(example.get("model_race") or "unknown")] += 1
        identity_count = len(glamour_ids_by_identity)
        multi_count = sum(
            len(glamour_ids) >= 2
            for glamour_ids in glamour_ids_by_identity.values()
        )
        output[slot] = {
            "eligible_images": len(eligible_by_slot[slot]),
            "eligible_glamours": len(
                {str(row.get("glamour_id")) for row in eligible_by_slot[slot]}
            ),
            "appearance_identity_count": identity_count,
            "multi_glamour_identity_count": multi_count,
            "multi_glamour_identity_rate": (
                multi_count / identity_count if identity_count else 0.0
            ),
            "race_counts": dict(sorted(races.items())),
        }
    return output


def recovered_candidate_coverage(
    examples: Sequence[Mapping[str, Any]],
    recovered_slots: Mapping[str, set[str]],
    sqlite_piece_sets: Mapping[str, set[int]],
    index_piece_sets: Mapping[str, Sequence[set[int]]],
) -> dict[str, Any]:
    examples_by_id = {
        str(example.get("example_id")): example for example in examples
    }
    checked = 0
    uncovered: list[dict[str, Any]] = []
    for example_id, slots in sorted(recovered_slots.items()):
        example = examples_by_id.get(example_id)
        for slot in sorted(slots):
            checked += 1
            piece_ids = set(label_piece_ids(example or {}, slot))
            missing_sources = []
            if not piece_ids.intersection(sqlite_piece_sets[slot]):
                missing_sources.append("sqlite")
            for index, candidate_ids in enumerate(index_piece_sets[slot]):
                if not piece_ids.intersection(candidate_ids):
                    missing_sources.append(f"index_{index}")
            if missing_sources:
                uncovered.append(
                    {
                        "example_id": example_id,
                        "slot": slot,
                        "piece_ids": sorted(piece_ids),
                        "missing_sources": missing_sources,
                    }
                )
    return {
        "checked_image_slot_pairs": checked,
        "covered_image_slot_pairs": checked - len(uncovered),
        "coverage_rate": (checked - len(uncovered)) / checked if checked else 0.0,
        "uncovered": uncovered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate manual per-slot recovery decisions and build an auditable "
            "slot eligibility manifest before fold regeneration."
        )
    )
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--quality-manifest", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument(
        "--visibility-reviews",
        type=Path,
        default=DEFAULT_VISIBILITY_REVIEWS,
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--eligibility-output", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    examples = read_jsonl(args.examples)
    quality_rows = read_jsonl(args.quality_manifest)
    queue_rows = read_jsonl(args.review_queue)
    review_rows = read_jsonl(args.reviews)
    visibility_review_rows = read_jsonl(args.visibility_reviews)
    visibility_review_counts = Counter(
        str(row.get("example_id") or "") for row in visibility_review_rows
    )
    duplicate_visibility_review_ids = sorted(
        key
        for key, count in visibility_review_counts.items()
        if not key or count > 1
    )
    visibility_reviews = {
        str(row["example_id"]): row for row in visibility_review_rows
    }
    errors, recovered_slots = validate_reviews(queue_rows, review_rows)
    error_count = sum(len(values) for values in errors.values())

    eligibility = build_eligibility_rows(
        examples,
        quality_rows,
        recovered_slots if not error_count else {},
        visibility_reviews,
    )
    write_jsonl(args.eligibility_output, eligibility)
    baseline = build_eligibility_rows(
        examples,
        quality_rows,
        {},
        visibility_reviews,
    )
    baseline_summary = summarize_slot_coverage(examples, baseline)
    post_summary = summarize_slot_coverage(examples, eligibility)

    sqlite_piece_sets = load_sqlite_piece_ids(args.db)
    index_piece_sets = {
        slot: [
            load_index_piece_ids(INDEX_DIRS[slot], slot),
            *(
                load_index_piece_ids(path, slot)
                for path in ENSEMBLE_INDEX_DIRS[slot]
            ),
        ]
        for slot in SLOTS
    }
    candidate_coverage = recovered_candidate_coverage(
        examples,
        recovered_slots,
        sqlite_piece_sets,
        index_piece_sets,
    )
    queue_by_id = {str(row["example_id"]): row for row in queue_rows}
    accepted_conversions = {
        slot: len(
            {
                int(queue_by_id[example_id]["slot_details"][slot]["appearance_identity"])
                for example_id, slots in recovered_slots.items()
                if slot in slots
                and slot
                in queue_by_id[example_id].get(
                    "single_to_multi_conversion_slots", []
                )
            }
        )
        for slot in SLOTS
    }
    coverage_ready = all(
        post_summary[slot]["multi_glamour_identity_rate"] >= 0.5
        for slot in SLOTS
    )
    head_roegadyn_ready = (
        post_summary["HEAD"]["race_counts"].get("Roegadyn", 0) >= 30
    )
    ready = (
        not error_count
        and not duplicate_visibility_review_ids
        and candidate_coverage["coverage_rate"] == 1.0
        and coverage_ready
        and head_roegadyn_ready
    )
    payload = {
        "version": 1,
        "status": "ready_for_fold_rebuild" if ready else "do_not_train",
        "training_started": False,
        "review_integrity": {"error_count": error_count, **errors},
        "visibility_review_integrity": {
            "duplicate_or_empty_ids": duplicate_visibility_review_ids,
        },
        "reviewed_images": len(review_rows),
        "accepted_recovery_images": len(recovered_slots),
        "accepted_image_slot_pairs": sum(map(len, recovered_slots.values())),
        "accepted_conversion_identities": accepted_conversions,
        "candidate_library_coverage": candidate_coverage,
        "baseline": baseline_summary,
        "post_recovery": post_summary,
        "gates": {
            "all_slot_multi_glamour_rate_at_least_50pct": coverage_ready,
            "head_roegadyn_images_at_least_30": head_roegadyn_ready,
            "recovered_items_in_all_active_candidate_sources": (
                candidate_coverage["coverage_rate"] == 1.0
            ),
            "folds_rebuilt": False,
            "strict_metrics_refreshed": False,
        },
        "sources": {
            "examples": portable_path(args.examples),
            "quality_manifest": portable_path(args.quality_manifest),
            "review_queue": portable_path(args.review_queue),
            "reviews": portable_path(args.reviews),
            "visibility_reviews": portable_path(args.visibility_reviews),
            "eligibility_output": portable_path(args.eligibility_output),
        },
        "interpretation_limit": (
            "This audit authorizes strict fold regeneration only. It does not "
            "authorize training until rebuilt folds pass leakage and volume checks."
        ),
    }
    write_json(args.summary, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
