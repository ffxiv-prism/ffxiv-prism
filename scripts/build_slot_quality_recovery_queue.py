from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.audit_glamour_training_readiness import (  # noqa: E402
    SLOTS,
    build_identity_roots,
    identity_for,
    label_piece_ids,
)
from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402


SLOT_IRRELEVANT_EDGE_REASONS = {
    "HEAD": frozenset({"feet_likely_cropped"}),
    "BODY": frozenset({"feet_likely_cropped", "head_likely_cropped"}),
    "HANDS": frozenset({"feet_likely_cropped", "head_likely_cropped"}),
    "LEGS": frozenset({"head_likely_cropped"}),
    "FEET": frozenset({"head_likely_cropped"}),
}
CONDITIONAL_REASONS = frozenset({"not_full_body"})
NEVER_RECOVER_REASONS = frozenset({"visual_review_reject"})
MANUAL_CONVERSION_BLOCKING_REASONS = {
    "HEAD": frozenset(
        {"head_likely_cropped", "person_too_few_pixels", "person_too_small_in_frame"}
    ),
    "BODY": frozenset({"person_too_few_pixels", "person_too_small_in_frame"}),
    "HANDS": frozenset({"person_too_few_pixels", "person_too_small_in_frame"}),
    "LEGS": frozenset({"feet_likely_cropped", "person_too_few_pixels", "person_too_small_in_frame"}),
    "FEET": frozenset({"feet_likely_cropped", "person_too_few_pixels", "person_too_small_in_frame"}),
}


def recoverable_for_slot(
    rejection_reasons: Sequence[str],
    slot: str,
    *,
    policy: str = "conservative_edge",
) -> bool:
    reasons = {str(reason) for reason in rejection_reasons}
    if not reasons or reasons & NEVER_RECOVER_REASONS:
        return False
    if policy == "manual_conversion":
        return not bool(reasons & MANUAL_CONVERSION_BLOCKING_REASONS[slot])
    if policy != "conservative_edge":
        raise ValueError(f"unknown recovery policy: {policy}")
    edge_reasons = reasons - CONDITIONAL_REASONS
    return bool(edge_reasons) and edge_reasons <= SLOT_IRRELEVANT_EDGE_REASONS[slot]


def current_identity_glamours(
    examples: Sequence[Mapping[str, Any]],
    passed_ids: set[str],
    roots_by_slot: Mapping[str, Mapping[int, int]],
) -> dict[str, dict[int, set[str]]]:
    output: dict[str, dict[int, set[str]]] = {
        slot: defaultdict(set) for slot in SLOTS
    }
    for example in examples:
        if str(example["example_id"]) not in passed_ids:
            continue
        for slot in SLOTS:
            if not label_piece_ids(example, slot):
                continue
            identity = identity_for(example, slot, roots_by_slot[slot])
            output[slot][identity].add(str(example["glamour_id"]))
    return output


def build_recovery_rows(
    examples: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    *,
    min_quality_score: float,
    policy: str = "conservative_edge",
) -> list[dict[str, Any]]:
    quality_by_id = {
        str(row["example_id"]): row for row in quality_rows
    }
    passed_ids = {
        example_id
        for example_id, row in quality_by_id.items()
        if row.get("passed")
    }
    roots_by_slot = {
        slot: build_identity_roots(examples, slot) for slot in SLOTS
    }
    current = current_identity_glamours(examples, passed_ids, roots_by_slot)
    rows = []
    for example in examples:
        example_id = str(example["example_id"])
        quality = quality_by_id.get(example_id)
        if quality is None or quality.get("passed"):
            continue
        quality_score = float(quality.get("quality_score") or 0.0)
        if quality_score < min_quality_score:
            continue
        reasons = [str(value) for value in quality.get("rejection_reasons") or []]
        glamour_id = str(example["glamour_id"])
        slots = []
        conversion_slots = []
        slot_details = {}
        for slot in SLOTS:
            if not label_piece_ids(example, slot) or not recoverable_for_slot(
                reasons,
                slot,
                policy=policy,
            ):
                continue
            identity = identity_for(example, slot, roots_by_slot[slot])
            glamour_ids = current[slot].get(identity, set())
            if glamour_id in glamour_ids:
                continue
            current_count = len(glamour_ids)
            if current_count == 0:
                continue
            if policy == "manual_conversion" and current_count != 1:
                continue
            slots.append(slot)
            if current_count == 1:
                conversion_slots.append(slot)
            slot_details[slot] = {
                "appearance_identity": identity,
                "current_independent_glamours": current_count,
                "item_name": str(
                    (
                        (example.get("core_slot_labels") or {}).get(slot) or {}
                    ).get("item_name")
                    or ""
                ),
            }
        if not slots:
            continue
        race = str(example.get("model_race") or "")
        priority_score = (
            10 * len(conversion_slots)
            + 3 * int("HEAD" in slots and race == "Roegadyn")
            + len(slots)
            + quality_score
        )
        rows.append(
            {
                "example_id": example_id,
                "glamour_id": glamour_id,
                "local_path": str(example["local_path"]),
                "model_race": race,
                "model_gender": str(example.get("model_gender") or ""),
                "quality_score": quality_score,
                "original_rejection_reasons": reasons,
                "recoverable_slots": slots,
                "single_to_multi_conversion_slots": conversion_slots,
                "slot_details": slot_details,
                "priority_score": priority_score,
                "review_source": "manual_slot_quality_recovery",
                "recovery_policy": policy,
                "slot_decisions": {slot: None for slot in slots},
                "global_decision": None,
                "issue": None,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["priority_score"]),
            -float(row["quality_score"]),
            str(row["example_id"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a manual-review queue for slot-safe recovery of images "
            "rejected only because the opposite end of the body was cropped."
        )
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("data/processed/glamour_examples_huiji_models_closed.jsonl"),
    )
    parser.add_argument(
        "--quality-manifest",
        type=Path,
        default=Path("data/processed/front_fullbody_quality_final_v46.jsonl"),
    )
    parser.add_argument("--min-quality-score", type=float, default=0.90)
    parser.add_argument(
        "--policy",
        choices=("conservative_edge", "manual_conversion"),
        default="conservative_edge",
    )
    parser.add_argument("--queue-size", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.min_quality_score <= 1.0:
        parser.error("--min-quality-score must be between zero and one")
    if args.queue_size < 1:
        parser.error("--queue-size must be positive")

    rows = build_recovery_rows(
        read_jsonl(args.examples),
        read_jsonl(args.quality_manifest),
        min_quality_score=args.min_quality_score,
        policy=args.policy,
    )
    queue = rows[: args.queue_size]
    write_jsonl(args.output, queue)
    summary = {
        "version": 1,
        "examples": portable_path(args.examples),
        "quality_manifest": portable_path(args.quality_manifest),
        "min_quality_score": args.min_quality_score,
        "recovery_policy": args.policy,
        "candidate_images": len(rows),
        "queued_images": len(queue),
        "queued_glamours": len({row["glamour_id"] for row in queue}),
        "single_to_multi_conversion_opportunities": {
            slot: len(
                {
                    (
                        row["slot_details"][slot]["appearance_identity"],
                        row["glamour_id"],
                    )
                    for row in queue
                    if slot in row["single_to_multi_conversion_slots"]
                }
            )
            for slot in SLOTS
        },
        "roegadyn_head_candidates": sum(
            row["model_race"] == "Roegadyn" and "HEAD" in row["recoverable_slots"]
            for row in queue
        ),
        "output": portable_path(args.output),
        "policy": (
            "Manual review only. The conservative policy requires every reason "
            "to concern the opposite image edge. The manual-conversion policy "
            "only queues existing single-glamour identities and blocks prior "
            "visual rejects, target-edge crops and tiny-person cases. No row is "
            "recovered automatically."
        ),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
