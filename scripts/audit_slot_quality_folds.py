from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import read_jsonl, write_json  # noqa: E402
from ffxiv_prism.paths import portable_path, resolve_project_path  # noqa: E402
from scripts.audit_glamour_training_readiness import (  # noqa: E402
    SLOTS,
    Thresholds,
    audit_folds,
    build_identity_roots,
    find_near_duplicate_pairs,
    inspect_passed_images,
)


DEFAULT_EXAMPLES = Path(
    "data/processed/glamour_examples_huiji_models_closed.jsonl"
)
DEFAULT_ELIGIBILITY = Path(
    "data/processed/slot_quality_eligibility_20260801.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/outputs/slot-quality-fold-audit-20260801/summary.json"
)
DEFAULT_REGIONS = Path("data/debug/outfit-regions/overlap-final-201")
DEFAULT_FOLD_DIRS = {
    slot: Path(
        f"data/interim/appearance-disjoint-cv-v55-slot-recovery-{slot.lower()}"
    )
    for slot in SLOTS
}
TRAINING_VISIBLE_STATES = frozenset({"visible", "partially_visible", "occluded"})


def parse_fold_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("fold dir must be SLOT=PATH")
    slot, path = value.split("=", 1)
    slot = slot.upper()
    if slot not in SLOTS:
        raise argparse.ArgumentTypeError(f"unsupported slot: {slot}")
    if not path:
        raise argparse.ArgumentTypeError("fold path must not be empty")
    return slot, Path(path)


def validate_eligibility(
    examples: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    example_ids = {str(row.get("example_id")) for row in examples}
    counts = Counter(str(row.get("example_id") or "") for row in rows)
    return {
        "duplicate_or_empty_ids": sorted(
            key for key, count in counts.items() if not key or count > 1
        ),
        "orphan_ids": sorted(set(counts) - example_ids),
        "invalid_slot_ids": sorted(
            str(row.get("example_id"))
            for row in rows
            if not row.get("eligible_slots")
            or any(slot not in SLOTS for slot in row.get("eligible_slots", []))
        ),
    }


def fold_gate(audit: Mapping[str, Any]) -> bool:
    return bool(
        audit.get("fold_count") == 3
        and not audit.get("identity_leak_count")
        and not audit.get("exact_content_leak_count")
        and not audit.get("near_content_leak_count")
        and not audit.get("assignment_rotation_issue_count")
        and all(
            not row.get("missing_assignment_count")
            and row.get("train_quantity_ok")
            and row.get("validation_quantity_ok")
            for row in audit.get("folds", [])
        )
    )


def collect_crop_record_presence(
    examples: Sequence[Mapping[str, Any]],
    regions_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        source_path = Path(str(example["local_path"]))
        stem = f"{source_path.parent.name}-{source_path.stem}"
        manifest_path = regions_dir / stem / f"{stem}-regions.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_crop_names = set()
        for path_value in manifest.get("crop_paths", []):
            crop_path = resolve_project_path(path_value)
            if crop_path is None or not crop_path.is_file():
                continue
            prefix = f"{stem}-"
            crop_stem = crop_path.stem
            crop_name = (
                crop_stem[len(prefix) :]
                if crop_stem.startswith(prefix)
                else crop_stem.rsplit("-", 1)[-1]
            )
            existing_crop_names.add(crop_name)
        rows = []
        for region in manifest.get("regions", []):
            if str(region.get("name")) not in existing_crop_names:
                continue
            box = region.get("box") or ()
            width = (
                max(0, int(box[2]) - int(box[0])) if len(box) == 4 else 0
            )
            height = (
                max(0, int(box[3]) - int(box[1])) if len(box) == 4 else 0
            )
            rows.append(
                {
                    "slot": str(region.get("slot")),
                    "name": str(region.get("name")),
                    "role": str(region.get("role")),
                    "visibility": str(region.get("visibility")),
                    "width": width,
                    "height": height,
                }
            )
        if rows:
            output[str(example["example_id"])] = rows
    return output


def audit_crop_coverage(
    eligibility_rows: Sequence[Mapping[str, Any]],
    crop_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    missing = []
    checked = 0
    for row in eligibility_rows:
        example_id = str(row["example_id"])
        records = crop_records.get(example_id, [])
        for slot in row.get("eligible_slots", []):
            checked += 1
            if not any(
                record.get("slot") == slot
                and str(record.get("visibility")) in TRAINING_VISIBLE_STATES
                for record in records
            ):
                missing.append({"example_id": example_id, "slot": slot})
    return {
        "checked_image_slot_pairs": checked,
        "covered_image_slot_pairs": checked - len(missing),
        "coverage_rate": (checked - len(missing)) / checked if checked else 0.0,
        "missing_count": len(missing),
        "missing": missing[:100],
    }


def summarize_numeric(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": ordered[-1],
    }


def audit_crop_resolution(
    eligibility_rows: Sequence[Mapping[str, Any]],
    crop_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_slot: dict[str, Any] = {}
    for slot in SLOTS:
        short_sides = []
        pixel_counts = []
        missing_primary = []
        eligible_count = 0
        for row in eligibility_rows:
            if slot not in row.get("eligible_slots", []):
                continue
            eligible_count += 1
            example_id = str(row["example_id"])
            primary = [
                record
                for record in crop_records.get(example_id, [])
                if record.get("slot") == slot
                and record.get("role") == "primary"
                and str(record.get("visibility")) in TRAINING_VISIBLE_STATES
                and int(record.get("width") or 0) > 0
                and int(record.get("height") or 0) > 0
            ]
            if not primary:
                missing_primary.append(example_id)
                continue
            short_sides.append(
                sum(
                    min(int(record["width"]), int(record["height"]))
                    for record in primary
                )
                / len(primary)
            )
            pixel_counts.append(
                sum(
                    int(record["width"]) * int(record["height"])
                    for record in primary
                )
                / len(primary)
            )
        by_slot[slot] = {
            "eligible_image_count": eligible_count,
            "measured_image_count": len(short_sides),
            "missing_primary_count": len(missing_primary),
            "missing_primary": missing_primary[:100],
            "primary_short_side_pixels": summarize_numeric(short_sides),
            "primary_crop_pixels": summarize_numeric(pixel_counts),
        }
    return {
        "status": "descriptive_only",
        "slots": by_slot,
        "interpretation_limit": (
            "Native crop dimensions do not establish how many pixels belong "
            "to the labelled equipment. A visible crop can still be dominated "
            "by background, body garments, weapons, or occlusion."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit regenerated slot-specific folds against the final per-slot "
            "quality eligibility manifest before training."
        )
    )
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument(
        "--slot-eligibility-manifest",
        type=Path,
        default=DEFAULT_ELIGIBILITY,
    )
    parser.add_argument("--fold-dir", type=parse_fold_dir, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--min-train-images", type=int, default=600)
    parser.add_argument("--min-validation-images", type=int, default=250)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    args = parser.parse_args()

    fold_dirs = dict(DEFAULT_FOLD_DIRS)
    fold_dirs.update(dict(args.fold_dir or []))
    examples = read_jsonl(args.examples)
    eligibility = read_jsonl(args.slot_eligibility_manifest)
    errors = validate_eligibility(examples, eligibility)
    error_count = sum(len(values) for values in errors.values())
    eligible_by_id = {
        str(row["example_id"]): set(row.get("eligible_slots", []))
        for row in eligibility
    }
    union_examples = [
        row for row in examples if str(row.get("example_id")) in eligible_by_id
    ]
    image_rows, image_audit = inspect_passed_images(union_examples)
    near_pairs = find_near_duplicate_pairs(
        union_examples,
        image_rows,
        args.near_duplicate_hamming,
    )
    crop_records = collect_crop_record_presence(union_examples, args.regions_dir)
    crop_coverage = audit_crop_coverage(eligibility, crop_records)
    crop_resolution = audit_crop_resolution(eligibility, crop_records)
    thresholds = Thresholds(
        min_train_images_per_fold=args.min_train_images,
        min_validation_images_per_fold=args.min_validation_images,
        near_duplicate_hamming=args.near_duplicate_hamming,
    )
    folds: dict[str, Any] = {}
    for slot in SLOTS:
        slot_examples = [
            row
            for row in union_examples
            if slot in eligible_by_id[str(row.get("example_id"))]
        ]
        folds[slot] = audit_folds(
            slot,
            slot_examples,
            build_identity_roots(examples, slot),
            image_rows,
            near_pairs,
            thresholds,
            registered_eligible=None,
            fold_dir=fold_dirs[slot],
        )
    image_ready = not any(
        image_audit[key]
        for key in (
            "missing_count",
            "unreadable_count",
            "dimension_mismatch_count",
            "exact_duplicate_group_count",
        )
    )
    slot_gates = {slot: fold_gate(audit) for slot, audit in folds.items()}
    ready = (
        not error_count
        and image_ready
        and crop_coverage["coverage_rate"] == 1.0
        and all(slot_gates.values())
    )
    manifest_hash = hashlib.sha256(
        args.slot_eligibility_manifest.read_bytes()
    ).hexdigest()
    payload = {
        "version": 2,
        "status": "ready_for_training" if ready else "do_not_train",
        "readiness_scope": "controlled_training_only",
        "target_accuracy_sufficiency": "not_established",
        "training_started": False,
        "eligibility_integrity": {"error_count": error_count, **errors},
        "eligibility_sha256": manifest_hash,
        "eligible_images": len(union_examples),
        "images": image_audit,
        "near_duplicate_pairs": len(near_pairs),
        "crop_coverage": crop_coverage,
        "crop_resolution": crop_resolution,
        "slot_gates": slot_gates,
        "folds": folds,
        "thresholds": {
            "min_train_images": args.min_train_images,
            "min_validation_images": args.min_validation_images,
            "near_duplicate_hamming": args.near_duplicate_hamming,
        },
        "sources": {
            "examples": portable_path(args.examples),
            "slot_eligibility_manifest": portable_path(
                args.slot_eligibility_manifest
            ),
            "regions_dir": portable_path(args.regions_dir),
            "fold_dirs": {
                slot: portable_path(path) for slot, path in fold_dirs.items()
            },
        },
        "interpretation_limit": (
            "This validates labels, files, crop presence, quantities and split "
            "readiness for controlled training. It does not prove that native "
            "equipment pixels or appearance diversity are sufficient for a "
            "target model accuracy. New strict metrics must be registered "
            "after training and evaluation."
        ),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
