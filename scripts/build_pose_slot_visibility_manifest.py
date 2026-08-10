from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


# Initialize Conda's BLAS/OpenMP runtime before importing CUDA PyTorch through
# the project modules. When a test process already imported Torch, avoid
# introducing the second runtime during this module import.
if "torch" not in sys.modules:
    np.linalg.inv(np.eye(2, dtype=np.float32))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path, resolve_project_path  # noqa: E402
from ffxiv_prism.torch_device import resolve_torch_device  # noqa: E402
from ffxiv_prism.vision.outfit_regions import (  # noqa: E402
    CropRegion,
    SinglePersonPoseEstimator,
    build_outfit_regions,
    pose_quality,
)


SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
PRIMARY_REGION_NAMES = {
    "HEAD": ("head_tight",),
    "BODY": ("body_torso", "body_upper_core"),
    "HANDS": ("hands_left", "hands_right"),
    "LEGS": ("legs_left", "legs_right"),
    "FEET": ("feet_left", "feet_right"),
}
MIN_REGION_SHORT_SIDE = {
    "HEAD": 24,
    "BODY": 36,
    "HANDS": 20,
    "LEGS": 24,
    "FEET": 18,
}
USABLE_VISIBILITY = {"visible", "partially_visible", "occluded"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lightweight, full-dataset pose visibility manifest. "
            "This is a structural prefilter, not a substitute for semantic "
            "human review of equipment occlusion."
        )
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path(
            "data/processed/glamour_examples_huiji_models_closed.jsonl"
        ),
    )
    parser.add_argument(
        "--quality-manifest",
        type=Path,
        default=Path(
            "data/processed/front_fullbody_quality_final_v46.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/outputs/training-readiness-audit-20260730/"
            "pose-slot-visibility.jsonl"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "data/outputs/training-readiness-audit-20260730/"
            "pose-slot-visibility.summary.json"
        ),
    )
    parser.add_argument(
        "--model-id",
        default="usyd-community/vitpose-base-simple",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def labelled_slots(example: Mapping[str, Any]) -> tuple[str, ...]:
    labels = example.get("core_slot_labels") or {}
    return tuple(
        slot
        for slot in SLOTS
        if ((labels.get(slot) or {}).get("piece_ids"))
    )


def region_short_side(region: CropRegion) -> int:
    left, top, right, bottom = region.box
    return min(max(0, right - left), max(0, bottom - top))


def summarize_slot_regions(
    slot: str,
    regions: Sequence[CropRegion],
) -> dict[str, Any]:
    expected_names = PRIMARY_REGION_NAMES[slot]
    by_name = {region.name: region for region in regions if region.slot == slot}
    selected = [by_name[name] for name in expected_names if name in by_name]
    missing_names = [name for name in expected_names if name not in by_name]
    visible_count = sum(
        region.visibility == "visible" for region in selected
    )
    usable_count = sum(
        region.visibility in USABLE_VISIBILITY for region in selected
    )
    short_sides = [region_short_side(region) for region in selected]
    confidences = [float(region.confidence) for region in selected]
    edge_risk = any(
        "source_edge_risk" in region.tags for region in selected
    )
    enough_pixels = bool(short_sides) and min(short_sides) >= (
        MIN_REGION_SHORT_SIDE[slot]
    )

    if (
        not missing_names
        and visible_count == len(expected_names)
        and enough_pixels
        and not edge_risk
    ):
        decision = "clear"
    elif usable_count and enough_pixels:
        decision = "partial"
    else:
        decision = "unusable"
    return {
        "decision": decision,
        "expected_regions": list(expected_names),
        "missing_regions": missing_names,
        "region_visibility": {
            region.name: region.visibility for region in selected
        },
        "region_confidence": {
            region.name: float(region.confidence) for region in selected
        },
        "region_short_side_px": {
            region.name: region_short_side(region) for region in selected
        },
        "minimum_confidence": min(confidences) if confidences else 0.0,
        "minimum_short_side_px": min(short_sides) if short_sides else 0,
        "source_edge_risk": edge_risk,
        "review_source": "pose_visibility_prefilter",
        "semantic_review_required": True,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "processed_images": len(rows),
        "pose_error_count": sum(bool(row.get("error")) for row in rows),
        "slots": {},
    }
    for slot in SLOTS:
        decisions = Counter(
            str((row.get("slot_visibility") or {}).get(slot, {}).get("decision"))
            for row in rows
            if slot in (row.get("labelled_slots") or [])
            and not row.get("error")
        )
        labelled = sum(
            slot in (row.get("labelled_slots") or []) for row in rows
        )
        output["slots"][slot] = {
            "labelled_images": labelled,
            "decision_counts": dict(sorted(decisions.items())),
        }
    return output


def main() -> int:
    args = parse_args()
    examples_path = resolve_input(args.examples)
    quality_path = resolve_input(args.quality_manifest)
    output_path = resolve_input(args.output)
    summary_path = resolve_input(args.summary)
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")

    quality_passed = {
        str(row["example_id"])
        for row in read_jsonl(quality_path)
        if row.get("passed")
    }
    examples = [
        row
        for row in read_jsonl(examples_path)
        if str(row.get("example_id")) in quality_passed
    ]
    if args.limit is not None:
        examples = examples[: args.limit]

    resumed_rows = (
        read_jsonl(output_path)
        if args.resume and output_path.is_file()
        else []
    )
    target_ids = {str(example.get("example_id")) for example in examples}
    resumed_by_id = {
        str(row.get("example_id")): row
        for row in resumed_rows
        if str(row.get("example_id")) in target_ids
    }
    rows = [
        resumed_by_id[example_id]
        for example in examples
        if (example_id := str(example.get("example_id"))) in resumed_by_id
    ]
    completed_ids = set(resumed_by_id)
    remaining = [
        example
        for example in examples
        if str(example.get("example_id")) not in completed_ids
    ]
    device = resolve_torch_device(args.device)
    estimator = SinglePersonPoseEstimator(args.model_id, str(device))
    for index, example in enumerate(remaining, start=1):
        example_id = str(example.get("example_id"))
        image_path = resolve_project_path(example.get("local_path"))
        labelled = labelled_slots(example)
        base = {
            "example_id": example_id,
            "glamour_id": str(example.get("glamour_id")),
            "local_path": str(example.get("local_path")),
            "labelled_slots": list(labelled),
        }
        try:
            if image_path is None or not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            keypoints, person_box = estimator.estimate(
                image,
                refine=not args.no_refine,
            )
            regions = build_outfit_regions(keypoints, image.size)
            row = {
                **base,
                "pose_quality": pose_quality(keypoints, person_box),
                "slot_visibility": {
                    slot: summarize_slot_regions(slot, regions)
                    for slot in labelled
                },
                "error": None,
            }
            image.close()
        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            UnidentifiedImageError,
            ValueError,
        ) as exc:
            row = {
                **base,
                "pose_quality": None,
                "slot_visibility": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        if index % args.checkpoint_every == 0:
            write_jsonl(output_path, rows)
            print(
                f"processed={len(rows)}/{len(examples)} device={device}",
                flush=True,
            )

    write_jsonl(output_path, rows)
    summary = {
        "version": 1,
        "examples": portable_path(examples_path),
        "quality_manifest": portable_path(quality_path),
        "output": portable_path(output_path),
        "model_id": args.model_id,
        "device": str(device),
        "policy": {
            "primary_region_names": PRIMARY_REGION_NAMES,
            "minimum_region_short_side_px": MIN_REGION_SHORT_SIDE,
            "semantic_review_required": True,
        },
        **summarize_rows(rows),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["pose_error_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
