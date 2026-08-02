from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path, resolve_project_path  # noqa: E402
from ffxiv_prism.vision.evaluation_split import (  # noqa: E402
    load_split_assignments,
)
from scripts.train_glamour_query_encoder import (  # noqa: E402
    load_slot_eligibility,
)


HANDS_CROP_SIDES = {
    "hands_left": ("left",),
    "hands_forearm_left": ("left",),
    "hands_right": ("right",),
    "hands_forearm_right": ("right",),
    "hands_arm_cover_context": ("left", "right"),
}


def normalize_keypoints(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, float, float]]:
    output = {}
    for row in rows:
        try:
            output[str(row["name"])] = (
                float(row["x"]),
                float(row["y"]),
                float(row.get("score", 1.0)),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return output


def arm_tube_mask(
    image_size: tuple[int, int],
    keypoints: Mapping[str, tuple[float, float, float]],
    *,
    sides: Sequence[str],
    width_ratio: float,
    wrist_extension: float,
    upper_extension: float,
    minimum_score: float,
    feather: float,
) -> Image.Image | None:
    if width_ratio <= 0:
        raise ValueError("width_ratio must be positive")
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    accepted = 0
    for side in sides:
        names = [f"{side}_{joint}" for joint in ("shoulder", "elbow", "wrist")]
        if any(name not in keypoints for name in names):
            continue
        shoulder, elbow, wrist = (keypoints[name] for name in names)
        if min(shoulder[2], elbow[2], wrist[2]) < minimum_score:
            continue
        forearm_length = math.dist(elbow[:2], wrist[:2])
        if forearm_length < 2:
            continue
        upper_point = (
            elbow[0] + (shoulder[0] - elbow[0]) * upper_extension,
            elbow[1] + (shoulder[1] - elbow[1]) * upper_extension,
        )
        hand_point = (
            wrist[0] + (wrist[0] - elbow[0]) * wrist_extension,
            wrist[1] + (wrist[1] - elbow[1]) * wrist_extension,
        )
        width = max(5, int(round(forearm_length * width_ratio)))
        draw.line(
            (upper_point, elbow[:2], wrist[:2], hand_point),
            fill=255,
            width=width,
            joint="curve",
        )
        radius = width / 2
        for x, y in (upper_point, elbow[:2], wrist[:2], hand_point):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        accepted += 1
    if not accepted:
        mask.close()
        return None
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def composite_crop(
    image: Image.Image,
    mask: Image.Image,
    box: Sequence[int],
    background: tuple[int, int, int],
) -> Image.Image:
    crop_box = tuple(int(value) for value in box)
    source_crop = image.crop(crop_box).convert("RGB")
    mask_crop = mask.crop(crop_box).convert("L")
    canvas = Image.new("RGB", source_crop.size, background)
    output = Image.composite(source_crop, canvas, mask_crop)
    source_crop.close()
    mask_crop.close()
    canvas.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build HANDS crops that suppress pixels outside pose-defined "
            "forearm/hand tubes before query encoding."
        )
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--slot-eligibility-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument(
        "--fold-roles",
        nargs="+",
        choices=("train", "validation"),
        default=["validation"],
    )
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width-ratio", type=float, default=0.58)
    parser.add_argument("--wrist-extension", type=float, default=0.48)
    parser.add_argument("--upper-extension", type=float, default=0.25)
    parser.add_argument("--minimum-score", type=float, default=0.2)
    parser.add_argument("--feather", type=float, default=2.0)
    parser.add_argument("--background", type=int, nargs=3, default=[238, 234, 224])
    args = parser.parse_args()

    if args.width_ratio <= 0:
        parser.error("width-ratio must be positive")
    if args.wrist_extension < 0 or args.upper_extension < 0:
        parser.error("extension ratios must not be negative")
    if not 0 <= args.minimum_score <= 2:
        parser.error("minimum-score is outside the supported range")
    if any(not 0 <= value <= 255 for value in args.background):
        parser.error("background channels must be in [0, 255]")

    examples = read_jsonl(args.examples)
    eligibility = load_slot_eligibility(args.slot_eligibility_manifest)
    assignments = load_split_assignments(args.fold_manifest)
    selected_roles = set(args.fold_roles)
    selected_examples = [
        example
        for example in examples
        if assignments.get(str(example["glamour_id"])) in selected_roles
        and "HANDS" in eligibility.get(str(example["example_id"]), set())
    ]
    if not selected_examples:
        raise ValueError("No eligible HANDS examples matched the requested fold roles")

    background = tuple(int(value) for value in args.background)
    status_counts: Counter[str] = Counter()
    written_examples = 0
    for example in selected_examples:
        source_path = resolve_project_path(example.get("local_path"))
        if source_path is None or not source_path.is_file():
            status_counts["missing_source"] += 1
            continue
        stem = f"{source_path.parent.name}-{source_path.stem}"
        manifest_path = args.regions_dir / stem / f"{stem}-regions.json"
        if not manifest_path.is_file():
            status_counts["missing_manifest"] += 1
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        keypoints = normalize_keypoints(manifest.get("keypoints") or [])
        output_regions = [
            dict(region)
            for region in manifest.get("regions") or []
            if str(region.get("name")) in HANDS_CROP_SIDES
        ]
        if not output_regions:
            status_counts["no_hands_regions"] += 1
            continue

        output_crop_dir = args.output_dir / stem / "crops"
        output_crop_dir.mkdir(parents=True, exist_ok=True)
        crop_paths = []
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        try:
            masks: dict[tuple[str, ...], Image.Image | None] = {}
            for region in output_regions:
                crop_name = str(region["name"])
                sides = HANDS_CROP_SIDES[crop_name]
                if sides not in masks:
                    masks[sides] = arm_tube_mask(
                        image.size,
                        keypoints,
                        sides=sides,
                        width_ratio=args.width_ratio,
                        wrist_extension=args.wrist_extension,
                        upper_extension=args.upper_extension,
                        minimum_score=args.minimum_score,
                        feather=args.feather,
                    )
                mask = masks[sides]
                # The parent directory and manifest retain the source stem.  Do
                # not repeat very long glamour titles in every derived crop;
                # that can exceed the legacy Windows path-length limit.
                crop_path = output_crop_dir / f"{crop_name}.jpg"
                if mask is None:
                    output = image.crop(tuple(int(v) for v in region["box"])).convert(
                        "RGB"
                    )
                    status_counts["fallback_unmasked_crop"] += 1
                else:
                    output = composite_crop(
                        image,
                        mask,
                        region["box"],
                        background,
                    )
                    status_counts["pose_tube_crop"] += 1
                output.save(crop_path, quality=94)
                output.close()
                crop_paths.append(portable_path(crop_path))
            for mask in masks.values():
                if mask is not None:
                    mask.close()
        finally:
            image.close()

        output_manifest = {
            **manifest,
            "regions": output_regions,
            "crop_paths": crop_paths,
            "source_image": portable_path(source_path),
            "hands_crop_pipeline": "pose_forearm_tube_v1",
            "hands_crop_parameters": {
                "width_ratio": args.width_ratio,
                "wrist_extension": args.wrist_extension,
                "upper_extension": args.upper_extension,
                "minimum_score": args.minimum_score,
                "feather": args.feather,
                "background": list(background),
            },
        }
        output_manifest_path = args.output_dir / stem / f"{stem}-regions.json"
        output_manifest_path.write_text(
            json.dumps(output_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written_examples += 1
        if written_examples % 100 == 0 or written_examples == len(selected_examples):
            print(
                f"pose_tube={written_examples}/{len(selected_examples)}",
                flush=True,
            )

    summary: dict[str, Any] = {
        "version": 1,
        "pipeline": "pose_forearm_tube_v1",
        "selected_example_count": len(selected_examples),
        "written_example_count": written_examples,
        "fold_roles": sorted(selected_roles),
        "blind_test_accessed": False,
        "status_counts": dict(status_counts),
        "parameters": {
            "width_ratio": args.width_ratio,
            "wrist_extension": args.wrist_extension,
            "upper_extension": args.upper_extension,
            "minimum_score": args.minimum_score,
            "feather": args.feather,
            "background": list(background),
        },
        "sources": {
            "examples": portable_path(args.examples),
            "slot_eligibility_manifest": portable_path(
                args.slot_eligibility_manifest
            ),
            "fold_manifest": portable_path(args.fold_manifest),
            "regions_dir": portable_path(args.regions_dir),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
