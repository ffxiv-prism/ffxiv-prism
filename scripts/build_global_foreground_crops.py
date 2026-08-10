from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import PROCESSED_DIR, portable_path, resolve_project_path
from ffxiv_prism.vision.human_parsing import (
    DEFAULT_HUMAN_PARSING_MODEL,
    GlobalForegroundResult,
    HumanParser,
    build_global_foreground_mask,
    composite_foreground,
    transparent_foreground,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_REGIONS = Path("data/debug/outfit-regions/overlap-final-201")
DEFAULT_OUTPUT = Path("data/debug/global-foreground")
CORE_SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse each original screenshot once, then crop the source and its "
            "global foreground mask with identical equipment-region boxes."
        )
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROCESSED_DIR / "glamour_examples.jsonl",
    )
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=DEFAULT_HUMAN_PARSING_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        help="Optional grouped split manifest used to limit source glamours.",
    )
    parser.add_argument(
        "--fold-split",
        choices=("train", "validation", "blind_test"),
        default="blind_test",
    )
    parser.add_argument(
        "--quality-manifest",
        type=Path,
        help="Optional image-quality manifest; only rows with passed=true remain.",
    )
    parser.add_argument(
        "--slots",
        nargs="+",
        choices=CORE_SLOTS,
        default=CORE_SLOTS,
        help="Only write crops for these equipment slots.",
    )
    parser.add_argument(
        "--crop-only",
        action="store_true",
        help="Skip full-resolution masks and transparent foreground PNGs.",
    )
    parser.add_argument("--qa-limit", type=int, default=60)
    parser.add_argument(
        "--background",
        default="127,127,127",
        help="RGB background used only for model-input crops.",
    )
    args = parser.parse_args()

    background = parse_background(args.background)
    examples = read_jsonl(args.examples)
    if args.fold_manifest is not None:
        fold_payload = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
        examples = filter_examples_by_fold(
            examples,
            fold_payload,
            split=args.fold_split,
        )
    if args.quality_manifest is not None:
        examples = filter_examples_by_quality(
            examples,
            read_jsonl(args.quality_manifest),
        )
    if args.limit is not None:
        examples = examples[: args.limit]
    work = collect_work(examples, args.regions_dir)
    if not work:
        raise ValueError("No examples with source images and region manifests")

    qa_indices = uniform_sample_indices(len(work), args.qa_limit)
    parser_model = HumanParser(args.model_id, args.device)
    review_rows = []
    flag_counts: Counter[str] = Counter()

    for start in range(0, len(work), args.batch_size):
        batch = work[start : start + args.batch_size]
        images = [load_rgb(row["source_path"]) for row in batch]
        try:
            segmentations = parser_model.parse(images)
            for offset, (row, image, segmentation) in enumerate(
                zip(batch, images, segmentations, strict=True)
            ):
                result = build_global_foreground_mask(
                    image,
                    segmentation,
                    keypoints=row["manifest"].get("keypoints", []),
                )
                example_index = start + offset
                review_rows.append(
                    save_outputs(
                        row,
                        image,
                        result,
                        args.output_dir,
                        background,
                        selected_slots=frozenset(args.slots),
                        write_full_foreground=not args.crop_only,
                        write_qa=example_index in qa_indices,
                    )
                )
                flag_counts.update(result.quality_flags)
                result.mask.close()
        finally:
            for image in images:
                image.close()
        done = min(start + args.batch_size, len(work))
        if done == len(work) or done % max(args.batch_size * 10, 1) == 0:
            print(f"global_foreground={done}/{len(work)}", flush=True)

    review_path = args.output_dir / "vision_review_queue.jsonl"
    write_jsonl(review_path, review_rows)
    summary = {
        "example_count": len(review_rows),
        "fold_manifest": portable_path(args.fold_manifest),
        "fold_split": args.fold_split if args.fold_manifest is not None else None,
        "quality_manifest": portable_path(args.quality_manifest),
        "slots": sorted(args.slots),
        "crop_only": args.crop_only,
        "parser_model_id": args.model_id,
        "background_for_model_crops": list(background),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "qa_preview_count": sum(bool(row.get("qa_preview_path")) for row in review_rows),
        "review_queue": portable_path(review_path),
        "review_schema": {
            "foreground_completeness": "0=catastrophic, 4=all character equipment retained",
            "background_suppression": "0=mostly background, 4=background cleanly removed",
            "region_alignment": "0=misaligned, 4=all model crops aligned with source",
            "decision": "accept, raw_only, or reject",
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"examples={len(review_rows)} qa={summary['qa_preview_count']} "
        f"review_queue={review_path}"
    )
    return 0


def filter_examples_by_fold(
    examples: list[dict[str, Any]],
    fold_payload: dict[str, Any],
    *,
    split: str,
) -> list[dict[str, Any]]:
    assignments = {
        str(glamour_id): str(value)
        for glamour_id, value in (fold_payload.get("assignments") or {}).items()
    }
    if not assignments:
        raise ValueError("Fold manifest has no grouped assignments")
    selected = [
        example
        for example in examples
        if assignments.get(str(example.get("glamour_id"))) == split
    ]
    if not selected:
        raise ValueError(f"Fold split contains no matching examples: {split}")
    return selected


def filter_examples_by_quality(
    examples: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    passed_by_id = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in quality_rows
    }
    missing = [
        str(example["example_id"])
        for example in examples
        if str(example["example_id"]) not in passed_by_id
    ]
    if missing:
        raise ValueError(f"Quality manifest is missing {len(missing)} examples")
    selected = [
        example
        for example in examples
        if passed_by_id[str(example["example_id"])]
    ]
    if not selected:
        raise ValueError("Quality filter contains no passing examples")
    return selected


def collect_work(
    examples: list[dict[str, Any]],
    regions_dir: Path,
) -> list[dict[str, Any]]:
    work = []
    for example in examples:
        source_path = resolve_project_path(example.get("local_path"))
        if source_path is None or not source_path.is_file():
            continue
        stem = f"{source_path.parent.name}-{source_path.stem}"
        manifest_path = regions_dir / stem / f"{stem}-regions.json"
        if not manifest_path.is_file():
            continue
        work.append(
            {
                "example": example,
                "source_path": source_path,
                "stem": stem,
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            }
        )
    return work


def save_outputs(
    row: dict[str, Any],
    image: Image.Image,
    result: GlobalForegroundResult,
    output_dir: Path,
    background: tuple[int, int, int],
    *,
    selected_slots: frozenset[str] = frozenset(CORE_SLOTS),
    write_full_foreground: bool = True,
    write_qa: bool,
) -> dict[str, Any]:
    stem = row["stem"]
    example_dir = output_dir / stem
    crop_dir = example_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    mask_path = None
    transparent_path = None
    if write_full_foreground:
        mask_path = example_dir / f"{stem}-foreground-mask.png"
        transparent_path = example_dir / f"{stem}-foreground.png"
        result.mask.save(mask_path)
        rgba = transparent_foreground(image, result.mask)
        rgba.save(transparent_path)
        rgba.close()

    crop_paths = []
    regions = select_regions(row["manifest"].get("regions", []), selected_slots)
    for region in regions:
        box = tuple(int(value) for value in region["box"])
        source_crop = image.crop(box)
        mask_crop = result.mask.crop(box)
        try:
            model_crop = composite_foreground(
                source_crop,
                mask_crop,
                background,
            )
            crop_path = crop_dir / f"{stem}-{region['name']}.jpg"
            model_crop.save(crop_path, quality=92)
            model_crop.close()
            crop_paths.append(crop_path)
        finally:
            source_crop.close()
            mask_crop.close()

    manifest = dict(row["manifest"])
    manifest.update(
        {
            "regions": regions,
            "source_image": portable_path(row["source_path"]),
            "foreground_pipeline": "original_image_global_mask_v1",
            "foreground_mask_path": portable_path(mask_path),
            "transparent_foreground_path": portable_path(transparent_path),
            "model_crop_background": list(background),
            "foreground_stats": foreground_stats(result),
            "crop_paths": [portable_path(path) for path in crop_paths],
        }
    )
    manifest_path = example_dir / f"{stem}-regions.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    qa_path = None
    if write_qa:
        qa_dir = output_dir / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_path = qa_dir / f"{stem}-foreground-qa.jpg"
        save_qa_preview(image, result, qa_path, row["example"])

    return {
        "example_id": row["example"]["example_id"],
        "glamour_id": row["example"]["glamour_id"],
        "source_image": portable_path(row["source_path"]),
        "mask_path": portable_path(mask_path),
        "transparent_foreground_path": portable_path(transparent_path),
        "regions_manifest_path": portable_path(manifest_path),
        "qa_preview_path": portable_path(qa_path),
        "automatic_stats": foreground_stats(result),
        "vision_review": {
            "status": "pending" if write_qa else "not_sampled",
            "foreground_completeness": None,
            "background_suppression": None,
            "region_alignment": None,
            "decision": None,
            "notes": None,
            "reviewer": None,
        },
    }


def select_regions(
    regions: list[dict[str, Any]],
    selected_slots: frozenset[str],
) -> list[dict[str, Any]]:
    if not selected_slots:
        raise ValueError("At least one crop slot is required")
    return [
        region
        for region in regions
        if str(region.get("slot")) in selected_slots
    ]


def foreground_stats(result: GlobalForegroundResult) -> dict[str, Any]:
    return {
        "semantic_person_fraction": result.person_fraction,
        "final_foreground_fraction": result.foreground_fraction,
        "repaired_hole_fraction": result.repaired_hole_fraction,
        "keypoint_coverage": result.keypoint_coverage,
        "used_pose_envelope": result.used_pose_envelope,
        "quality_flags": list(result.quality_flags),
    }


def save_qa_preview(
    image: Image.Image,
    result: GlobalForegroundResult,
    output_path: Path,
    example: dict[str, Any],
) -> None:
    foreground = composite_foreground(image, result.mask)
    try:
        panels = [image.convert("RGB"), result.mask.convert("RGB"), foreground]
        width = 360
        resized = []
        for panel in panels:
            copy = panel.copy()
            copy.thumbnail((width, 640), Image.Resampling.LANCZOS)
            resized.append(copy)
        height = max(panel.height for panel in resized)
        canvas = Image.new("RGB", (width * 3, height + 34), "white")
        for index, panel in enumerate(resized):
            canvas.paste(panel, (index * width + (width - panel.width) // 2, 34))
            panel.close()
        draw = ImageDraw.Draw(canvas)
        flags = ",".join(result.quality_flags) or "none"
        draw.text(
            (4, 8),
            (
                f"glamour={example['glamour_id']} "
                f"foreground={result.foreground_fraction:.2f} "
                f"keypoints={result.keypoint_coverage:.2f} flags={flags}"
            ),
            fill="black",
        )
        canvas.save(output_path, quality=90)
        canvas.close()
    finally:
        foreground.close()


def uniform_sample_indices(count: int, limit: int) -> set[int]:
    if count <= 0 or limit <= 0:
        return set()
    if limit >= count:
        return set(range(count))
    return {
        int(round(value))
        for value in np.linspace(0, count - 1, num=limit)
    }


def parse_background(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError("Background must be three comma-separated values from 0 to 255")
    return parts[0], parts[1], parts[2]


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


if __name__ == "__main__":
    raise SystemExit(main())
