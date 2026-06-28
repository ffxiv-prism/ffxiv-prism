from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from ffxiv_prism.paths import portable_path
from ffxiv_prism.vision.outfit_regions import (
    DEFAULT_POSE_MODEL,
    SinglePersonPoseEstimator,
    build_outfit_regions,
    region_manifest,
    render_crop_contact_sheet,
    render_debug_overlay,
    save_region_crops,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pose-driven FFXIV outfit region crops for single-person screenshots.")
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--dataset", type=Path, help="Processed glamour JSONL whose local_path values should be analyzed.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("data/debug/outfit-regions"))
    parser.add_argument("--model-id", default=DEFAULT_POSE_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing per-image region manifest and crops.",
    )
    args = parser.parse_args()

    image_paths = list(args.images)
    if args.dataset:
        for line in args.dataset.read_text(encoding="utf-8").splitlines():
            if line.strip():
                image_paths.append(Path(json.loads(line)["local_path"]))
    image_paths = list(dict.fromkeys(image_paths))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        parser.error("Provide image paths or --dataset")

    estimator = SinglePersonPoseEstimator(args.model_id, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    visibility_counts: Counter[tuple[str, str]] = Counter()
    for image_path in image_paths:
        stem = f"{image_path.parent.name}-{image_path.stem}"
        manifest_path = args.output_dir / stem / f"{stem}-regions.json"
        if args.skip_existing and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summaries.append(manifest)
            for slot in ("HEAD", "BODY", "HANDS", "LEGS", "FEET"):
                slot_regions = [
                    region
                    for region in manifest.get("regions", [])
                    if region.get("slot") == slot
                ]
                if slot_regions:
                    best = max(
                        slot_regions,
                        key=lambda region: float(region.get("confidence", 0.0)),
                    )
                    visibility_counts[(slot, str(best["visibility"]))] += 1
            continue
        image = Image.open(image_path).convert("RGB")
        keypoints, person_box = estimator.estimate(image, refine=not args.no_refine)
        regions = build_outfit_regions(keypoints, image.size)
        crop_dir = args.output_dir / stem / "crops"
        crop_paths = save_region_crops(image, regions, crop_dir, stem)
        overlay_path = args.output_dir / stem / f"{stem}-overlay.jpg"
        sheet_path = args.output_dir / stem / f"{stem}-crops.jpg"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        render_debug_overlay(image, keypoints, person_box, regions).save(overlay_path, quality=94)
        render_crop_contact_sheet(image, regions).save(sheet_path, quality=92)
        manifest = {
            "source_image": portable_path(image_path),
            "model_id": estimator.model_id,
            "single_person_assumption": True,
            **region_manifest(keypoints, person_box, regions),
            "crop_paths": [portable_path(path) for path in crop_paths],
            "overlay_path": portable_path(overlay_path),
            "contact_sheet_path": portable_path(sheet_path),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(manifest)
        if not args.quiet:
            print(f"{image_path}: person_box={person_box}")
        for slot in ("HEAD", "BODY", "HANDS", "LEGS", "FEET"):
            slot_regions = [region for region in regions if region.slot == slot]
            best = max(slot_regions, key=lambda region: region.confidence)
            visibility_counts[(slot, best.visibility)] += 1
            if not args.quiet:
                print(f"  {slot}: {best.visibility} confidence={best.confidence:.3f} crops={len(slot_regions)}")
        if not args.quiet:
            print(f"  overlay={overlay_path}")
            print(f"  crops={sheet_path}")

    summary_path = args.output_dir / "summary.json"
    summary = {
        "image_count": len(summaries),
        "single_person_assumption": True,
        "model_id": estimator.model_id,
        "by_slot_visibility": {
            slot: {
                visibility: visibility_counts[(slot, visibility)]
                for visibility in ("visible", "partially_visible", "occluded", "out_of_frame", "pose_uncertain")
                if visibility_counts[(slot, visibility)]
            }
            for slot in ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
        },
        "examples": summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
