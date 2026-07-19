from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import write_json
from ffxiv_prism.paths import portable_path, resolve_project_path
from ffxiv_prism.vision.outfit_regions import (
    Keypoint,
    build_outfit_regions,
    region_manifest,
    render_crop_contact_sheet,
    render_debug_overlay,
    save_region_crops,
)

SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild crop QA images from saved pose manifests without rerunning pose."
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--slots",
        nargs="+",
        choices=SLOTS,
        default=list(SLOTS),
        help="Only save and render the selected slot crops.",
    )
    args = parser.parse_args()

    summaries = []
    for manifest_path in args.manifests:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_value = Path(saved["source_image"])
        source = (
            source_value
            if source_value.is_absolute()
            else resolve_project_path(source_value)
        )
        used_overlay_fallback = False
        if source is None or not source.is_file():
            overlay_fallback = manifest_path.parent / "overlay.jpg"
            if not overlay_fallback.is_file():
                raise FileNotFoundError(
                    f"Source image is unavailable: {saved['source_image']}"
                )
            source = overlay_fallback
            used_overlay_fallback = True
        keypoints = [
            Keypoint(float(row["x"]), float(row["y"]), float(row["score"]))
            for row in saved["keypoints"]
        ]
        person_box = tuple(int(value) for value in saved["person_box"])
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        regions = [
            region
            for region in build_outfit_regions(keypoints, image.size)
            if region.slot in args.slots
        ]

        sample_name = manifest_path.parent.name
        sample_dir = args.output_dir / sample_name
        crop_paths = save_region_crops(image, regions, sample_dir / "crops", source.stem)
        overlay_path = sample_dir / "overlay.jpg"
        sheet_path = sample_dir / "crops.jpg"
        sample_dir.mkdir(parents=True, exist_ok=True)
        render_debug_overlay(image, keypoints, person_box, regions).save(
            overlay_path,
            quality=94,
        )
        render_crop_contact_sheet(image, regions).save(sheet_path, quality=92)
        image.close()

        rebuilt = {
            "source_manifest": portable_path(manifest_path),
            "source_image": portable_path(source),
            "used_annotated_overlay_fallback": used_overlay_fallback,
            **region_manifest(keypoints, person_box, regions),
            "crop_paths": [portable_path(path) for path in crop_paths],
            "overlay_path": portable_path(overlay_path),
            "contact_sheet_path": portable_path(sheet_path),
        }
        write_json(sample_dir / "regions.json", rebuilt)
        summaries.append(rebuilt)

    write_json(
        args.output_dir / "summary.json",
        {"image_count": len(summaries), "examples": summaries},
    )
    print(
        json.dumps(
            {"output": portable_path(args.output_dir), "image_count": len(summaries)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
