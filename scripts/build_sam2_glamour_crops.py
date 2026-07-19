from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from scripts.build_global_foreground_crops import uniform_sample_indices
from scripts.evaluate_retrieval_funnel import build_fold_context
from scripts.evaluate_sam2_mask_ablation import build_sam2_crops, load_sam2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build fold-filtered SAM2 HANDS crops with compatible manifests."
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-model-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--background", type=int, default=127)
    parser.add_argument("--qa-limit", type=int, default=48)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    examples = read_jsonl(args.examples)
    quality_by_id = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
    }
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality_by_id,
        regions_dir=args.regions_dir,
        slot="HANDS",
    )
    selected_examples = list(context["examples"])
    if args.limit is not None:
        selected_examples = selected_examples[: args.limit]
    work = []
    output_path_by_key: dict[tuple[str, str], Path] = {}
    for example in selected_examples:
        example_id = str(example["example_id"])
        source_path = Path(str(example["local_path"]))
        stem = f"{source_path.parent.name}-{source_path.stem}"
        crop_dir = args.output_dir / stem / "crops"
        for record in context["records"].get(example_id, []):
            if str(record.get("slot")) != "HANDS":
                continue
            crop_name = str(record["name"])
            output_path = crop_dir / f"{stem}-{crop_name}.jpg"
            work.append({**record, "masked_output_path": output_path})
            output_path_by_key[(example_id, crop_name)] = output_path
    if not work:
        raise ValueError("No HANDS crops were selected for SAM2")

    device = resolve_torch_device(args.device)
    model, processor = load_sam2(args.sam_model_id, device)
    mask_rows, _ = build_sam2_crops(
        work,
        model,
        processor,
        device,
        args.output_dir,
        args.batch_size,
        args.background,
        save_masks=False,
    )
    mask_by_key = {
        (str(row["example_id"]), str(row["crop_name"])): row
        for row in mask_rows
    }

    qa_indices = uniform_sample_indices(len(selected_examples), args.qa_limit)
    review_rows = []
    for example_index, example in enumerate(selected_examples):
        example_id = str(example["example_id"])
        source_path = Path(str(example["local_path"]))
        stem = f"{source_path.parent.name}-{source_path.stem}"
        source_manifest_path = args.regions_dir / stem / f"{stem}-regions.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        regions = [
            region
            for region in source_manifest.get("regions", [])
            if str(region.get("slot")) == "HANDS"
            and (example_id, str(region.get("name"))) in output_path_by_key
        ]
        crop_paths = [
            output_path_by_key[(example_id, str(region["name"]))]
            for region in regions
        ]
        manifest = dict(source_manifest)
        manifest.update(
            {
                "regions": regions,
                "crop_paths": [portable_path(path) for path in crop_paths],
                "sam2_pipeline": "crop_inset_box_multimask_v1",
                "sam2_model_id": args.sam_model_id,
                "sam2_background": args.background,
                "sam2_stats": {
                    str(region["name"]): mask_by_key[
                        (example_id, str(region["name"]))
                    ]
                    for region in regions
                },
            }
        )
        example_dir = args.output_dir / stem
        example_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = example_dir / f"{stem}-regions.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        qa_path = None
        if example_index in qa_indices:
            qa_path = args.output_dir / "qa" / f"{stem}-sam2-hands-qa.jpg"
            save_qa(
                context["records"].get(example_id, []),
                output_path_by_key,
                example_id,
                qa_path,
            )
        review_rows.append(
            {
                "example_id": example_id,
                "glamour_id": int(example["glamour_id"]),
                "manifest_path": portable_path(manifest_path),
                "qa_path": portable_path(qa_path),
                "crop_count": len(crop_paths),
            }
        )

    write_jsonl(args.output_dir / "vision_review_queue.jsonl", review_rows)
    fractions = np.asarray(
        [float(row["mask_fraction"]) for row in mask_rows],
        dtype=np.float64,
    )
    ious = np.asarray(
        [float(row["sam_iou"]) for row in mask_rows],
        dtype=np.float64,
    )
    by_crop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mask_rows:
        by_crop[str(row["crop_name"])].append(row)
    summary = {
        "example_count": len(selected_examples),
        "crop_count": len(mask_rows),
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "quality_manifest": portable_path(args.quality_manifest),
        "regions_dir": portable_path(args.regions_dir),
        "sam_model_id": args.sam_model_id,
        "background": args.background,
        "qa_count": len(qa_indices),
        "mask_fraction_quantiles": quantiles(fractions),
        "sam_iou_quantiles": quantiles(ious),
        "low_iou_count": int((ious < 0.5).sum()),
        "by_crop": {
            crop: {
                "count": len(rows),
                "mean_mask_fraction": float(
                    np.mean([float(row["mask_fraction"]) for row in rows])
                ),
                "mean_sam_iou": float(
                    np.mean([float(row["sam_iou"]) for row in rows])
                ),
            }
            for crop, rows in sorted(by_crop.items())
        },
        "output": portable_path(args.output_dir),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p05", "p50", "p95", "max"),
            np.quantile(values, (0.0, 0.05, 0.5, 0.95, 1.0)),
            strict=True,
        )
    }


def save_qa(
    records: list[dict[str, Any]],
    output_path_by_key: dict[tuple[str, str], Path],
    example_id: str,
    output_path: Path,
) -> None:
    rows = [row for row in records if str(row.get("slot")) == "HANDS"]
    panels = []
    for row in rows:
        masked_path = output_path_by_key.get((example_id, str(row["name"])))
        if masked_path is None:
            continue
        with Image.open(Path(str(row["path"]))) as source:
            raw = source.convert("RGB")
        with Image.open(masked_path) as source:
            masked = source.convert("RGB")
        raw.thumbnail((220, 170), Image.Resampling.LANCZOS)
        masked.thumbnail((220, 170), Image.Resampling.LANCZOS)
        panels.append((str(row["name"]), raw, masked))
    width = 440
    height = 28 + sum(max(raw.height, masked.height) + 22 for _, raw, masked in panels)
    canvas = Image.new("RGB", (width, max(height, 1)), "white")
    draw = ImageDraw.Draw(canvas)
    y = 4
    for name, raw, masked in panels:
        draw.text((4, y), name, fill="black")
        y += 18
        canvas.paste(raw, (0, y))
        canvas.paste(masked, (220, y))
        y += max(raw.height, masked.height) + 4
        raw.close()
        masked.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)
    canvas.close()


if __name__ == "__main__":
    raise SystemExit(main())
