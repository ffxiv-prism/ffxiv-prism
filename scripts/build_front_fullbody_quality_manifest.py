from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import PROCESSED_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.evaluation_quality import (
    FrontFullBodyThresholds,
    QUALITY_POLICY_VERSION,
    assess_front_full_body,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a prediction-independent frontal full-body test manifest."
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROCESSED_DIR / "glamour_examples.jsonl",
    )
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--min-clip-front-probability", type=float, default=0.50)
    parser.add_argument("--max-clip-multiple-probability", type=float, default=0.10)
    parser.add_argument(
        "--clip-view-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "front_fullbody_quality_v1.jsonl",
    )
    args = parser.parse_args()

    rows = []
    missing_regions = 0
    for example in read_jsonl(args.examples):
        example_id = str(example["example_id"])
        source_path = Path(str(example["local_path"]))
        region_path = find_region_report(args.regions_dir, source_path)
        if region_path is None:
            missing_regions += 1
            rows.append(
                {
                    "example_id": example_id,
                    "glamour_id": str(example["glamour_id"]),
                    "passed": False,
                    "quality_score": 0.0,
                    "rejection_reasons": ["missing_regions"],
                    "source_image": portable_path(source_path),
                }
            )
            continue
        report = json.loads(region_path.read_text(encoding="utf-8"))
        with Image.open(source_path) as image:
            image_size = image.size
        assessment = assess_front_full_body(
            report,
            image_size,
            FrontFullBodyThresholds(),
        )
        row = {
            "example_id": example_id,
            "glamour_id": str(example["glamour_id"]),
            "passed": assessment.passed,
            "quality_score": assessment.quality_score,
            "rejection_reasons": list(assessment.rejection_reasons),
            "metrics": dict(assessment.metrics),
            "source_image": portable_path(source_path),
            "regions_report": portable_path(region_path),
        }
        rows.append(row)
    if args.clip_view_check:
        clip_scores = score_clip_views(
            [Path(str(row["source_image"])) for row in rows],
            model_id=args.clip_model_id,
            device=args.clip_device,
            batch_size=args.clip_batch_size,
        )
        for row, scores in zip(rows, clip_scores, strict=True):
            row.setdefault("metrics", {}).update(
                {
                    f"clip_{name}_probability": score
                    for name, score in scores.items()
                }
            )
            non_front = max(
                scores["back"],
                scores["side"],
                scores["multiple"],
            )
            if row["passed"] and (
                scores["front"] <= non_front
                or scores["front"] < args.min_clip_front_probability
                or scores["multiple"] > args.max_clip_multiple_probability
            ):
                row["passed"] = False
                row["rejection_reasons"].append("clip_not_front_view")
            row["quality_score"] = float(
                (float(row["quality_score"]) + scores["front"]) / 2.0
            )

    write_jsonl(args.output, rows)
    accepted = sum(bool(row["passed"]) for row in rows)
    reasons = Counter(
        reason
        for row in rows
        for reason in row.get("rejection_reasons", [])
    )
    summary_path = args.output.with_suffix(".summary.json")
    write_json(
        summary_path,
        {
            "policy_version": QUALITY_POLICY_VERSION,
            "examples": len(rows),
            "accepted": accepted,
            "rejected": len(rows) - accepted,
            "coverage": accepted / len(rows) if rows else 0.0,
            "missing_regions": missing_regions,
            "rejection_reason_counts": dict(reasons.most_common()),
            "thresholds": FrontFullBodyThresholds().__dict__,
            "clip_view_check": args.clip_view_check,
            "clip_model_id": args.clip_model_id if args.clip_view_check else None,
            "min_clip_front_probability": (
                args.min_clip_front_probability
                if args.clip_view_check
                else None
            ),
            "max_clip_multiple_probability": (
                args.max_clip_multiple_probability
                if args.clip_view_check
                else None
            ),
            "output": portable_path(args.output),
        },
    )
    print(
        f"examples={len(rows)} accepted={accepted} "
        f"coverage={accepted / len(rows):.3f} output={args.output}"
    )
    return 0


def find_region_report(regions_dir: Path, source_path: Path) -> Path | None:
    source_stem = f"{source_path.parent.name}-{source_path.stem}"
    direct = regions_dir / source_stem / f"{source_stem}-regions.json"
    if direct.is_file():
        return direct
    matches = list(regions_dir.glob(f"{source_stem}/*-regions.json"))
    return matches[0] if len(matches) == 1 else None


def score_clip_views(
    paths: list[Path],
    *,
    model_id: str,
    device: str,
    batch_size: int,
) -> list[dict[str, float]]:
    from transformers import CLIPModel, CLIPProcessor

    resolved_device = resolve_torch_device(device)
    cache_dir = PROJECT_ROOT / "var" / "cache" / "huggingface" / "hub"
    processor = CLIPProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model = CLIPModel.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
    ).to(resolved_device).eval()
    prompt_groups = {
        "front": (
            "front view of a person",
            "a person facing toward the camera",
            "the front of a person's body is visible",
            "a person seen directly from the front",
        ),
        "back": (
            "back view of a person",
            "a person facing away from the camera",
            "the back of a person's body is visible",
            "a person seen directly from behind",
        ),
        "side": (
            "side view of a person",
            "a person seen in profile from the side",
        ),
        "cropped": (
            "a close-up of part of a person",
            "a cropped image where the full body is not visible",
        ),
        "multiple": (
            "multiple people in one image",
            "a comparison sheet showing more than one person",
        ),
    }
    labels = tuple(prompt_groups)
    prompts = [
        prompt
        for group in prompt_groups.values()
        for prompt in group
    ]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    text_inputs = {
        key: value.to(resolved_device)
        for key, value in text_inputs.items()
    }
    with torch.inference_mode():
        text_outputs = model.text_model(**text_inputs)
        text_features = model.text_projection(text_outputs.pooler_output)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        prototypes = []
        start = 0
        for group in prompt_groups.values():
            end = start + len(group)
            prototype = text_features[start:end].mean(dim=0)
            prototypes.append(
                torch.nn.functional.normalize(prototype, dim=-1)
            )
            start = end
        text_features = torch.stack(prototypes)

    output = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        for image in images:
            image.close()
        pixel_values = inputs["pixel_values"].to(resolved_device)
        with torch.inference_mode():
            image_outputs = model.vision_model(pixel_values=pixel_values)
            image_features = model.visual_projection(
                image_outputs.pooler_output
            )
            image_features = torch.nn.functional.normalize(
                image_features,
                dim=-1,
            )
            probabilities = torch.softmax(
                100.0 * image_features @ text_features.T,
                dim=-1,
            ).float().cpu().numpy()
        output.extend(
            {
                label: float(value)
                for label, value in zip(labels, row, strict=True)
            }
            for row in np.asarray(probabilities)
        )
    return output


if __name__ == "__main__":
    raise SystemExit(main())
