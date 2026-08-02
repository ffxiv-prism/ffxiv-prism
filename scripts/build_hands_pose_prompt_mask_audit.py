from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402
from ffxiv_prism.matching.hands_morphology import (  # noqa: E402
    HANDS_MORPHOLOGY_CLASSES,
    hands_morphology_class,
)
from ffxiv_prism.paths import portable_path, resolve_project_path  # noqa: E402
from ffxiv_prism.torch_device import resolve_torch_device  # noqa: E402
from ffxiv_prism.vision.evaluation_split import (  # noqa: E402
    load_split_assignments,
)
from scripts.evaluate_glamour_matching import collect_crop_records  # noqa: E402
from scripts.evaluate_sam2_mask_ablation import load_sam2  # noqa: E402
from scripts.train_glamour_query_encoder import (  # noqa: E402
    load_slot_eligibility,
)


PRIMARY_CROPS = ("hands_left", "hands_right")
MORPHOLOGY_FOREARM_FRACTIONS = {
    "minimal": 0.10,
    "glove": 0.35,
    "long_armor": 0.70,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict-fold HANDS annotation audit using pose-guided "
            "positive/negative SAM2 prompts rather than a whole-crop box."
        )
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--slot-eligibility-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--fold-role", choices=("train", "validation"), default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-per-morphology", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--sam-model-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--background", type=int, default=127)
    args = parser.parse_args()
    if args.sample_per_morphology < 1:
        parser.error("sample-per-morphology must be positive")

    examples = read_jsonl(args.examples)
    eligibility = load_slot_eligibility(args.slot_eligibility_manifest)
    assignments = load_split_assignments(args.fold_manifest)
    eligible = [
        row
        for row in examples
        if "HANDS" in eligibility.get(str(row.get("example_id")), set())
        and assignments.get(str(row.get("glamour_id"))) == args.fold_role
        and ((row.get("core_slot_labels") or {}).get("HANDS") or {}).get("item_name")
    ]
    records = collect_crop_records(eligible, args.regions_dir)
    eligible = [
        row
        for row in eligible
        if set(PRIMARY_CROPS).issubset(
            {
                str(record.get("name"))
                for record in records.get(str(row["example_id"]), [])
            }
        )
        and source_manifest_path(row, args.regions_dir).is_file()
    ]
    selected = stratified_examples(
        eligible,
        per_morphology=args.sample_per_morphology,
        seed=args.seed,
    )
    expected_count = args.sample_per_morphology * len(HANDS_MORPHOLOGY_CLASSES)
    if len(selected) != expected_count:
        raise ValueError(
            f"Only selected {len(selected)}/{expected_count} diverse examples"
        )

    reference_by_id = select_reference_paths(
        selected,
        read_jsonl(args.index_dir / "metadata.jsonl"),
    )
    missing_references = sorted(
        str(row["example_id"])
        for row in selected
        if str(row["example_id"]) not in reference_by_id
    )
    if missing_references:
        raise ValueError(
            "Canonical HANDS references are missing for selected examples: "
            f"{missing_references}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = build_prompt_records(selected, records, args.regions_dir)
    device = resolve_torch_device(args.device)
    model, processor = load_sam2(args.sam_model_id, device)
    mask_rows = run_pose_prompt_sam(
        work,
        model=model,
        processor=processor,
        device=device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        background=args.background,
    )
    mask_by_key = {
        (str(row["example_id"]), str(row["crop_name"])): row for row in mask_rows
    }
    queue = []
    for example in selected:
        example_id = str(example["example_id"])
        qa_path = args.output_dir / "qa" / f"{example_id.replace(':', '-')}.jpg"
        save_example_qa(
            example,
            records[example_id],
            mask_by_key,
            reference_by_id[example_id],
            qa_path,
        )
        label = (example.get("core_slot_labels") or {}).get("HANDS") or {}
        queue.append(
            {
                "example_id": example_id,
                "glamour_id": int(example["glamour_id"]),
                "item_name": str(label.get("item_name") or ""),
                "morphology": hands_morphology_class(label.get("item_name")),
                "model_gender": str(example.get("model_gender") or "unknown"),
                "model_race": str(example.get("model_race") or "unknown"),
                "qa_path": portable_path(qa_path),
                "canonical_reference_path": portable_path(reference_by_id[example_id]),
                "crops": {
                    crop_name: mask_by_key[(example_id, crop_name)]
                    for crop_name in PRIMARY_CROPS
                },
                "manual_decision": None,
                "manual_failure_reason": None,
            }
        )
    write_jsonl(args.output_dir / "review_queue.jsonl", queue)
    summary = summarize_audit(
        queue,
        fold_role=args.fold_role,
        fold_manifest=args.fold_manifest,
        eligibility_manifest=args.slot_eligibility_manifest,
        sam_model_id=args.sam_model_id,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
    )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def source_manifest_path(example: Mapping[str, Any], regions_dir: Path) -> Path:
    source_path = Path(str(example["local_path"]))
    stem = f"{source_path.parent.name}-{source_path.stem}"
    return regions_dir / stem / f"{stem}-regions.json"


def appearance_key(example: Mapping[str, Any]) -> tuple[int, ...]:
    label = (example.get("core_slot_labels") or {}).get("HANDS") or {}
    return tuple(sorted(int(value) for value in label.get("piece_ids") or []))


def select_reference_paths(
    examples: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    def available(row: Mapping[str, Any]) -> bool:
        path = resolve_project_path(row.get("local_path"))
        return path is not None and path.is_file()

    output = {}
    for example in examples:
        example_id = str(example["example_id"])
        expected_ids = set(appearance_key(example))
        gender = str(example.get("model_gender") or "").lower()
        candidates = [
            row
            for row in metadata
            if str(row.get("slot")) == "HANDS"
            and str(row.get("candidate_crop")) == "both_hands"
            and str(row.get("model_gender") or "").lower() == gender
            and expected_ids.intersection(
                int(piece["piece_id"]) for piece in row.get("pieces", [])
            )
            and available(row)
        ]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda row: (
                str(row.get("race") or ""),
                str(row.get("local_path") or ""),
            ),
        )
        path = resolve_project_path(selected.get("local_path"))
        if path is not None:
            output[example_id] = path
    return output


def stable_sample_key(example: Mapping[str, Any], seed: int) -> bytes:
    return hashlib.sha256(
        f"hands-mask-audit:{seed}:{example.get('example_id')}".encode("utf-8")
    ).digest()


def stratified_examples(
    examples: Sequence[dict[str, Any]],
    *,
    per_morphology: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = []
    used_glamours: set[str] = set()
    used_appearances: set[tuple[int, ...]] = set()
    for morphology in HANDS_MORPHOLOGY_CLASSES:
        candidates = sorted(
            (
                row
                for row in examples
                if hands_morphology_class(
                    ((row.get("core_slot_labels") or {}).get("HANDS") or {}).get(
                        "item_name"
                    )
                )
                == morphology
            ),
            key=lambda row: stable_sample_key(row, seed),
        )
        morphology_rows = []
        for row in candidates:
            glamour_id = str(row.get("glamour_id"))
            identity = appearance_key(row)
            if glamour_id in used_glamours or identity in used_appearances:
                continue
            morphology_rows.append(row)
            used_glamours.add(glamour_id)
            used_appearances.add(identity)
            if len(morphology_rows) == per_morphology:
                break
        selected.extend(morphology_rows)
    return selected


def build_prompt_records(
    examples: Sequence[dict[str, Any]],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    regions_dir: Path,
) -> list[dict[str, Any]]:
    output = []
    for example in examples:
        example_id = str(example["example_id"])
        manifest = json.loads(
            source_manifest_path(example, regions_dir).read_text(encoding="utf-8")
        )
        keypoints = {str(row.get("name")): row for row in manifest.get("keypoints", [])}
        regions = {str(row.get("name")): row for row in manifest.get("regions", [])}
        record_by_name = {str(row.get("name")): row for row in records[example_id]}
        item_name = ((example.get("core_slot_labels") or {}).get("HANDS") or {}).get(
            "item_name"
        )
        morphology = hands_morphology_class(item_name)
        for crop_name in PRIMARY_CROPS:
            side = crop_name.rsplit("_", 1)[-1]
            prompt = pose_prompt_for_crop(
                keypoints,
                regions[crop_name].get("box") or (),
                side=side,
                morphology=morphology,
            )
            if prompt is None:
                raise ValueError(f"Missing pose prompt for {example_id} {crop_name}")
            output.append(
                {
                    "example_id": example_id,
                    "crop_name": crop_name,
                    "path": str(record_by_name[crop_name]["path"]),
                    "morphology": morphology,
                    **prompt,
                }
            )
    return output


def pose_prompt_for_crop(
    keypoints: Mapping[str, Mapping[str, Any]],
    crop_box: Sequence[float],
    *,
    side: str,
    morphology: str,
    minimum_score: float = 0.2,
) -> dict[str, Any] | None:
    if len(crop_box) != 4 or morphology not in MORPHOLOGY_FOREARM_FRACTIONS:
        return None
    wrist = keypoints.get(f"{side}_wrist")
    elbow = keypoints.get(f"{side}_elbow")
    if wrist is None or elbow is None:
        return None
    if (
        min(float(wrist.get("score") or 0), float(elbow.get("score") or 0))
        < minimum_score
    ):
        return None
    left, top, right, bottom = (float(value) for value in crop_box)
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    wrist_xy = np.asarray(
        [float(wrist["x"]) - left, float(wrist["y"]) - top],
        dtype=np.float64,
    )
    elbow_xy = np.asarray(
        [float(elbow["x"]) - left, float(elbow["y"]) - top],
        dtype=np.float64,
    )
    outward = wrist_xy - elbow_xy
    arm_length = float(np.linalg.norm(outward))
    if arm_length < 2.0:
        return None
    hand = wrist_xy + 0.22 * outward
    forearm_fraction = MORPHOLOGY_FOREARM_FRACTIONS[morphology]
    forearm = wrist_xy + forearm_fraction * (elbow_xy - wrist_xy)
    positive = [hand, wrist_xy]
    if morphology != "minimal":
        positive.append(forearm)
    negative_forearm_fraction = min(0.95, forearm_fraction + 0.25)
    negative_forearm = wrist_xy + negative_forearm_fraction * (elbow_xy - wrist_xy)
    radius = max(4.0, arm_length * (0.18 if morphology == "minimal" else 0.24))
    endpoints = np.stack([hand, forearm])
    prompt_box = [
        float(np.clip(endpoints[:, 0].min() - radius, 0, width - 1)),
        float(np.clip(endpoints[:, 1].min() - radius, 0, height - 1)),
        float(np.clip(endpoints[:, 0].max() + radius, 1, width)),
        float(np.clip(endpoints[:, 1].max() + radius, 1, height)),
    ]
    corners = np.asarray(
        [
            [1.0, 1.0],
            [width - 2.0, 1.0],
            [1.0, height - 2.0],
            [width - 2.0, height - 2.0],
        ]
    )
    distances = np.linalg.norm(corners - wrist_xy, axis=1)
    negative = [negative_forearm, corners[int(distances.argmax())]]
    points = [
        [
            float(np.clip(point[0], 0, width - 1)),
            float(np.clip(point[1], 0, height - 1)),
        ]
        for point in [*positive, *negative]
    ]
    return {
        "input_points": points,
        "input_labels": [1] * len(positive) + [0] * len(negative),
        "input_box": prompt_box,
        "crop_size": [int(round(width)), int(round(height))],
        "pose_arm_length": arm_length,
    }


def mask_prompt_metrics(
    mask: np.ndarray,
    points: Sequence[Sequence[float]],
    labels: Sequence[int],
    prompt_box: Sequence[float],
) -> dict[str, float]:
    height, width = mask.shape
    samples = []
    for point in points:
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        samples.append(bool(mask[y, x]))
    positives = [
        value for value, label in zip(samples, labels, strict=True) if label == 1
    ]
    negatives = [
        value for value, label in zip(samples, labels, strict=True) if label == 0
    ]
    left, top, right, bottom = (
        int(np.clip(round(float(value)), 0, width if index % 2 == 0 else height))
        for index, value in enumerate(prompt_box)
    )
    inside = np.zeros_like(mask, dtype=bool)
    inside[max(0, top) : min(height, bottom), max(0, left) : min(width, right)] = True
    area = int(mask.sum())
    outside_fraction = float((mask & ~inside).sum() / max(area, 1))
    perimeter = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    return {
        "mask_fraction": float(mask.mean()),
        "positive_coverage": float(np.mean(positives)) if positives else 0.0,
        "negative_leak": float(np.mean(negatives)) if negatives else 0.0,
        "outside_box_fraction": outside_fraction,
        "edge_touch_fraction": float(perimeter.mean()),
    }


def select_pose_prompt_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    points: Sequence[Sequence[float]],
    labels: Sequence[int],
    prompt_box: Sequence[float],
) -> tuple[int, list[dict[str, float]]]:
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError("SAM masks and scores must have matching candidate axes")
    diagnostics = []
    for mask, predicted_iou in zip(masks, scores, strict=True):
        metrics = mask_prompt_metrics(mask, points, labels, prompt_box)
        area = metrics["mask_fraction"]
        area_penalty = max(0.01 - area, 0.0) * 20.0 + max(area - 0.75, 0.0) * 3.0
        selection_score = (
            float(predicted_iou)
            + 0.45 * metrics["positive_coverage"]
            - 0.75 * metrics["negative_leak"]
            - 0.55 * metrics["outside_box_fraction"]
            - 0.20 * metrics["edge_touch_fraction"]
            - area_penalty
        )
        diagnostics.append(
            {
                "sam_iou": float(predicted_iou),
                "selection_score": selection_score,
                **metrics,
            }
        )
    selected = max(
        range(len(diagnostics)), key=lambda index: diagnostics[index]["selection_score"]
    )
    return selected, diagnostics


def run_pose_prompt_sam(
    records: Sequence[Mapping[str, Any]],
    *,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    background: int,
) -> list[dict[str, Any]]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = []
        for row in batch:
            with Image.open(str(row["path"])) as source:
                images.append(source.convert("RGB"))
        try:
            inputs = processor(
                images=images,
                input_boxes=[[row["input_box"]] for row in batch],
                input_points=[[row["input_points"]] for row in batch],
                input_labels=[[row["input_labels"]] for row in batch],
                return_tensors="pt",
            )
            original_sizes = inputs["original_sizes"].cpu()
            model_inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if key != "original_sizes"
            }
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ),
            ):
                model_output = model(**model_inputs, multimask_output=True)
            masks_by_image = processor.post_process_masks(
                model_output.pred_masks.float().cpu(), original_sizes
            )
            scores = model_output.iou_scores.float().cpu().numpy()
            for row, image, mask_tensor, image_scores in zip(
                batch, images, masks_by_image, scores, strict=True
            ):
                masks = mask_tensor[0].numpy().astype(bool, copy=False)
                selected, diagnostics = select_pose_prompt_mask(
                    masks,
                    image_scores[0],
                    row["input_points"],
                    row["input_labels"],
                    row["input_box"],
                )
                target_dir = (
                    output_dir / "masks" / str(row["example_id"]).replace(":", "-")
                )
                target_dir.mkdir(parents=True, exist_ok=True)
                stem = str(row["crop_name"])
                mask_path = target_dir / f"{stem}-mask.png"
                composite_path = target_dir / f"{stem}-masked.jpg"
                Image.fromarray(masks[selected].astype(np.uint8) * 255).save(mask_path)
                composite_mask(image, masks[selected], background).save(
                    composite_path, quality=94
                )
                output.append(
                    {
                        "example_id": str(row["example_id"]),
                        "crop_name": stem,
                        "source_path": portable_path(row["path"]),
                        "mask_path": portable_path(mask_path),
                        "masked_path": portable_path(composite_path),
                        "morphology": str(row["morphology"]),
                        "selected_mask": selected,
                        "selected_metrics": diagnostics[selected],
                        "candidate_metrics": diagnostics,
                        "input_points": row["input_points"],
                        "input_labels": row["input_labels"],
                        "input_box": row["input_box"],
                        "pose_arm_length": row["pose_arm_length"],
                    }
                )
        finally:
            for image in images:
                image.close()
        print(
            f"sam2={min(start + batch_size, len(records))}/{len(records)}", flush=True
        )
    return output


def composite_mask(
    image: Image.Image, mask: np.ndarray, background: int
) -> Image.Image:
    foreground = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    backdrop = Image.new("RGB", image.size, (background,) * 3)
    return Image.composite(image.convert("RGB"), backdrop, foreground)


def prompt_overlay(
    image: Image.Image,
    mask: np.ndarray,
    points: Sequence[Sequence[float]],
    labels: Sequence[int],
    prompt_box: Sequence[float],
) -> Image.Image:
    rgb = image.convert("RGBA")
    tint = Image.new("RGBA", rgb.size, (255, 40, 40, 0))
    alpha = Image.fromarray(mask.astype(np.uint8) * 100, mode="L")
    tint.putalpha(alpha)
    rgb = Image.alpha_composite(rgb, tint)
    draw = ImageDraw.Draw(rgb)
    draw.rectangle(tuple(prompt_box), outline=(30, 140, 255, 255), width=2)
    for point, label in zip(points, labels, strict=True):
        x, y = (float(value) for value in point)
        color = (30, 220, 60, 255) if label == 1 else (255, 210, 0, 255)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=(0, 0, 0, 255))
    return rgb.convert("RGB")


def save_example_qa(
    example: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    mask_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    canonical_reference_path: Path,
    output_path: Path,
) -> None:
    example_id = str(example["example_id"])
    by_name = {str(row.get("name")): row for row in records}
    rows = []
    with Image.open(canonical_reference_path) as source:
        canonical = source.convert("RGB")
    for crop_name in PRIMARY_CROPS:
        row = mask_by_key[(example_id, crop_name)]
        with Image.open(str(by_name[crop_name]["path"])) as source:
            raw = source.convert("RGB")
        mask_path = resolve_project_path(row["mask_path"])
        masked_path = resolve_project_path(row["masked_path"])
        if mask_path is None or masked_path is None:
            raise ValueError("Mask output path cannot be resolved")
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert("L")) > 0
        with Image.open(masked_path) as source:
            masked = source.convert("RGB")
        overlay = prompt_overlay(
            raw,
            mask,
            row["input_points"],
            row["input_labels"],
            row["input_box"],
        )
        rows.append(
            [
                ("canonical", canonical.copy()),
                (f"{crop_name} raw", raw),
                (f"selected {row['selected_mask']} overlay", overlay),
                ("masked", masked),
            ]
        )
    canonical.close()
    cell = 220
    header = 44
    canvas = Image.new("RGB", (4 * cell, 2 * (cell + 22) + header), "white")
    draw = ImageDraw.Draw(canvas)
    label = (example.get("core_slot_labels") or {}).get("HANDS") or {}
    draw.text(
        (6, 5),
        f"{example_id} | {label.get('item_name')} | {hands_morphology_class(label.get('item_name'))}",
        fill="black",
    )
    for row_index, panels in enumerate(rows):
        for column, (name, panel) in enumerate(panels):
            thumb = ImageOps.contain(
                panel,
                (cell - 8, cell - 8),
                Image.Resampling.LANCZOS,
            )
            x = column * cell + (cell - thumb.width) // 2
            y = header + row_index * (cell + 22)
            canvas.paste(thumb, (x, y))
            draw.text((column * cell + 4, y + cell), name, fill="black")
            thumb.close()
            panel.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    canvas.close()


def summarize_audit(
    queue: Sequence[Mapping[str, Any]],
    *,
    fold_role: str,
    fold_manifest: Path,
    eligibility_manifest: Path,
    sam_model_id: str,
    index_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    crops = [crop for row in queue for crop in row["crops"].values()]
    metrics = [crop["selected_metrics"] for crop in crops]
    return {
        "version": 1,
        "status": "awaiting_visual_review",
        "fold_role": fold_role,
        "example_count": len(queue),
        "crop_count": len(crops),
        "morphology_counts": {
            morphology: sum(row["morphology"] == morphology for row in queue)
            for morphology in HANDS_MORPHOLOGY_CLASSES
        },
        "gender_counts": {
            gender: sum(row["model_gender"] == gender for row in queue)
            for gender in sorted({str(row["model_gender"]) for row in queue})
        },
        "sam_model_id": sam_model_id,
        "index_dir": portable_path(index_dir),
        "selected_metric_means": {
            key: float(np.mean([float(row[key]) for row in metrics]))
            for key in (
                "sam_iou",
                "mask_fraction",
                "positive_coverage",
                "negative_leak",
                "outside_box_fraction",
                "edge_touch_fraction",
            )
        },
        "fold_manifest": portable_path(fold_manifest),
        "slot_eligibility_manifest": portable_path(eligibility_manifest),
        "review_queue": portable_path(output_dir / "review_queue.jsonl"),
        "qa_dir": portable_path(output_dir / "qa"),
        "output_dir": portable_path(output_dir),
        "interpretation_limit": (
            "SAM IoU is model-predicted and is not ground-truth glove IoU. "
            "The masks cannot become training targets until visual review "
            "confirms glove coverage and rejects sleeve, weapon, or body masks."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
