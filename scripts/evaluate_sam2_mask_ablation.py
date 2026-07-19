from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import CACHE_DIR, PROCESSED_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.rank_fusion import (
    appearance_union_recall,
    fuse_case_results_rrf,
)
from scripts.evaluate_glamour_embedding_model import DINO_MEAN, DINO_STD, search_crops
from scripts.evaluate_glamour_matching import (
    ROLE_PROFILES,
    SLOTS,
    TAG_PENALTY_PROFILES,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    collect_crop_records,
    evaluate_config,
    summarize,
)

DEFAULT_EXAMPLES = PROCESSED_DIR / "glamour_examples.jsonl"
DEFAULT_QUALITY = PROCESSED_DIR / "front_fullbody_quality_final_v1.jsonl"
DEFAULT_REGIONS = Path("data/debug/outfit-regions/overlap-final-201")
DEFAULT_METADATA = Path("data/indexes/clip-vit-base-patch32/metadata.jsonl")
DEFAULT_CANDIDATES = Path("data/indexes/dinov2-small/embeddings.npy")
DEFAULT_RAW_CROPS = PROCESSED_DIR / "glamour_crop_embeddings_dinov2_small.npz"
DEFAULT_OUTPUT_DIR = Path("data/debug/sam2-mask-ablation")
DEFAULT_REPORT = PROCESSED_DIR / "sam2_mask_ablation_dinov2_small.json"

ROLE_ORDER = {"primary": 0, "auxiliary": 1, "context": 2}
AREA_RANGES = {
    "HEAD": (0.04, 0.85),
    "BODY": (0.12, 0.90),
    "HANDS": (0.02, 0.75),
    "LEGS": (0.06, 0.85),
    "FEET": (0.02, 0.75),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw DINOv2 crop retrieval with SAM2-masked retrieval on "
            "the same high-quality glamour examples."
        )
    )
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--quality-manifest", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--candidate-embeddings", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--raw-crop-cache", type=Path, default=DEFAULT_RAW_CROPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sam-model-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--dino-model-id", default="facebook/dinov2-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sam-batch-size", type=int, default=2)
    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--limit-examples", type=int, default=5)
    parser.add_argument("--max-crops-per-slot", type=int, default=2)
    parser.add_argument("--slots", nargs="+", choices=SLOTS, default=list(SLOTS))
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--background", type=int, default=127)
    parser.add_argument("--rrf-primary-weight", type=float, default=0.6)
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    args.device = str(device)
    examples, crop_records = select_experiment_examples(
        read_jsonl(args.examples),
        args.quality_manifest,
        args.regions_dir,
        set(args.slots),
        args.limit_examples,
        args.max_crops_per_slot,
    )
    if not examples:
        raise ValueError("No local high-quality examples with region crops were found")

    raw_embeddings = load_embedding_cache(args.raw_crop_cache)
    crop_records = retain_cached_records(crop_records, raw_embeddings)
    work = [
        record
        for records in crop_records.values()
        for record in records
    ]
    if not work:
        raise ValueError("Selected examples have no crops in the raw embedding cache")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sam_model, sam_processor = load_sam2(args.sam_model_id, device)
    mask_rows, masked_paths = build_sam2_crops(
        work,
        sam_model,
        sam_processor,
        device,
        args.output_dir,
        args.sam_batch_size,
        args.background,
    )
    del sam_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dino_model = load_dino(args.dino_model_id, device)
    masked_matrix = encode_dino_paths(
        dino_model,
        masked_paths,
        device,
        args.dino_batch_size,
    )
    masked_embeddings = {
        str(record["path"]): embedding
        for record, embedding in zip(work, masked_matrix, strict=True)
    }

    metadata = read_jsonl(args.metadata)
    candidate_embeddings = np.load(args.candidate_embeddings, mmap_mode="r")
    if len(metadata) != len(candidate_embeddings):
        raise ValueError("Candidate metadata and embedding counts differ")

    common_kwargs = {
        "metadata": metadata,
        "candidate_embeddings": candidate_embeddings,
        "crop_records": crop_records,
        "candidate_k": args.candidate_k,
        "device": args.device,
    }
    raw_search = search_crops(
        crop_embeddings=raw_embeddings,
        **common_kwargs,
    )
    masked_search = search_crops(
        crop_embeddings=masked_embeddings,
        **common_kwargs,
    )
    raw_summary, raw_cases = evaluate(
        examples,
        crop_records,
        raw_search,
        metadata,
        args.top_k,
    )
    masked_summary, masked_cases = evaluate(
        examples,
        crop_records,
        masked_search,
        metadata,
        args.top_k,
    )
    fused_cases = fuse_case_results_rrf(
        raw_cases,
        masked_cases,
        primary_weight=args.rrf_primary_weight,
        top_k=args.top_k,
    )
    fused_summary = summarize(fused_cases, args.top_k)
    payload = {
        "status": "pilot",
        "sam_model_id": args.sam_model_id,
        "dino_model_id": args.dino_model_id,
        "device": args.device,
        "example_count": len(examples),
        "example_ids": [str(example["example_id"]) for example in examples],
        "crop_count": len(work),
        "slots": list(args.slots),
        "max_crops_per_slot": args.max_crops_per_slot,
        "raw": raw_summary,
        "sam2_masked": masked_summary,
        "raw_sam2_rrf": fused_summary,
        "rrf_primary_weight": args.rrf_primary_weight,
        "appearance_union_recall_at_k": appearance_union_recall(raw_cases, masked_cases),
        "delta": metric_delta(raw_summary, masked_summary),
        "mask_area_by_slot": summarize_mask_areas(mask_rows),
        "mask_rows": mask_rows,
        "raw_cases": raw_cases,
        "sam2_cases": masked_cases,
        "output_dir": portable_path(args.output_dir),
    }
    write_json(args.report, payload)
    print_comparison(raw_summary, masked_summary, args.top_k, fused_summary)
    print(f"report={args.report} crops={len(work)} examples={len(examples)}")
    return 0


def select_experiment_examples(
    examples: list[dict[str, Any]],
    quality_manifest: Path,
    regions_dir: Path,
    slots: set[str],
    limit_examples: int,
    max_crops_per_slot: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    passed_ids: set[str] | None = None
    if quality_manifest.is_file():
        passed_ids = {
            str(row["example_id"])
            for row in read_jsonl(quality_manifest)
            if row.get("passed")
        }
    local_examples = [
        example
        for example in examples
        if Path(str(example["local_path"])).is_file()
        and (passed_ids is None or str(example["example_id"]) in passed_ids)
    ]
    records = collect_crop_records(local_examples, regions_dir)
    selected_examples = []
    selected_records: dict[str, list[dict[str, Any]]] = {}
    for example in local_examples:
        example_id = str(example["example_id"])
        rows = [
            row
            for row in records.get(example_id, [])
            if row["slot"] in slots
        ]
        if not rows:
            continue
        by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_slot[row["slot"]].append(row)
        chosen = []
        for slot in SLOTS:
            ranked = sorted(
                by_slot.get(slot, []),
                key=lambda row: (
                    ROLE_ORDER.get(row.get("role", "primary"), 9),
                    -float(row.get("confidence", 0.0)),
                    row["name"],
                ),
            )
            chosen.extend(ranked[:max_crops_per_slot])
        if chosen:
            selected_examples.append(example)
            selected_records[example_id] = chosen
        if len(selected_examples) >= limit_examples:
            break
    return selected_examples, selected_records


def retain_cached_records(
    records: dict[str, list[dict[str, Any]]],
    embeddings: dict[str, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    return {
        example_id: [
            record
            for record in rows
            if str(record["path"]) in embeddings
        ]
        for example_id, rows in records.items()
    }


def load_embedding_cache(path: Path) -> dict[str, np.ndarray]:
    cache = np.load(path, allow_pickle=False)
    return {
        str(crop_path): embedding
        for crop_path, embedding in zip(
            cache["paths"].tolist(),
            cache["embeddings"],
            strict=True,
        )
    }


def load_sam2(model_id: str, device: torch.device) -> tuple[torch.nn.Module, Any]:
    from transformers import Sam2Model, Sam2Processor

    cache_dir = CACHE_DIR / "huggingface" / "hub"
    processor = Sam2Processor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model = Sam2Model.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
    ).to(device).eval()
    return model, processor


def load_dino(model_id: str, device: torch.device) -> torch.nn.Module:
    from transformers import AutoModel

    cache_dir = CACHE_DIR / "huggingface" / "hub"
    try:
        model = AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except OSError:
        model = AutoModel.from_pretrained(
            model_id,
            local_files_only=True,
        )
    return model.to(device).eval()


def build_sam2_crops(
    records: list[dict[str, Any]],
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    background: int,
    save_masks: bool = True,
) -> tuple[list[dict[str, Any]], list[Path]]:
    mask_rows = []
    masked_paths = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = [load_rgb(Path(record["path"])) for record in batch]
        boxes = [[inset_prompt_box(image.size)] for image in images]
        try:
            inputs = processor(
                images=images,
                input_boxes=boxes,
                return_tensors="pt",
            )
            original_sizes = inputs["original_sizes"].cpu()
            model_inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if key != "original_sizes"
            }
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), amp:
                outputs = model(**model_inputs, multimask_output=True)
            masks_by_image = processor.post_process_masks(
                outputs.pred_masks.float().cpu(),
                original_sizes,
            )
            scores = outputs.iou_scores.float().cpu().numpy()
            for record, image, masks, image_scores in zip(
                batch,
                images,
                masks_by_image,
                scores,
                strict=True,
            ):
                masks_array = masks[0].numpy().astype(bool, copy=False)
                score_array = image_scores[0]
                selected = select_sam_mask(
                    masks_array,
                    score_array,
                    record["slot"],
                )
                mask = masks_array[selected]
                configured_output = record.get("masked_output_path")
                if configured_output is None:
                    target_dir = output_dir / str(record["example_id"]).replace(":", "-")
                    output_path = target_dir / f"{Path(record['path']).stem}-sam2.jpg"
                else:
                    output_path = Path(str(configured_output))
                    target_dir = output_path.parent
                target_dir.mkdir(parents=True, exist_ok=True)
                mask_path = (
                    target_dir / f"{Path(record['path']).stem}-mask.png"
                    if save_masks
                    else None
                )
                composite_mask(image, mask, background).save(
                    output_path,
                    quality=94,
                )
                if mask_path is not None:
                    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                masked_paths.append(output_path)
                mask_rows.append(
                    {
                        "example_id": str(record["example_id"]),
                        "slot": record["slot"],
                        "crop_name": record["name"],
                        "source_path": portable_path(record["path"]),
                        "masked_path": portable_path(output_path),
                        "mask_path": portable_path(mask_path),
                        "selected_mask": selected,
                        "sam_iou": float(score_array[selected]),
                        "mask_fraction": float(mask.mean()),
                    }
                )
        finally:
            for image in images:
                image.close()
        done = min(start + batch_size, len(records))
        progress_interval = max(batch_size * 20, batch_size)
        if done == len(records) or done % progress_interval == 0:
            print(f"sam2={done}/{len(records)}", flush=True)
    return mask_rows, masked_paths


def inset_prompt_box(size: tuple[int, int], inset_fraction: float = 0.04) -> list[float]:
    width, height = size
    inset_x = max(1.0, width * inset_fraction)
    inset_y = max(1.0, height * inset_fraction)
    return [
        inset_x,
        inset_y,
        max(inset_x + 1.0, width - inset_x),
        max(inset_y + 1.0, height - inset_y),
    ]


def select_sam_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    slot: str,
) -> int:
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError("SAM masks and scores must have matching candidate axes")
    minimum, maximum = AREA_RANGES[slot]
    best_index = 0
    best_score = -float("inf")
    for index, (mask, predicted_iou) in enumerate(zip(masks, scores, strict=True)):
        fraction = float(mask.mean())
        distance = max(minimum - fraction, 0.0) + max(fraction - maximum, 0.0)
        empty_penalty = 2.0 if fraction <= 0.001 else 0.0
        selection_score = float(predicted_iou) - 1.5 * distance - empty_penalty
        if selection_score > best_score:
            best_index = index
            best_score = selection_score
    return best_index


def composite_mask(image: Image.Image, mask: np.ndarray, background: int) -> Image.Image:
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_image = mask_image.filter(ImageFilter.MaxFilter(5))
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(1.2))
    backdrop = Image.new("RGB", image.size, (background,) * 3)
    return Image.composite(image.convert("RGB"), backdrop, mask_image)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


@torch.inference_mode()
def encode_dino_paths(
    model: torch.nn.Module,
    paths: Sequence[Path],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(paths), batch_size):
        arrays = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as source:
                image = source.convert("RGB").resize(
                    (224, 224),
                    Image.Resampling.BICUBIC,
                )
            array = np.asarray(image, dtype=np.float32) / 255.0
            arrays.append(np.transpose((array - DINO_MEAN) / DINO_STD, (2, 0, 1)))
        pixels = torch.from_numpy(np.stack(arrays)).to(device)
        features = model(pixel_values=pixels).last_hidden_state[:, 0]
        features = torch.nn.functional.normalize(features, dim=-1)
        outputs.append(features.float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def evaluate(
    examples: list[dict[str, Any]],
    crop_records: dict[str, list[dict[str, Any]]],
    search_results: dict[str, list[dict[str, Any]]],
    metadata: list[dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outcomes = evaluate_config(
        examples,
        crop_records,
        search_results,
        available_pieces_by_slot(metadata),
        available_appearance_keys_by_slot(metadata),
        top_k,
        ROLE_PROFILES["overlap"],
        TAG_PENALTY_PROFILES["none"],
        "max",
    )
    return summarize(outcomes, top_k), outcomes


def metric_delta(raw: dict[str, Any], masked: dict[str, Any]) -> dict[str, float]:
    keys = (
        "appearance_recall_at_1",
        "appearance_recall_at_k",
        "appearance_mrr",
    )
    return {
        key: float(masked[key]) - float(raw[key])
        for key in keys
    }


def summarize_mask_areas(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_slot: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_slot[row["slot"]].append(float(row["mask_fraction"]))
    return {
        slot: {
            "count": len(values),
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
        for slot, values in sorted(by_slot.items())
    }


def print_comparison(
    raw: dict[str, Any],
    masked: dict[str, Any],
    top_k: int,
    fused: dict[str, Any] | None = None,
) -> None:
    print(
        "overall "
        f"raw_top1={raw['appearance_recall_at_1']:.3f} "
        f"sam2_top1={masked['appearance_recall_at_1']:.3f} "
        f"raw_r{top_k}={raw['appearance_recall_at_k']:.3f} "
        f"sam2_r{top_k}={masked['appearance_recall_at_k']:.3f}"
    )
    if fused is not None:
        print(
            "rrf "
            f"top1={fused['appearance_recall_at_1']:.3f} "
            f"r{top_k}={fused['appearance_recall_at_k']:.3f} "
            f"mrr={fused['appearance_mrr']:.3f}"
        )
    for slot in SLOTS:
        raw_slot = raw["by_slot"][slot]
        masked_slot = masked["by_slot"][slot]
        if not raw_slot["eligible_case_count"]:
            continue
        print(
            f"  {slot}: n={raw_slot['eligible_case_count']} "
            f"raw_top1={raw_slot['appearance_recall_at_1']:.3f} "
            f"sam2_top1={masked_slot['appearance_recall_at_1']:.3f} "
            f"raw_r{top_k}={raw_slot['appearance_recall_at_k']:.3f} "
            f"sam2_r{top_k}={masked_slot['appearance_recall_at_k']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
