from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import CACHE_DIR, portable_path
from ffxiv_prism.vision.human_parsing import HumanParser
from scripts.build_slot_aligned_candidate_index import crop_variants_for_slot
from scripts.evaluate_glamour_embedding_model import DINO_MEAN, DINO_STD

DEFAULT_INPUT = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_OUTPUT = Path("data/indexes/facebook-dinov2-base-equipment-only")

EQUIPMENT_LABEL_IDS = {
    "HEAD": frozenset((1, 3, 17)),
    "BODY": frozenset((4, 7, 8, 17)),
    # Armoured gloves and long sleeves are frequently parsed as upper-clothes
    # or dresses rather than bare left/right arms.  The geometric one-arm crop
    # already excludes most of the opposite side, so retaining these clothing
    # labels preserves the equipment without reintroducing the mannequin.
    "HANDS": frozenset((4, 7, 8, 14, 15)),
    "LEGS": frozenset((5, 6, 7, 8)),
    "FEET": frozenset((9, 10)),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical candidate index that removes background, face, "
            "skin and unrelated mannequin regions with human parsing."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parse-batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--qa-limit", type=int, default=50)
    parser.add_argument("--background", type=int, default=127)
    parser.add_argument(
        "--mask-mode",
        choices=("parser", "template-diff", "hybrid"),
        default="parser",
        help=(
            "Human parsing, standardized-render template subtraction, or "
            "their union."
        ),
    )
    parser.add_argument("--template-samples", type=int, default=128)
    parser.add_argument("--template-diff-threshold", type=float, default=24.0)
    parser.add_argument(
        "--masked-slots",
        nargs="+",
        choices=tuple(EQUIPMENT_LABEL_IDS),
        default=tuple(EQUIPMENT_LABEL_IDS),
        help=(
            "Slots that receive the selected mask strategy. Other slots keep "
            "their original aligned crop."
        ),
    )
    parser.add_argument("--limit", type=int, help="Debug only: source image limit.")
    args = parser.parse_args()

    all_metadata = read_jsonl(args.input_dir / "metadata.jsonl")
    templates = {}
    if args.mask_mode in {"template-diff", "hybrid"}:
        templates = build_median_templates(
            all_metadata,
            sample_count=args.template_samples,
        )
    metadata = all_metadata
    if args.limit is not None:
        metadata = metadata[: args.limit]
    device = torch.device(args.device)
    parser_model = (
        HumanParser(device=args.device)
        if args.mask_mode in {"parser", "hybrid"}
        else None
    )
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device).eval()
    output_rows = []
    output_embeddings = []
    qa_dir = args.output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_written = 0

    for start in range(0, len(metadata), args.parse_batch_size):
        source_rows = metadata[start : start + args.parse_batch_size]
        images = []
        for row in source_rows:
            with Image.open(str(row["local_path"])) as source:
                images.append(source.convert("RGB"))
        segmentations = [None] * len(images)
        if parser_model is not None:
            parse_indexes = [
                index
                for index, row in enumerate(source_rows)
                if str(row["slot"]) in args.masked_slots
            ]
            parsed = parser_model.parse([images[index] for index in parse_indexes])
            for index, segmentation in zip(parse_indexes, parsed, strict=True):
                segmentations[index] = segmentation
        candidate_images = []
        candidate_rows = []
        for row, image, segmentation in zip(
            source_rows,
            images,
            segmentations,
            strict=True,
        ):
            slot = str(row["slot"])
            for crop_name, crop_box in crop_variants_for_slot(slot):
                if slot not in args.masked_slots:
                    equipment, stats = build_original_crop(
                        image,
                        normalized_box=crop_box,
                    )
                elif args.mask_mode == "parser":
                    equipment, stats = build_equipment_only_crop(
                        image,
                        segmentation,
                        slot=slot,
                        normalized_box=crop_box,
                        background=args.background,
                    )
                else:
                    template = templates[template_group_key(row)]
                    equipment, stats = build_template_difference_crop(
                        image,
                        template,
                        normalized_box=crop_box,
                        background=args.background,
                        diff_threshold=args.template_diff_threshold,
                        parser_segmentation=(
                            segmentation if args.mask_mode == "hybrid" else None
                        ),
                        slot=slot,
                    )
                output_row = dict(row)
                output_row["source_image_row_id"] = row.get("image_row_id")
                output_row["image_row_id"] = len(output_rows) + len(candidate_rows)
                output_row["candidate_crop"] = crop_name
                output_row["candidate_crop_box_normalized"] = list(crop_box)
                output_row["equipment_only"] = True
                output_row["equipment_mask_stats"] = stats
                candidate_rows.append(output_row)
                candidate_images.append(equipment)
                if qa_written < args.qa_limit:
                    qa_path = qa_dir / (
                        f"{qa_written:04d}-{slot.lower()}-{crop_name}.jpg"
                    )
                    equipment.save(qa_path, quality=92)
                    qa_written += 1
            image.close()

        for encode_start in range(
            0,
            len(candidate_images),
            args.encode_batch_size,
        ):
            batch_images = candidate_images[
                encode_start : encode_start + args.encode_batch_size
            ]
            pixels = preprocess_images(batch_images).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                tokens = encoder(pixel_values=pixels).last_hidden_state
                embeddings = torch.nn.functional.normalize(
                    tokens[:, 0],
                    dim=-1,
                )
            output_embeddings.append(embeddings.float().cpu().numpy())
        for image in candidate_images:
            image.close()
        output_rows.extend(candidate_rows)
        if (start // args.parse_batch_size + 1) % 50 == 0:
            print(
                f"parsed={min(start + args.parse_batch_size, len(metadata))}"
                f"/{len(metadata)} candidates={len(output_rows)}",
                flush=True,
            )

    matrix = np.concatenate(output_embeddings, axis=0).astype(
        np.float32,
        copy=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", matrix)
    write_jsonl(args.output_dir / "metadata.jsonl", output_rows)
    fallback_count = sum(
        bool(row["equipment_mask_stats"]["used_fallback"])
        for row in output_rows
    )
    write_json(
        args.output_dir / "summary.json",
        {
            "source_index": portable_path(args.input_dir),
            "model_id": args.model_id,
            "source_image_count": len(metadata),
            "candidate_count": len(output_rows),
            "fallback_count": fallback_count,
            "fallback_fraction": fallback_count / len(output_rows),
            "equipment_label_ids": {
                slot: sorted(ids)
                for slot, ids in EQUIPMENT_LABEL_IDS.items()
            },
            "mask_mode": args.mask_mode,
            "masked_slots": sorted(args.masked_slots),
            "template_samples": (
                args.template_samples if templates else None
            ),
            "template_diff_threshold": (
                args.template_diff_threshold if templates else None
            ),
            "background": args.background,
            "output": portable_path(args.output_dir),
        },
    )
    print(
        f"source={len(metadata)} candidates={len(output_rows)} "
        f"fallback={fallback_count} output={args.output_dir}"
    )
    return 0


def build_equipment_only_crop(
    image: Image.Image,
    segmentation: np.ndarray,
    *,
    slot: str,
    normalized_box: tuple[float, float, float, float],
    background: int = 127,
    min_selected_fraction: float = 0.002,
) -> tuple[Image.Image, dict[str, Any]]:
    if segmentation.shape != (image.height, image.width):
        raise ValueError("Segmentation shape must match source image")
    if slot not in EQUIPMENT_LABEL_IDS:
        raise ValueError(f"Unsupported slot: {slot}")
    box = normalized_crop_box(image.size, normalized_box)
    crop = image.crop(box)
    left, top, right, bottom = box
    segmentation_crop = segmentation[top:bottom, left:right]
    selected = np.isin(
        segmentation_crop,
        tuple(EQUIPMENT_LABEL_IDS[slot]),
    )
    selected_fraction = float(selected.mean()) if selected.size else 0.0
    used_fallback = selected_fraction < min_selected_fraction
    if used_fallback:
        return crop, {
            "selected_fraction": selected_fraction,
            "used_fallback": True,
            "mask_source": "parser",
        }
    mask = Image.fromarray(selected.astype(np.uint8) * 255, mode="L")
    mask = mask.filter(ImageFilter.MaxFilter(7))
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    canvas = Image.new(
        "RGB",
        crop.size,
        (background, background, background),
    )
    output = Image.composite(crop, canvas, mask)
    crop.close()
    canvas.close()
    mask.close()
    return output, {
        "selected_fraction": selected_fraction,
        "used_fallback": False,
        "mask_source": "parser",
    }


def build_original_crop(
    image: Image.Image,
    *,
    normalized_box: tuple[float, float, float, float],
) -> tuple[Image.Image, dict[str, Any]]:
    return image.crop(normalized_crop_box(image.size, normalized_box)), {
        "selected_fraction": 1.0,
        "used_fallback": False,
        "mask_source": "original",
    }


def template_group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["slot"]),
        str(row.get("race") or ""),
        str(row.get("model_gender") or ""),
        str(row.get("dye_state") or ""),
    )


def build_median_templates(
    metadata: Sequence[dict[str, Any]],
    *,
    sample_count: int,
) -> dict[tuple[str, str, str, str], Image.Image]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    grouped: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in metadata:
        grouped[template_group_key(row)].append(row)
    templates = {}
    for key, rows in grouped.items():
        indexes = np.linspace(
            0,
            len(rows) - 1,
            min(sample_count, len(rows)),
            dtype=np.int64,
        )
        arrays = []
        expected_size = None
        for index in indexes:
            with Image.open(str(rows[int(index)]["local_path"])) as source:
                image = source.convert("RGB")
            if expected_size is None:
                expected_size = image.size
            if image.size != expected_size:
                image.close()
                raise ValueError(f"Inconsistent render size in group {key}")
            arrays.append(np.asarray(image, dtype=np.uint8))
            image.close()
        median = np.median(np.stack(arrays, axis=0), axis=0).astype(np.uint8)
        templates[key] = Image.fromarray(median, mode="RGB")
        print(f"template={key} samples={len(arrays)}", flush=True)
    return templates


def build_template_difference_crop(
    image: Image.Image,
    template: Image.Image,
    *,
    normalized_box: tuple[float, float, float, float],
    background: int = 127,
    diff_threshold: float = 24.0,
    min_selected_fraction: float = 0.002,
    parser_segmentation: np.ndarray | None = None,
    slot: str | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    if image.size != template.size:
        raise ValueError("Template size must match source image")
    image_array = np.asarray(image, dtype=np.int16)
    template_array = np.asarray(template, dtype=np.int16)
    difference = np.max(np.abs(image_array - template_array), axis=2)
    selected = difference >= diff_threshold
    if parser_segmentation is not None:
        if parser_segmentation.shape != (image.height, image.width):
            raise ValueError("Segmentation shape must match source image")
        if slot not in EQUIPMENT_LABEL_IDS:
            raise ValueError(f"Unsupported slot: {slot}")
        selected |= np.isin(
            parser_segmentation,
            tuple(EQUIPMENT_LABEL_IDS[slot]),
        )

    box = normalized_crop_box(image.size, normalized_box)
    left, top, right, bottom = box
    crop = image.crop(box)
    selected_crop = selected[top:bottom, left:right]
    selected_fraction = (
        float(selected_crop.mean()) if selected_crop.size else 0.0
    )
    used_fallback = selected_fraction < min_selected_fraction
    if used_fallback:
        return crop, {
            "selected_fraction": selected_fraction,
            "used_fallback": True,
            "mask_source": "template-diff",
        }
    mask = Image.fromarray(selected_crop.astype(np.uint8) * 255, mode="L")
    mask = mask.filter(ImageFilter.MedianFilter(3))
    mask = mask.filter(ImageFilter.MaxFilter(9))
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    canvas = Image.new(
        "RGB",
        crop.size,
        (background, background, background),
    )
    output = Image.composite(crop, canvas, mask)
    crop.close()
    canvas.close()
    mask.close()
    return output, {
        "selected_fraction": selected_fraction,
        "used_fallback": False,
        "mask_source": (
            "template-diff+parser"
            if parser_segmentation is not None
            else "template-diff"
        ),
    }


def normalized_crop_box(
    size: tuple[int, int],
    normalized_box: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    width, height = size
    left, top, right, bottom = normalized_box
    return (
        max(0, min(width - 1, int(round(left * width)))),
        max(0, min(height - 1, int(round(top * height)))),
        max(1, min(width, int(round(right * width)))),
        max(1, min(height, int(round(bottom * height)))),
    )


def preprocess_images(images: Sequence[Image.Image]) -> torch.Tensor:
    arrays = []
    for image in images:
        resized = image.resize((224, 224), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        resized.close()
        arrays.append(
            np.transpose(
                (array - DINO_MEAN) / DINO_STD,
                (2, 0, 1),
            )
        )
    return torch.from_numpy(np.stack(arrays))


if __name__ == "__main__":
    raise SystemExit(main())
