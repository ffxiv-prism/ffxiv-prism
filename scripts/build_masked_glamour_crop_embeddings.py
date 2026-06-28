from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import CACHE_DIR, PROCESSED_DIR, portable_path
from ffxiv_prism.vision.human_parsing import (
    DEFAULT_HUMAN_PARSING_MODEL,
    HumanParser,
    MaskResult,
    apply_human_mask,
)
from scripts.evaluate_glamour_embedding_model import DINO_MEAN, DINO_STD
from scripts.evaluate_glamour_matching import collect_crop_records

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_MODEL_ID = "facebook/dinov2-base"
DEFAULT_SOURCE_CACHE = PROCESSED_DIR / "glamour_crop_embeddings_facebook-dinov2-base-color.npz"
DEFAULT_REGIONS = Path("data/debug/outfit-regions/overlap-final-201")
DEFAULT_QA_DIR = Path("data/debug/human-parsing")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build DINOv2 crop embeddings after human-parsing masks."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--parser-model-id", default=DEFAULT_HUMAN_PARSING_MODEL)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--examples", type=Path, default=PROCESSED_DIR / "glamour_examples.jsonl")
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--modes", default="person,slot")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    parser.add_argument("--qa-limit", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not modes or any(mode not in {"person", "slot"} for mode in modes):
        raise ValueError("Mask modes must be person and/or slot")

    source = np.load(args.source_cache, allow_pickle=False)
    paths = [str(path) for path in source["paths"].tolist()]
    if args.limit is not None:
        paths = paths[: args.limit]
    examples = read_jsonl(args.examples)
    crop_records = collect_crop_records(examples, args.regions_dir)
    path_slots = {
        str(record["path"]): record["slot"]
        for records in crop_records.values()
        for record in records
    }
    missing_slots = [path for path in paths if path not in path_slots]
    if missing_slots:
        raise ValueError(f"{len(missing_slots)} cached crop paths have no slot metadata")

    human_parser = HumanParser(args.parser_model_id, args.device)
    encoder = load_dino_encoder(args.model_id, args.device)
    embeddings_by_mode: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    stats = {
        mode: {
            "fallbacks": Counter(),
            "person_fractions": defaultdict(list),
            "selected_fractions": defaultdict(list),
        }
        for mode in modes
    }
    qa_written = Counter()

    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        images = [load_rgb(Path(path)) for path in batch_paths]
        try:
            segmentations = human_parser.parse(images)
            for mode in modes:
                masked_results = [
                    apply_human_mask(
                        image,
                        segmentation,
                        mode=mode,
                        slot=path_slots[path],
                    )
                    for image, segmentation, path in zip(
                        images,
                        segmentations,
                        batch_paths,
                        strict=True,
                    )
                ]
                embeddings_by_mode[mode].append(
                    encode_dino_images(encoder, [result.image for result in masked_results])
                )
                for path, image, result in zip(
                    batch_paths,
                    images,
                    masked_results,
                    strict=True,
                ):
                    slot = path_slots[path]
                    stats[mode]["person_fractions"][slot].append(result.person_fraction)
                    stats[mode]["selected_fractions"][slot].append(result.selected_fraction)
                    if result.used_fallback:
                        stats[mode]["fallbacks"][slot] += 1
                    if qa_written[mode] < args.qa_limit:
                        save_qa(
                            image,
                            result,
                            slot,
                            Path(path),
                            args.qa_dir / mode,
                        )
                        qa_written[mode] += 1
                for result in masked_results:
                    result.image.close()
                    result.mask.close()
        finally:
            for image in images:
                image.close()
        done = min(start + args.batch_size, len(paths))
        if done == len(paths) or done % (args.batch_size * 20) == 0:
            print(f"parsed_and_embedded={done}/{len(paths)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        output = args.output_dir / (
            f"glamour_crop_embeddings_facebook-dinov2-base-color-mask-{mode}.npz"
        )
        embeddings = np.concatenate(embeddings_by_mode[mode], axis=0)
        np.savez_compressed(
            output,
            paths=np.asarray(paths),
            embeddings=embeddings,
            model_id=np.asarray(args.model_id),
            preprocess=np.asarray("color"),
            mask_mode=np.asarray(mode),
            parser_model_id=np.asarray(args.parser_model_id),
        )
        summary = summarize_stats(
            mode,
            stats[mode],
            paths,
            path_slots,
            args,
            output,
        )
        summary_path = args.output_dir / f"glamour_mask_{mode}_summary.json"
        write_json(summary_path, summary)
        print(f"mode={mode} output={output} summary={summary_path}")
    return 0


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_dino_encoder(model_id: str, device: str) -> tuple[torch.nn.Module, torch.device]:
    from transformers import AutoModel

    torch_device = torch.device(device)
    model = AutoModel.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(torch_device).eval()
    return model, torch_device


@torch.inference_mode()
def encode_dino_images(
    encoder: tuple[torch.nn.Module, torch.device],
    images: Sequence[Image.Image],
) -> np.ndarray:
    model, device = encoder
    arrays = []
    for image in images:
        resized = image.resize((224, 224), Image.Resampling.BICUBIC)
        try:
            array = np.asarray(resized, dtype=np.float32) / 255.0
        finally:
            resized.close()
        arrays.append(np.transpose((array - DINO_MEAN) / DINO_STD, (2, 0, 1)))
    pixel_values = torch.from_numpy(np.stack(arrays)).to(device)
    outputs = model(pixel_values=pixel_values)
    features = torch.nn.functional.normalize(
        outputs.last_hidden_state[:, 0],
        dim=-1,
    )
    return features.cpu().numpy().astype(np.float32, copy=False)


def save_qa(
    original: Image.Image,
    result: MaskResult,
    slot: str,
    source_path: Path,
    output_dir: Path,
) -> None:
    width, height = original.size
    preview = Image.new("RGB", (width * 3, height + 28), "white")
    preview.paste(original, (0, 28))
    preview.paste(result.mask.convert("RGB"), (width, 28))
    preview.paste(result.image, (width * 2, 28))
    draw = ImageDraw.Draw(preview)
    draw.text(
        (4, 6),
        (
            f"{slot} person={result.person_fraction:.2f} "
            f"selected={result.selected_fraction:.2f} "
            f"fallback={result.used_fallback}"
        ),
        fill="black",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    preview.thumbnail((1500, 900), Image.Resampling.LANCZOS)
    preview.save(
        output_dir / f"{source_path.parent.name}-{source_path.stem}.jpg",
        quality=88,
    )
    preview.close()


def summarize_stats(
    mode: str,
    stats: dict,
    paths: list[str],
    path_slots: dict[str, str],
    args: argparse.Namespace,
    output: Path,
) -> dict:
    slots = sorted(set(path_slots[path] for path in paths))
    return {
        "mode": mode,
        "crop_count": len(paths),
        "model_id": args.model_id,
        "parser_model_id": args.parser_model_id,
        "source_cache": portable_path(args.source_cache),
        "output": portable_path(output),
        "by_slot": {
            slot: {
                "crop_count": sum(path_slots[path] == slot for path in paths),
                "fallback_count": int(stats["fallbacks"][slot]),
                "mean_person_fraction": mean(stats["person_fractions"][slot]),
                "mean_selected_fraction": mean(stats["selected_fractions"][slot]),
            }
            for slot in slots
        },
    }


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
