from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import CACHE_DIR, portable_path
from scripts.evaluate_glamour_embedding_model import DINO_MEAN, DINO_STD

DEFAULT_INPUT = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_OUTPUT = Path("data/indexes/facebook-dinov2-base-slot-aligned")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-embed canonical renders with slot-aligned crops so mannequin "
            "skin and unrelated body regions do not dominate matching."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source_rows = read_jsonl(args.input_dir / "metadata.jsonl")
    rows = expand_slot_aligned_rows(source_rows)
    device = torch.device(args.device)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device).eval()
    embeddings = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        pixels = load_candidate_batch(batch).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            tokens = model(pixel_values=pixels).last_hidden_state
            features = torch.nn.functional.normalize(tokens[:, 0], dim=-1)
        embeddings.append(features.float().cpu().numpy())
        if (start // args.batch_size + 1) % 50 == 0:
            print(f"embedded={min(start + args.batch_size, len(rows))}/{len(rows)}")
    matrix = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", matrix)
    write_jsonl(args.output_dir / "metadata.jsonl", rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "source_index": portable_path(args.input_dir),
            "model_id": args.model_id,
            "source_image_count": len(source_rows),
            "aligned_candidate_count": len(rows),
            "slot_policy": {
                "HEAD": ["full"],
                "BODY": ["upper_body"],
                "HANDS": [
                    "both_hands",
                    "left_arm_tight",
                    "right_arm_tight",
                ],
                "LEGS": ["full"],
                "FEET": ["full"],
            },
            "output": portable_path(args.output_dir),
        },
    )
    print(
        f"source={len(source_rows)} aligned={len(rows)} "
        f"dimension={matrix.shape[1]} output={args.output_dir}"
    )
    return 0


def expand_slot_aligned_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        slot = str(row.get("slot", ""))
        variants = crop_variants_for_slot(slot)
        for name, box in variants:
            aligned = dict(row)
            aligned["candidate_crop"] = name
            aligned["candidate_crop_box_normalized"] = list(box)
            aligned["source_image_row_id"] = row.get("image_row_id")
            aligned["image_row_id"] = len(output)
            output.append(aligned)
    return output


def crop_variants_for_slot(
    slot: str,
) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    if slot == "BODY":
        return (("upper_body", (0.0, 0.0, 1.0, 0.72)),)
    if slot == "HANDS":
        return (
            # Most canonical hand renders use a crossed-arms pose.  A full-width
            # variant is required because each hand and sleeve crosses the body
            # centre and is truncated by either one-sided crop.
            ("both_hands", (0.0, 0.12, 1.0, 1.0)),
            ("left_arm_tight", (0.0, 0.20, 0.45, 1.0)),
            ("right_arm_tight", (0.55, 0.12, 1.0, 0.94)),
        )
    return (("full", (0.0, 0.0, 1.0, 1.0)),)


def load_candidate_batch(rows: Sequence[dict[str, Any]]) -> torch.Tensor:
    arrays = []
    for row in rows:
        with Image.open(str(row["local_path"])) as source:
            image = source.convert("RGB")
        left, top, right, bottom = row["candidate_crop_box_normalized"]
        box = (
            int(round(left * image.width)),
            int(round(top * image.height)),
            int(round(right * image.width)),
            int(round(bottom * image.height)),
        )
        crop = image.crop(box)
        image.close()
        resized = crop.resize((224, 224), Image.Resampling.BICUBIC)
        crop.close()
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
