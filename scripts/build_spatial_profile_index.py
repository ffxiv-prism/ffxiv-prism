from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

np.linalg.inv(np.eye(2, dtype=np.float32))

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.spatial_profiles import (
    DEFAULT_PROJECTION_DIM,
    SpatialProfileIndex,
    encode_dino_row_profiles,
    make_projection,
    save_spatial_profile_index,
)
from ffxiv_prism.paths import CACHE_DIR
from ffxiv_prism.torch_device import resolve_torch_device


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build compact grayscale DINO spatial profiles for candidate renders."
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-crop", default="full")
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--projection-dim", type=int, default=DEFAULT_PROJECTION_DIM)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from transformers import AutoModel

    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    selected = [
        (index, row)
        for index, row in enumerate(metadata)
        if str(row.get("candidate_crop")) == args.candidate_crop
    ]
    if not selected:
        raise ValueError(f"No candidate rows use crop {args.candidate_crop!r}")
    device = resolve_torch_device(args.device)
    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    projection = make_projection(
        int(model.config.hidden_size),
        args.projection_dim,
    )
    profile_rows = [
        {
            "path": PROJECT_ROOT / str(row["local_path"]),
            "crop_box_normalized": row.get("candidate_crop_box_normalized"),
        }
        for _, row in selected
    ]
    profiles = encode_dino_row_profiles(
        model,
        profile_rows,
        projection=projection,
        batch_size=args.batch_size,
        device=device,
    )
    index = SpatialProfileIndex(
        profiles=profiles,
        projection=projection,
        metadata_indices=np.asarray([value for value, _ in selected], dtype=np.int64),
        image_row_ids=np.asarray(
            [int(row["image_row_id"]) for _, row in selected], dtype=np.int64
        ),
        piece_ids_by_profile=tuple(
            tuple(int(piece["piece_id"]) for piece in row["pieces"])
            for _, row in selected
        ),
        model_id=args.model_id,
        preprocess="gray-row-profile-v1",
        candidate_crop=args.candidate_crop,
    )
    save_spatial_profile_index(args.output, index)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profile_count": len(index.profiles),
                "profile_shape": list(index.profiles.shape),
                "model_id": index.model_id,
                "candidate_crop": index.candidate_crop,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
