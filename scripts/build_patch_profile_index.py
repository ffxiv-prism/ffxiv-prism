from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.patch_profiles import (
    PatchProfileIndex,
    encode_dino_patch_profiles,
    save_patch_profile_index,
)
from ffxiv_prism.matching.spatial_profiles import make_projection
from ffxiv_prism.paths import CACHE_DIR
from ffxiv_prism.torch_device import resolve_torch_device


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build compact local DINO patch profiles for one candidate crop."
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-crop", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--preprocess", choices=("color", "gray"), default="color")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    selected = [
        (index, row)
        for index, row in enumerate(metadata)
        if str(row.get("candidate_crop")) == args.candidate_crop
    ]
    if not selected:
        raise ValueError(f"No candidate rows use crop {args.candidate_crop!r}")
    device = resolve_torch_device(args.device)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device).eval()
    projection = make_projection(
        int(model.config.hidden_size),
        args.projection_dim,
        args.projection_seed,
    )
    profiles = encode_dino_patch_profiles(
        model,
        [
            {
                "path": PROJECT_ROOT / str(row["local_path"]),
                "crop_box_normalized": row.get("candidate_crop_box_normalized"),
            }
            for _, row in selected
        ],
        projection=projection,
        grid_size=args.grid_size,
        batch_size=args.batch_size,
        device=device,
        preprocess=args.preprocess,
    )
    index = PatchProfileIndex(
        profiles=profiles,
        projection=projection,
        metadata_indices=np.asarray(
            [metadata_index for metadata_index, _ in selected],
            dtype=np.int64,
        ),
        piece_ids_by_profile=tuple(
            tuple(int(piece["piece_id"]) for piece in row.get("pieces") or [])
            for _, row in selected
        ),
        model_genders=tuple(
            str(row.get("model_gender") or "") for _, row in selected
        ),
        model_id=args.model_id,
        preprocess=args.preprocess,
        candidate_crop=args.candidate_crop,
        grid_size=args.grid_size,
    )
    save_patch_profile_index(args.output, index)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profile_count": len(index.profiles),
                "profile_shape": list(index.profiles.shape),
                "model_id": index.model_id,
                "candidate_crop": index.candidate_crop,
                "preprocess": index.preprocess,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
