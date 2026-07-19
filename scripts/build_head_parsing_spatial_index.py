from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

np.linalg.inv(np.eye(2, dtype=np.float32))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.coverage_profiles import (
    encode_head_parsing_spatial_profiles,
)
from ffxiv_prism.matching.spatial_profiles import (
    SpatialProfileIndex,
    save_spatial_profile_index,
)
from ffxiv_prism.vision.human_parsing import (
    DEFAULT_HUMAN_PARSING_MODEL,
    HumanParser,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build scale-normalized 2D human-parsing profiles for HEAD renders."
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-crop", default="full")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    selected = [
        (index, row)
        for index, row in enumerate(metadata)
        if str(row.get("candidate_crop")) == args.candidate_crop
    ]
    stop = None if args.limit is None else args.start + args.limit
    selected = selected[args.start : stop]
    if not selected:
        parser.error("selected candidate slice is empty")
    human_parser = HumanParser(device=args.device)
    profiles = encode_head_parsing_spatial_profiles(
        human_parser,
        [
            {
                "path": PROJECT_ROOT / str(row["local_path"]),
                "crop_box_normalized": row.get("candidate_crop_box_normalized"),
            }
            for _, row in selected
        ],
        batch_size=args.batch_size,
        grid_size=args.grid_size,
    )
    index = SpatialProfileIndex(
        profiles=profiles,
        projection=np.empty((0, 0), dtype=np.float32),
        metadata_indices=np.asarray([value for value, _ in selected], dtype=np.int64),
        image_row_ids=np.asarray(
            [int(row["image_row_id"]) for _, row in selected], dtype=np.int64
        ),
        piece_ids_by_profile=tuple(
            tuple(int(piece["piece_id"]) for piece in row["pieces"])
            for _, row in selected
        ),
        model_id=DEFAULT_HUMAN_PARSING_MODEL,
        preprocess=f"human-parsing-head-layout-{args.grid_size}x{args.grid_size}-v1",
        candidate_crop=args.candidate_crop,
    )
    save_spatial_profile_index(args.output, index)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profile_count": len(index.profiles),
                "profile_shape": list(index.profiles.shape),
                "start": args.start,
                "limit": args.limit,
                "first_metadata_index": int(index.metadata_indices[0]),
                "last_metadata_index": int(index.metadata_indices[-1]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
