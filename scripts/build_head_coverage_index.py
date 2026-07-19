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
    HeadCoverageIndex,
    encode_head_coverage_profiles,
    save_head_coverage_index,
)
from ffxiv_prism.vision.human_parsing import (
    DEFAULT_HUMAN_PARSING_MODEL,
    HumanParser,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build headwear coverage profiles from fixed candidate renders."
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-crop", default="full")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    selected = [
        (index, row)
        for index, row in enumerate(metadata)
        if str(row.get("candidate_crop")) == args.candidate_crop
    ]
    parser_model = HumanParser(device=args.device)
    exposed, person = encode_head_coverage_profiles(
        parser_model,
        [
            {
                "path": PROJECT_ROOT / str(row["local_path"]),
                "crop_box_normalized": row.get("candidate_crop_box_normalized"),
            }
            for _, row in selected
        ],
        batch_size=args.batch_size,
    )
    index = HeadCoverageIndex(
        exposed_fractions=exposed,
        person_fractions=person,
        metadata_indices=np.asarray([value for value, _ in selected], dtype=np.int64),
        piece_ids_by_profile=tuple(
            tuple(int(piece["piece_id"]) for piece in row["pieces"])
            for _, row in selected
        ),
        model_id=DEFAULT_HUMAN_PARSING_MODEL,
        candidate_crop=args.candidate_crop,
    )
    save_head_coverage_index(args.output, index)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profile_count": len(index.exposed_fractions),
                "exposed_mean": float(index.exposed_fractions.mean()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
