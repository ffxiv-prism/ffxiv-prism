from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ffxiv_prism.jsonl import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare canonical index embeddings for two pieces.")
    parser.add_argument("piece_a", type=int)
    parser.add_argument("piece_b", type=int)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indexes/facebook-dinov2-base-color"),
    )
    parser.add_argument("--race", default="hyur")
    parser.add_argument("--model-gender", default="female")
    args = parser.parse_args()

    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    embeddings = np.load(args.index_dir / "embeddings.npy", mmap_mode="r")

    def indices(piece_id: int) -> list[int]:
        return [
            index
            for index, row in enumerate(metadata)
            if row.get("race") == args.race
            and row.get("model_gender") == args.model_gender
            and any(int(piece["piece_id"]) == piece_id for piece in row.get("pieces", []))
        ]

    left = indices(args.piece_a)
    right = indices(args.piece_b)
    if not left or not right:
        raise SystemExit(f"missing canonical rows: left={len(left)} right={len(right)}")
    scores = np.asarray(embeddings[left]) @ np.asarray(embeddings[right]).T
    best_flat = int(np.argmax(scores))
    left_offset, right_offset = np.unravel_index(best_flat, scores.shape)
    left_index = left[int(left_offset)]
    right_index = right[int(right_offset)]
    print(f"cosine={float(scores[left_offset, right_offset]):.6f}")
    print(f"piece_a={metadata[left_index]['local_path']}")
    print(f"piece_b={metadata[right_index]['local_path']}")


if __name__ == "__main__":
    main()
