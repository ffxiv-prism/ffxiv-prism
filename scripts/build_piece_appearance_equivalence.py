from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from ffxiv_prism.jsonl import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build direct same-appearance candidates from canonical image embeddings."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indexes/facebook-dinov2-base-color"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--race", default="hyur")
    parser.add_argument("--model-gender", default="female")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.975,
        help="Minimum cosine similarity for pieces from different canonical families.",
    )
    parser.add_argument(
        "--within-family-threshold",
        type=float,
        default=0.94,
        help="Minimum cosine similarity for pieces sharing a canonical family slug.",
    )
    parser.add_argument("--db", type=Path, default=Path("data/processed/eorzea.sqlite"))
    args = parser.parse_args()

    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    embeddings = np.load(args.index_dir / "embeddings.npy", mmap_mode="r")
    conn = sqlite3.connect(args.db)
    try:
        piece_families = {
            int(piece_id): str(family_slug)
            for piece_id, family_slug in conn.execute(
                """
                SELECT gp.piece_id, MIN(i.family_slug)
                FROM gear_pieces gp
                JOIN gear_piece_images gpi ON gpi.piece_id = gp.piece_id
                JOIN images i ON i.image_row_id = gpi.image_row_id
                WHERE i.family_slug IS NOT NULL AND i.family_slug != ''
                GROUP BY gp.piece_id
                """
            )
        }
    finally:
        conn.close()
    indices_by_slot_piece: dict[str, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    names: dict[int, str] = {}
    for index, row in enumerate(metadata):
        if row.get("race") != args.race or row.get("model_gender") != args.model_gender:
            continue
        slot = str(row.get("slot") or "")
        for piece in row.get("pieces", []):
            piece_id = int(piece["piece_id"])
            indices_by_slot_piece[slot][piece_id].append(index)
            names[piece_id] = str(piece.get("name_en") or piece_id)

    equivalents: dict[int, set[int]] = defaultdict(set)
    edge_scores: list[dict] = []
    slot_piece_counts: dict[str, int] = {}
    for slot, indices_by_piece in sorted(indices_by_slot_piece.items()):
        piece_ids = sorted(indices_by_piece)
        slot_piece_counts[slot] = len(piece_ids)
        vectors = []
        for piece_id in piece_ids:
            vector = np.asarray(embeddings[indices_by_piece[piece_id]], dtype=np.float32).mean(axis=0)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            vectors.append(vector)
        matrix = np.stack(vectors)
        similarities = matrix @ matrix.T
        left, right = np.where(
            np.triu(similarities, k=1)
            >= min(args.threshold, args.within_family_threshold)
        )
        for left_index, right_index in zip(left.tolist(), right.tolist(), strict=True):
            piece_a = piece_ids[left_index]
            piece_b = piece_ids[right_index]
            score = float(similarities[left_index, right_index])
            family_a = piece_families.get(piece_a)
            family_b = piece_families.get(piece_b)
            same_family = bool(family_a and family_a == family_b)
            required_threshold = (
                args.within_family_threshold if same_family else args.threshold
            )
            if score < required_threshold:
                continue
            equivalents[piece_a].add(piece_b)
            equivalents[piece_b].add(piece_a)
            edge_scores.append(
                {
                    "piece_a": piece_a,
                    "piece_a_name": names[piece_a],
                    "piece_b": piece_b,
                    "piece_b_name": names[piece_b],
                    "score": score,
                    "slot": slot,
                    "same_family": same_family,
                }
            )

    payload = {
        "direct_equivalents": {
            str(piece_id): sorted(values)
            for piece_id, values in sorted(equivalents.items())
        },
        "edge_count": len(edge_scores),
        "edges": sorted(edge_scores, key=lambda row: row["score"], reverse=True),
        "index_dir": args.index_dir.as_posix(),
        "database": args.db.as_posix(),
        "model_gender": args.model_gender,
        "race": args.race,
        "slot_piece_counts": slot_piece_counts,
        "threshold": args.threshold,
        "within_family_threshold": args.within_family_threshold,
    }
    write_json(args.output, payload)
    print(
        f"pieces={sum(slot_piece_counts.values())} edges={len(edge_scores)} "
        f"mapped={len(equivalents)} output={args.output}"
    )


if __name__ == "__main__":
    main()
