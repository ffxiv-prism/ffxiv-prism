from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from ffxiv_prism.jsonl import write_jsonl
from ffxiv_prism.paths import SAMPLES_DIR, portable_path, resolve_project_path
from ffxiv_prism.vision.retrieval import (
    AUGMENTATION_PROFILES,
    DEFAULT_INDEX_DIR,
    VALID_SLOTS,
    augment_eval_image,
    load_index,
    open_rgb,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate noisy retrieval query images and an evaluation manifest.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--profile", choices=sorted(AUGMENTATION_PROFILES), required=True)
    parser.add_argument("--per-slot", type=int, default=6)
    parser.add_argument("--images-dir", type=Path, default=SAMPLES_DIR / "retrieval_eval_images")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index = load_index(args.index_dir)
    candidates = exact_candidates_by_slot(index.metadata)
    recipes = AUGMENTATION_PROFILES[args.profile]
    rows = []
    profile_dir = args.images_dir / args.profile
    profile_dir.mkdir(parents=True, exist_ok=True)

    for slot in VALID_SLOTS:
        slot_candidates = candidates[slot]
        positions = sample_positions(len(slot_candidates), args.per_slot)
        for offset, position in enumerate(positions):
            source = slot_candidates[position]
            recipe = recipes[offset % len(recipes)]
            source_path = resolve_project_path(source["local_path"])
            seed = stable_seed(source["image_row_id"], recipe)
            image = open_rgb(source_path)
            try:
                augmented = augment_eval_image(image, recipe, seed)
                try:
                    output_path = profile_dir / f"{slot.lower()}-{source['piece_id']}-{recipe}.jpg"
                    quality = 88 if args.profile == "mild" else 68
                    augmented.save(output_path, "JPEG", quality=quality, optimize=True, subsampling=2)
                finally:
                    augmented.close()
            finally:
                image.close()
            rows.append(
                {
                    "case_id": f"{args.profile}-{slot.lower()}-{source['piece_id']}-{recipe}",
                    "profile": args.profile,
                    "augmentation": recipe,
                    "slot": slot,
                    "query_path": portable_path(output_path),
                    "source_path": source["local_path"],
                    "source_image_row_id": source["image_row_id"],
                    "expected_piece_id": source["piece_id"],
                    "expected_name_en": source.get("name_en"),
                }
            )

    write_jsonl(args.output, rows)
    print(f"evaluation_cases={len(rows)} profile={args.profile} output={args.output}")
    return 0


def exact_candidates_by_slot(metadata: list[dict]) -> dict[str, list[dict]]:
    by_slot: dict[str, list[dict]] = defaultdict(list)
    seen_pieces: set[int] = set()
    for image in metadata:
        exact_pieces = [piece for piece in image["pieces"] if piece["link_type"] == "exact_slot"]
        if not exact_pieces:
            continue
        piece = exact_pieces[0]
        piece_id = int(piece["piece_id"])
        if piece_id in seen_pieces:
            continue
        seen_pieces.add(piece_id)
        by_slot[image["slot"]].append(
            {
                "image_row_id": int(image["image_row_id"]),
                "local_path": image["local_path"],
                "piece_id": piece_id,
                "name_en": piece.get("name_en"),
            }
        )
    return by_slot


def sample_positions(candidate_count: int, sample_count: int) -> list[int]:
    if candidate_count == 0 or sample_count <= 0:
        return []
    if sample_count == 1:
        return [candidate_count // 2]
    if sample_count >= candidate_count:
        return list(range(candidate_count))
    return [round(index * (candidate_count - 1) / (sample_count - 1)) for index in range(sample_count)]


def stable_seed(image_row_id: int, recipe: str) -> int:
    digest = hashlib.sha256(f"{image_row_id}:{recipe}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


if __name__ == "__main__":
    raise SystemExit(main())
