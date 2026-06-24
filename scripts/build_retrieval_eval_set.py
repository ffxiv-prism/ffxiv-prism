from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ffxiv_prism.jsonl import write_jsonl
from ffxiv_prism.paths import SAMPLES_DIR
from ffxiv_prism.vision.retrieval import (
    DEFAULT_INDEX_DIR,
    EVAL_TRANSFORMS,
    VALID_SLOTS,
    load_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic visual retrieval evaluation set.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output", type=Path, default=SAMPLES_DIR / "retrieval_eval.generated.jsonl")
    parser.add_argument("--per-slot", type=int, default=2)
    parser.add_argument("--mode", choices=("transformed", "cross-view"), default="transformed")
    args = parser.parse_args()

    index = load_index(args.index_dir)
    candidates_by_slot = build_candidates(index.metadata, args.mode)
    selected: list[dict] = []
    for slot in VALID_SLOTS:
        candidates = candidates_by_slot[slot]
        for offset, position in enumerate(sample_positions(len(candidates), args.per_slot)):
            case = dict(candidates[position])
            if args.mode == "transformed":
                case["query_transform"] = EVAL_TRANSFORMS[offset % len(EVAL_TRANSFORMS)]
            selected.append(case)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, selected)
    print(f"evaluation_cases={len(selected)} mode={args.mode} output={args.output}")
    return 0


def build_candidates(metadata: list[dict], mode: str) -> dict[str, list[dict]]:
    by_slot: dict[str, list[dict]] = defaultdict(list)
    if mode == "transformed":
        for image in metadata:
            exact_pieces = [piece for piece in image["pieces"] if piece["link_type"] == "exact_slot"]
            if not exact_pieces:
                continue
            piece = exact_pieces[0]
            by_slot[image["slot"]].append(case_payload(image, piece))
        return by_slot

    by_piece: dict[int, list[dict]] = defaultdict(list)
    for image in metadata:
        for piece in image["pieces"]:
            by_piece[int(piece["piece_id"])].append(image)
    for piece_id in sorted(by_piece):
        unique_images = {int(row["image_row_id"]): row for row in by_piece[piece_id]}
        if len(unique_images) < 2:
            continue
        query = unique_images[min(unique_images)]
        piece = next(piece for piece in query["pieces"] if int(piece["piece_id"]) == piece_id)
        payload = case_payload(query, piece)
        payload["exclude_query_image"] = True
        by_slot[query["slot"]].append(payload)
    return by_slot


def case_payload(image: dict, piece: dict) -> dict:
    piece_id = int(piece["piece_id"])
    return {
        "case_id": f"{image['slot'].lower()}-{piece_id}",
        "slot": image["slot"],
        "query_image_row_id": image["image_row_id"],
        "query_path": image["local_path"],
        "expected_piece_id": piece_id,
        "expected_name_en": piece.get("name_en"),
    }


def sample_positions(candidate_count: int, sample_count: int) -> list[int]:
    if candidate_count == 0 or sample_count <= 0:
        return []
    if sample_count == 1:
        return [candidate_count // 2]
    if sample_count >= candidate_count:
        return list(range(candidate_count))
    return [round(index * (candidate_count - 1) / (sample_count - 1)) for index in range(sample_count)]


if __name__ == "__main__":
    raise SystemExit(main())
