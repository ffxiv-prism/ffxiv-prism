from __future__ import annotations

import argparse
from pathlib import Path

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import PROCESSED_DIR, resolve_project_path
from ffxiv_prism.vision.retrieval import (
    DEFAULT_INDEX_DIR,
    ClipImageEncoder,
    load_index,
    open_rgb,
    search_index,
    transform_eval_image,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate single-slot visual retrieval.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--cases", type=Path, default=Path("data/samples/retrieval_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "retrieval_eval_results.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    index = load_index(args.index_dir)
    encoder = ClipImageEncoder(index.manifest["model_id"], args.device)
    outcomes = []
    for case in read_jsonl(args.cases):
        query_path = resolve_project_path(case["query_path"])
        image = open_rgb(query_path)
        try:
            transform = case.get("query_transform")
            query_image = transform_eval_image(image, transform) if transform else image
            try:
                query_embedding = encoder.encode_image(query_image)
            finally:
                if query_image is not image:
                    query_image.close()
        finally:
            image.close()
        exclude_image_row_id = int(case["query_image_row_id"]) if case.get("exclude_query_image") else None
        results = search_index(
            index,
            query_embedding,
            case["slot"],
            args.top_k,
            exclude_image_row_id,
        )
        expected_piece_id = int(case["expected_piece_id"])
        rank = next((row["rank"] for row in results if int(row["piece_id"]) == expected_piece_id), None)
        outcomes.append({**case, "hit": rank is not None, "rank": rank, "results": results})

    summary = summarize(outcomes, args.top_k)
    write_json(args.output, summary)
    print(
        f"cases={summary['case_count']} recall@{args.top_k}={summary['recall_at_k']:.3f} "
        f"mrr={summary['mean_reciprocal_rank']:.3f}"
    )
    return 0


def summarize(outcomes: list[dict], top_k: int) -> dict:
    hits = sum(bool(row["hit"]) for row in outcomes)
    by_slot = {}
    for slot in sorted({row["slot"] for row in outcomes}):
        slot_rows = [row for row in outcomes if row["slot"] == slot]
        slot_hits = sum(bool(row["hit"]) for row in slot_rows)
        by_slot[slot] = {
            "case_count": len(slot_rows),
            "hits": slot_hits,
            "recall_at_k": slot_hits / len(slot_rows),
            "mean_reciprocal_rank": (
                sum(1.0 / row["rank"] for row in slot_rows if row["rank"]) / len(slot_rows)
            ),
        }
    return {
        "case_count": len(outcomes),
        "top_k": top_k,
        "hits": hits,
        "recall_at_k": hits / len(outcomes) if outcomes else 0.0,
        "mean_reciprocal_rank": (
            sum(1.0 / row["rank"] for row in outcomes if row["rank"]) / len(outcomes)
            if outcomes
            else 0.0
        ),
        "by_slot": by_slot,
        "cases": outcomes,
    }


if __name__ == "__main__":
    raise SystemExit(main())
