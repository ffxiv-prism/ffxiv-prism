from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.collectors.eorzea_collection.glamour_parser import (  # noqa: E402
    appearance_name_key,
)
from ffxiv_prism.jsonl import read_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402
from ffxiv_prism.torch_device import resolve_torch_device  # noqa: E402


GENDER_PAIRS = {"female": "male", "male": "female"}


def truth_sets(row: Mapping[str, Any]) -> tuple[set[int], set[str]]:
    piece_ids = {
        int(piece["piece_id"])
        for piece in row.get("pieces", [])
        if piece.get("piece_id") is not None
    }
    appearance_keys = {
        key
        for key in (
            appearance_name_key(piece.get("name_en"))
            for piece in row.get("pieces", [])
        )
        if key
    }
    return piece_ids, appearance_keys


def candidate_truth_sets(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[int], set[str]]:
    piece_ids: set[int] = set()
    appearance_keys: set[str] = set()
    for row in rows:
        row_ids, row_keys = truth_sets(row)
        piece_ids.update(row_ids)
        appearance_keys.update(row_keys)
    return piece_ids, appearance_keys


def rank_cross_view_result(
    ordered_rows: Sequence[Mapping[str, Any]],
    expected_piece_ids: set[int],
    expected_appearance_keys: set[str],
) -> int | None:
    seen_piece_ids: set[int] = set()
    piece_rank = 0
    for row in ordered_rows:
        for piece in row.get("pieces", []):
            piece_id = int(piece["piece_id"])
            if piece_id in seen_piece_ids:
                continue
            seen_piece_ids.add(piece_id)
            piece_rank += 1
            key = appearance_name_key(piece.get("name_en"))
            if piece_id in expected_piece_ids or (
                key and key in expected_appearance_keys
            ):
                return piece_rank
    return None


def summarize_cases(
    cases: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    eligible = [case for case in cases if case.get("status") == "eligible"]
    ranks = [int(case["rank"]) for case in eligible if case.get("rank") is not None]
    denominator = len(eligible)
    return {
        "status_counts": dict(Counter(str(case.get("status")) for case in cases)),
        "eligible_case_count": denominator,
        "retrieved_truth_count": len(ranks),
        "hits": {
            str(cutoff): sum(rank <= cutoff for rank in ranks)
            for cutoff in cutoffs
        },
        "recall": {
            str(cutoff): (
                sum(rank <= cutoff for rank in ranks) / denominator
                if denominator
                else 0.0
            )
            for cutoff in cutoffs
        },
        "mrr": (
            sum(1.0 / rank for rank in ranks) / denominator
            if denominator
            else 0.0
        ),
    }


def select_query_indices(
    metadata: Sequence[Mapping[str, Any]],
    *,
    slot: str,
    query_crop: str,
) -> list[int]:
    selected = []
    seen_source_rows: set[int] = set()
    for index, row in enumerate(metadata):
        if (
            str(row.get("slot")) != slot
            or str(row.get("candidate_crop")) != query_crop
            or str(row.get("model_gender", "")).lower() not in GENDER_PAIRS
        ):
            continue
        source_row = int(row.get("source_image_row_id", row.get("image_row_id", index)))
        if source_row in seen_source_rows:
            continue
        seen_source_rows.add(source_row)
        selected.append(index)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure candidate-library cross-gender retrieval without player "
            "images, isolating representation quality from screenshot domain gap."
        )
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--slot", default="HANDS")
    parser.add_argument("--query-crop", default="both_hands")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--row-top-k", type=int, default=1000)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100])
    args = parser.parse_args()

    if args.batch_size < 1 or args.row_top_k < 1:
        parser.error("batch-size and row-top-k must be positive")
    cutoffs = tuple(sorted({int(value) for value in args.cutoffs}))
    if not cutoffs or cutoffs[0] < 1 or cutoffs[-1] > args.row_top_k:
        parser.error("cutoffs must be positive and no larger than row-top-k")

    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    embeddings = np.load(args.index_dir / "embeddings.npy", mmap_mode="r")
    if len(metadata) != len(embeddings):
        raise ValueError("Candidate metadata and embeddings are misaligned")
    query_indices = select_query_indices(
        metadata,
        slot=args.slot,
        query_crop=args.query_crop,
    )
    if args.limit_queries is not None:
        query_indices = query_indices[: args.limit_queries]
    if not query_indices:
        raise ValueError("No cross-gender query rows were selected")

    device = resolve_torch_device(args.device)
    cases: list[dict[str, Any]] = []
    for query_gender, target_gender in GENDER_PAIRS.items():
        gender_query_indices = [
            index
            for index in query_indices
            if str(metadata[index].get("model_gender", "")).lower()
            == query_gender
        ]
        candidate_indices = np.asarray(
            [
                index
                for index, row in enumerate(metadata)
                if str(row.get("slot")) == args.slot
                and str(row.get("model_gender", "")).lower() == target_gender
            ],
            dtype=np.int64,
        )
        candidate_rows = [metadata[int(index)] for index in candidate_indices]
        available_ids, available_keys = candidate_truth_sets(candidate_rows)
        candidate_tensor = torch.as_tensor(
            np.asarray(embeddings[candidate_indices]),
            device=device,
        )
        top_k = min(args.row_top_k, len(candidate_indices))
        for start in range(0, len(gender_query_indices), args.batch_size):
            batch_indices = gender_query_indices[start : start + args.batch_size]
            query_tensor = torch.as_tensor(
                np.asarray(embeddings[batch_indices]),
                device=device,
            )
            with torch.inference_mode():
                order = torch.topk(
                    query_tensor @ candidate_tensor.T,
                    k=top_k,
                    dim=1,
                ).indices.cpu().numpy()
            for query_index, ordered_local in zip(
                batch_indices,
                order,
                strict=True,
            ):
                row = metadata[query_index]
                expected_ids, expected_keys = truth_sets(row)
                status = "eligible"
                if not (expected_ids & available_ids) and not (
                    expected_keys & available_keys
                ):
                    status = "truth_not_in_opposite_gender_index"
                rank = (
                    rank_cross_view_result(
                        [candidate_rows[int(index)] for index in ordered_local],
                        expected_ids,
                        expected_keys,
                    )
                    if status == "eligible"
                    else None
                )
                cases.append(
                    {
                        "query_metadata_index": query_index,
                        "source_image_row_id": row.get("source_image_row_id"),
                        "query_gender": query_gender,
                        "target_gender": target_gender,
                        "expected_piece_ids": sorted(expected_ids),
                        "expected_names": sorted(
                            {
                                str(piece.get("name_en") or "")
                                for piece in row.get("pieces", [])
                            }
                        ),
                        "status": status,
                        "rank": rank,
                    }
                )
        del candidate_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    by_query_gender = {
        gender: summarize_cases(
            [case for case in cases if case["query_gender"] == gender],
            cutoffs,
        )
        for gender in GENDER_PAIRS
    }
    norms = np.linalg.norm(
        np.asarray(embeddings[query_indices], dtype=np.float32),
        axis=1,
    )
    payload = {
        "version": 1,
        "diagnostic": "candidate_cross_gender_retrieval",
        "index_dir": portable_path(args.index_dir),
        "slot": args.slot,
        "query_crop": args.query_crop,
        "row_top_k": args.row_top_k,
        "query_count": len(query_indices),
        "embedding_norm": {
            "min": float(norms.min()),
            "mean": float(norms.mean()),
            "max": float(norms.max()),
        },
        "summary": summarize_cases(cases, cutoffs),
        "by_query_gender": by_query_gender,
        "cases": cases,
        "interpretation": (
            "This measures representation invariance inside the canonical "
            "candidate domain. It does not measure player-screenshot accuracy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": portable_path(args.output),
                "summary": payload["summary"],
                "by_query_gender": by_query_gender,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
