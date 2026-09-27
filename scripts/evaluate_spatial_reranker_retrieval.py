from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

np.linalg.inv(np.eye(2, dtype=np.float32))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.spatial_profiles import (
    encode_dino_row_profiles,
    load_spatial_profile_index,
    spatial_profile_scores,
)
from ffxiv_prism.paths import CACHE_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from scripts.evaluate_cached_query_retrieval import load_query_embeddings
from scripts.evaluate_glamour_embedding_model import search_crops
from scripts.evaluate_retrieval_funnel import (
    ROLE_PROFILES,
    VISIBLE_WEIGHTS,
    appearance_name_key,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    build_fold_context,
    crop_record_weight,
    summarize_ranked_cases,
    validate_cutoffs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-rank the leading LEGS retrieval candidates with vertically ordered "
            "grayscale DINO patch-token profiles."
        )
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--spatial-index", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="Spatial percentile weight; 1 uses spatial order within rerank-k.",
    )
    parser.add_argument("--rrf-k", type=int, default=1)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()
    if not 0.0 <= args.blend <= 1.0:
        parser.error("--blend must be between 0 and 1")
    if args.rerank_k < 1 or args.rerank_k > args.candidate_k:
        parser.error("--rerank-k must be within candidate-k")

    from transformers import AutoModel

    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)
    examples = read_jsonl(args.examples)
    quality_by_id = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
    }
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(args.index_dir / "embeddings.npy", mmap_mode="r")
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality_by_id,
        regions_dir=args.regions_dir,
        slot="LEGS",
    )
    query_embeddings = load_query_embeddings(args.query_cache)
    search = search_crops(
        metadata,
        candidate_embeddings,
        context["records"],
        query_embeddings,
        args.candidate_k,
        device=args.device,
    )

    spatial_index = load_spatial_profile_index(args.spatial_index)
    device = resolve_torch_device(args.device)
    model = AutoModel.from_pretrained(
        spatial_index.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    spatial_record_by_example = select_spatial_records(context["records"])
    ordered_examples = [
        str(example["example_id"])
        for example in context["examples"]
        if str(example["example_id"]) in spatial_record_by_example
    ]
    query_profiles = encode_dino_row_profiles(
        model,
        [spatial_record_by_example[value] for value in ordered_examples],
        projection=spatial_index.projection,
        batch_size=args.batch_size,
        device=device,
    )
    query_profile_by_example = dict(
        zip(ordered_examples, query_profiles, strict=True)
    )
    profile_indices_by_piece: dict[int, list[int]] = defaultdict(list)
    for profile_index, piece_ids in enumerate(spatial_index.piece_ids_by_profile):
        for piece_id in piece_ids:
            profile_indices_by_piece[piece_id].append(profile_index)

    baseline_cases, reranked_cases, diagnostics = rank_cases(
        context["examples"],
        context["records"],
        search,
        metadata,
        query_profile_by_example,
        spatial_index.profiles,
        profile_indices_by_piece,
        rerank_k=args.rerank_k,
        blend=args.blend,
        rrf_k=args.rrf_k,
    )
    payload = {
        "slot": "LEGS",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "index_dir": portable_path(args.index_dir),
        "spatial_index": portable_path(args.spatial_index),
        "query_cache": portable_path(args.query_cache),
        "candidate_k": args.candidate_k,
        "rerank_k": args.rerank_k,
        "blend": args.blend,
        "rrf_k": args.rrf_k,
        "cutoffs": list(cutoffs),
        "baseline_summary": summarize_ranked_cases(baseline_cases, cutoffs),
        "reranked_summary": summarize_ranked_cases(reranked_cases, cutoffs),
        "baseline_cases": baseline_cases,
        "reranked_cases": reranked_cases,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "baseline": payload["baseline_summary"],
                "reranked": payload["reranked_summary"],
            },
            indent=2,
        )
    )
    return 0


def select_spatial_records(
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    priority = {"legs_context": 0, "legs_thighs": 1, "legs_left": 2, "legs_right": 3}
    output = {}
    for example_id, rows in records.items():
        usable = [
            row
            for row in rows
            if row.get("slot") == "LEGS"
            and row.get("visibility") not in {"pose_uncertain", "out_of_frame"}
            and row.get("name") in priority
        ]
        if usable:
            row = min(usable, key=lambda value: priority[str(value["name"])])
            output[str(example_id)] = {"path": row["path"]}
    return output


def rank_cases(
    examples: Sequence[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    search_results_by_crop: dict[str, list[dict[str, Any]]],
    metadata: list[dict[str, Any]],
    query_profiles: dict[str, np.ndarray],
    candidate_profiles: np.ndarray,
    profile_indices_by_piece: dict[int, list[int]],
    *,
    rerank_k: int,
    blend: float,
    rrf_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    slot = "LEGS"
    available_ids = available_pieces_by_slot(metadata)[slot]
    available_keys = available_appearance_keys_by_slot(metadata)[slot]
    role_weights = ROLE_PROFILES["balanced"]
    baseline_cases = []
    reranked_cases = []
    diagnostics = []
    for example in examples:
        example_id = str(example["example_id"])
        label = (example.get("core_slot_labels") or {}).get(slot)
        base = {
            "example_id": example_id,
            "glamour_id": int(example["glamour_id"]),
            "model_gender": example.get("model_gender"),
            "model_race": example.get("model_race"),
            "expected_name": label.get("item_name") if label else None,
            "rank": None,
        }
        status, expected_ids, expected_key = label_status(
            label, available_ids, available_keys
        )
        slot_records = [
            record
            for record in records.get(example_id, [])
            if record.get("slot") == slot
            and VISIBLE_WEIGHTS.get(str(record.get("visibility")), 0.0) > 0.0
        ]
        if status == "eligible" and not slot_records:
            status = "no_usable_crop"
        if status != "eligible":
            row = {**base, "status": status}
            baseline_cases.append(row)
            reranked_cases.append(dict(row))
            continue

        scores: dict[int, float] = {}
        keys_by_piece: dict[int, str] = {}
        for record in slot_records:
            weight = crop_record_weight(record, role_weights, {})
            for offset, result in enumerate(
                search_results_by_crop.get(str(record["path"]), []), start=1
            ):
                piece_id = int(result["piece_id"])
                result_rank = int(result.get("rank", offset))
                scores[piece_id] = scores.get(piece_id, 0.0) + weight / (
                    rrf_k + result_rank
                )
                keys_by_piece[piece_id] = appearance_name_key(result.get("name_en"))
        ranked = sorted(scores, key=scores.get, reverse=True)
        baseline_rank = truth_rank(ranked, expected_ids, expected_key, keys_by_piece)
        reranked = list(ranked)
        spatial_scores: dict[int, float] = {}
        if example_id in query_profiles:
            leading = ranked[:rerank_k]
            for piece_id in leading:
                indices = profile_indices_by_piece.get(piece_id, [])
                if indices:
                    values = spatial_profile_scores(
                        query_profiles[example_id], candidate_profiles[indices]
                    )
                    spatial_scores[piece_id] = float(values.max())
            if spatial_scores:
                retrieval_percentile = percentile_scores(
                    {piece_id: scores[piece_id] for piece_id in leading}
                )
                spatial_percentile = percentile_scores(spatial_scores)
                reranked_leading = sorted(
                    leading,
                    key=lambda piece_id: (
                        (1.0 - blend) * retrieval_percentile.get(piece_id, 0.0)
                        + blend * spatial_percentile.get(piece_id, 0.0),
                        scores[piece_id],
                    ),
                    reverse=True,
                )
                reranked[:rerank_k] = reranked_leading
        reranked_rank = truth_rank(
            reranked, expected_ids, expected_key, keys_by_piece
        )
        baseline_cases.append({**base, "status": status, "rank": baseline_rank})
        reranked_cases.append({**base, "status": status, "rank": reranked_rank})
        diagnostics.append(
            {
                "example_id": example_id,
                "baseline_top_piece_id": ranked[0] if ranked else None,
                "reranked_top_piece_id": reranked[0] if reranked else None,
                "truth_baseline_rank": baseline_rank,
                "truth_reranked_rank": reranked_rank,
                "spatial_scores": {
                    str(piece_id): spatial_scores[piece_id]
                    for piece_id in ranked[:rerank_k]
                    if piece_id in spatial_scores
                },
            }
        )
    return baseline_cases, reranked_cases, diagnostics


def label_status(
    label: dict[str, Any] | None,
    available_ids: set[int],
    available_keys: set[str],
) -> tuple[str, set[int], str]:
    if not label or label.get("piece_match_type") == "empty":
        return "empty_label", set(), ""
    expected_ids = {int(value) for value in label.get("piece_ids", [])}
    if not expected_ids:
        return "missing_truth_piece_id", set(), ""
    expected_key = appearance_name_key(label.get("item_name"))
    if not (expected_ids & available_ids) and not (
        expected_key and expected_key in available_keys
    ):
        return "truth_not_in_index", expected_ids, expected_key
    return "eligible", expected_ids & available_ids, expected_key


def truth_rank(
    ranked_ids: Sequence[int],
    expected_ids: set[int],
    expected_key: str,
    keys_by_piece: dict[int, str],
) -> int | None:
    return next(
        (
            rank
            for rank, piece_id in enumerate(ranked_ids, start=1)
            if piece_id in expected_ids or keys_by_piece.get(piece_id) == expected_key
        ),
        None,
    )


def percentile_scores(scores: dict[int, float]) -> dict[int, float]:
    ordered = sorted(scores, key=scores.get)
    denominator = max(1, len(ordered) - 1)
    return {
        piece_id: rank / denominator
        for rank, piece_id in enumerate(ordered)
    }


if __name__ == "__main__":
    raise SystemExit(main())
