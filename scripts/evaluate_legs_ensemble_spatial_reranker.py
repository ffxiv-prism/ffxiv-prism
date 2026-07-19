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
from ffxiv_prism.matching.inference import fuse_crop_results
from ffxiv_prism.matching.spatial_profiles import (
    encode_dino_row_profiles,
    load_spatial_profile_index,
    rerank_candidates_with_spatial_profile,
)
from ffxiv_prism.paths import CACHE_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from scripts.evaluate_cached_query_retrieval import load_query_embeddings
from scripts.evaluate_glamour_embedding_model import search_crops
from scripts.evaluate_rank_ensemble_retrieval import (
    constrain_to_final_candidate_pool,
    fuse_branch_searches,
)
from scripts.evaluate_retrieval_funnel import (
    appearance_name_key,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    build_fold_context,
    summarize_ranked_cases,
    validate_cutoffs,
)
from scripts.evaluate_spatial_reranker_retrieval import (
    label_status,
    select_spatial_records,
    truth_rank,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate final-pool LEGS ensemble plus spatial row reranking."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--branch", action="append", nargs=2, metavar=("INDEX_DIR", "QUERY_CACHE"), required=True
    )
    parser.add_argument("--spatial-index", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--branch-rrf-k", type=int, default=1)
    parser.add_argument("--pool-branches", type=int, default=1)
    parser.add_argument("--pool-rrf-k", type=int, default=1)
    parser.add_argument("--final-pool-branches", type=int, default=1)
    parser.add_argument("--crop-rrf-k", type=int, default=1)
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument("--blend", type=float, default=0.35)
    parser.add_argument(
        "--mode", choices=("apply", "late-rrf"), default="apply"
    )
    parser.add_argument("--late-rrf-k", type=int, default=1)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)

    metadata = read_jsonl(args.metadata)
    expected_paths = [str(row["local_path"]) for row in metadata]
    examples = read_jsonl(args.examples)
    quality = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
    }
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality,
        regions_dir=args.regions_dir,
        slot="LEGS",
    )
    searches = []
    branches = []
    for index_value, cache_value in args.branch:
        index_dir = Path(index_value)
        cache_path = Path(cache_value)
        branch_metadata = read_jsonl(index_dir / "metadata.jsonl")
        if [str(row["local_path"]) for row in branch_metadata] != expected_paths:
            raise ValueError(f"Candidate metadata mismatch in {index_dir}")
        searches.append(
            search_crops(
                branch_metadata,
                np.load(index_dir / "embeddings.npy", mmap_mode="r"),
                context["records"],
                load_query_embeddings(cache_path),
                args.candidate_k,
                device=args.device,
            )
        )
        branches.append(
            {"index_dir": portable_path(index_dir), "query_cache": portable_path(cache_path)}
        )
    if not 1 <= args.pool_branches <= len(searches):
        parser.error("invalid pool branch count")
    if not 1 <= args.final_pool_branches <= len(searches):
        parser.error("invalid final pool branch count")
    per_crop_pool = fuse_branch_searches(
        searches[: args.pool_branches],
        rrf_k=args.pool_rrf_k,
        candidate_k=args.candidate_k,
    )
    merged = fuse_branch_searches(
        searches,
        rrf_k=args.branch_rrf_k,
        candidate_k=args.candidate_k,
        candidate_pool=per_crop_pool,
    )
    final_pool_search = fuse_branch_searches(
        searches[: args.final_pool_branches],
        rrf_k=args.pool_rrf_k,
        candidate_k=args.candidate_k,
    )

    spatial_index = load_spatial_profile_index(args.spatial_index)
    device = resolve_torch_device(args.device)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        spatial_index.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    spatial_records = select_spatial_records(context["records"])
    ordered_ids = sorted(spatial_records)
    profiles = encode_dino_row_profiles(
        model,
        [spatial_records[value] for value in ordered_ids],
        projection=spatial_index.projection,
        batch_size=args.batch_size,
        device=device,
    )
    query_profiles = dict(zip(ordered_ids, profiles, strict=True))

    baseline_cases, reranked_cases = rank_cases(
        context["examples"],
        context["records"],
        final_pool_search,
        merged,
        metadata,
        spatial_index,
        query_profiles,
        candidate_k=args.candidate_k,
        crop_rrf_k=args.crop_rrf_k,
        rerank_k=args.rerank_k,
        blend=args.blend,
        mode=args.mode,
        late_rrf_k=args.late_rrf_k,
    )
    payload = {
        "slot": "LEGS",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "branches": branches,
        "spatial_index": portable_path(args.spatial_index),
        "candidate_k": args.candidate_k,
        "branch_rrf_k": args.branch_rrf_k,
        "pool_branches": args.pool_branches,
        "final_pool_branches": args.final_pool_branches,
        "crop_rrf_k": args.crop_rrf_k,
        "rerank_k": args.rerank_k,
        "blend": args.blend,
        "mode": args.mode,
        "late_rrf_k": args.late_rrf_k,
        "baseline_summary": summarize_ranked_cases(baseline_cases, cutoffs),
        "reranked_summary": summarize_ranked_cases(reranked_cases, cutoffs),
        "baseline_cases": baseline_cases,
        "reranked_cases": reranked_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "baseline": payload["baseline_summary"], "reranked": payload["reranked_summary"]}, indent=2))
    return 0


def rank_cases(
    examples: list[dict[str, object]],
    records: dict[str, list[dict[str, object]]],
    pool_search: dict[str, list[dict[str, object]]],
    ensemble_search: dict[str, list[dict[str, object]]],
    metadata: list[dict[str, object]],
    spatial_index: object,
    query_profiles: dict[str, np.ndarray],
    *,
    candidate_k: int,
    crop_rrf_k: int,
    rerank_k: int,
    blend: float,
    mode: str,
    late_rrf_k: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    available_ids = available_pieces_by_slot(metadata)["LEGS"]
    available_keys = available_appearance_keys_by_slot(metadata)["LEGS"]
    baseline_cases = []
    reranked_cases = []
    for example in examples:
        example_id = str(example["example_id"])
        label = (example.get("core_slot_labels") or {}).get("LEGS")
        base = {
            "example_id": example_id,
            "glamour_id": int(example["glamour_id"]),
            "expected_name": label.get("item_name") if label else None,
            "rank": None,
        }
        status, expected_ids, expected_key = label_status(label, available_ids, available_keys)
        if status != "eligible" or not records.get(example_id):
            if status == "eligible":
                status = "no_usable_crop"
            row = {**base, "status": status}
            baseline_cases.append(row)
            reranked_cases.append(dict(row))
            continue
        pool_candidates = fuse_crop_results(
            records[example_id], pool_search, top_k=candidate_k, fusion="rrf", rrf_k=crop_rrf_k
        )
        ensemble_candidates = fuse_crop_results(
            records[example_id], ensemble_search, top_k=candidate_k, fusion="rrf", rrf_k=crop_rrf_k
        )
        candidates = constrain_to_final_candidate_pool(pool_candidates, ensemble_candidates)
        keys = {int(row["piece_id"]): appearance_name_key(row.get("name_en")) for row in candidates}
        baseline_ids = [int(row["piece_id"]) for row in candidates]
        baseline_rank = truth_rank(baseline_ids, expected_ids, expected_key, keys)
        reranked = candidates
        if example_id in query_profiles:
            spatial_source = pool_candidates if mode == "late-rrf" else candidates
            spatial_reranked = rerank_candidates_with_spatial_profile(
                spatial_source,
                spatial_index,
                query_profiles[example_id],
                rerank_k=rerank_k,
                blend=blend,
            )
            reranked = (
                fuse_final_candidate_rankings(
                    candidates, spatial_reranked, rrf_k=late_rrf_k
                )
                if mode == "late-rrf"
                else spatial_reranked
            )
        reranked_ids = [int(row["piece_id"]) for row in reranked]
        reranked_rank = truth_rank(reranked_ids, expected_ids, expected_key, keys)
        baseline_cases.append({**base, "status": "eligible", "rank": baseline_rank})
        reranked_cases.append({**base, "status": "eligible", "rank": reranked_rank})
    return baseline_cases, reranked_cases


def fuse_final_candidate_rankings(
    first: list[dict[str, object]],
    second: list[dict[str, object]],
    *,
    rrf_k: int,
) -> list[dict[str, object]]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    scores: dict[int, float] = {}
    rows: dict[int, dict[str, object]] = {}
    for ranking in (first, second):
        for offset, row in enumerate(ranking, start=1):
            piece_id = int(row["piece_id"])
            rank = int(row.get("rank", offset))
            scores[piece_id] = scores.get(piece_id, 0.0) + 1.0 / (rrf_k + rank)
            rows.setdefault(piece_id, dict(row))
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [
        {**rows[piece_id], "rank": rank, "score": scores[piece_id]}
        for rank, piece_id in enumerate(ordered, start=1)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
