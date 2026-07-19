from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

np.linalg.inv(np.eye(2, dtype=np.float32))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.head_morphology import (
    rerank_candidates_by_covered_probability,
    rerank_candidates_by_enclosing_probability,
)
from ffxiv_prism.matching.inference import (
    HEAD_PROMOTED_CANDIDATE_CROP_ROUTES,
    fuse_crop_results,
)
from ffxiv_prism.paths import portable_path
from scripts.evaluate_cached_query_retrieval import load_query_embeddings
from scripts.evaluate_glamour_embedding_model import search_crops
from scripts.evaluate_head_coverage_reranker import (
    label_status,
    truth_rank,
)
from scripts.evaluate_rank_ensemble_retrieval import fuse_branch_searches
from scripts.evaluate_retrieval_funnel import (
    appearance_name_key,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    build_fold_context,
    summarize_ranked_cases,
    validate_cutoffs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a train-only enclosing/non-enclosing HEAD router."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--branch", action="append", nargs=2, metavar=("INDEX_DIR", "QUERY_CACHE"), required=True
    )
    parser.add_argument("--morphology-predictions", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument("--minimum-confidence", type=float)
    parser.add_argument("--branch-rrf-k", type=int, default=10)
    parser.add_argument("--pool-branches", type=int, default=2)
    parser.add_argument("--pool-rrf-k", type=int, default=1)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)

    metadata = read_jsonl(args.metadata)
    examples = read_jsonl(args.examples)
    quality_by_id = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
    }
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality_by_id,
        regions_dir=args.regions_dir,
        slot="HEAD",
    )
    records = {
        example_id: [
            row for row in rows if str(row.get("name")) == "head_upper_accessory"
        ]
        for example_id, rows in context["records"].items()
    }
    expected_paths = [str(row["local_path"]) for row in metadata]
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
                records,
                load_query_embeddings(cache_path),
                args.candidate_k,
                device=args.device,
                candidate_crop_routes=HEAD_PROMOTED_CANDIDATE_CROP_ROUTES,
            )
        )
        branches.append(
            {"index_dir": portable_path(index_dir), "query_cache": portable_path(cache_path)}
        )
    if not 1 <= args.pool_branches <= len(searches):
        parser.error("--pool-branches exceeds available branches")
    candidate_pool = fuse_branch_searches(
        searches[: args.pool_branches],
        rrf_k=args.pool_rrf_k,
        candidate_k=args.candidate_k,
    )
    merged = fuse_branch_searches(
        searches,
        rrf_k=args.branch_rrf_k,
        candidate_k=args.candidate_k,
        candidate_pool=candidate_pool,
    )
    prediction_payload = json.loads(
        args.morphology_predictions.read_text(encoding="utf-8")
    )
    minimum_confidence = (
        args.minimum_confidence
        if args.minimum_confidence is not None
        else float(prediction_payload["selected_minimum_confidence"])
    )
    probability_key = (
        "covered"
        if "covered" in prediction_payload["classes"]
        else "enclosing"
    )
    morphology_probability_by_id = {
        str(row["example_id"]): float(row["probabilities"][probability_key])
        for row in prediction_payload["test_predictions"]
    }
    baseline, reranked, diagnostics = rank_cases(
        context["examples"],
        records,
        merged,
        metadata,
        morphology_probability_by_id,
        probability_kind=probability_key,
        rerank_k=args.rerank_k,
        minimum_confidence=minimum_confidence,
    )
    payload = {
        "slot": "HEAD",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "branches": branches,
        "morphology_predictions": portable_path(args.morphology_predictions),
        "candidate_k": args.candidate_k,
        "rerank_k": args.rerank_k,
        "minimum_confidence": minimum_confidence,
        "branch_rrf_k": args.branch_rrf_k,
        "pool_branches": args.pool_branches,
        "pool_rrf_k": args.pool_rrf_k,
        "baseline_summary": summarize_ranked_cases(baseline, cutoffs),
        "reranked_summary": summarize_ranked_cases(reranked, cutoffs),
        "baseline_cases": baseline,
        "reranked_cases": reranked,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "baseline": payload["baseline_summary"], "reranked": payload["reranked_summary"]}, indent=2))
    return 0


def rank_cases(
    examples: Sequence[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    search: dict[str, list[dict[str, Any]]],
    metadata: list[dict[str, Any]],
    morphology_probability_by_id: dict[str, float],
    *,
    probability_kind: str,
    rerank_k: int,
    minimum_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    available_ids = available_pieces_by_slot(metadata)["HEAD"]
    available_keys = available_appearance_keys_by_slot(metadata)["HEAD"]
    baseline_cases = []
    reranked_cases = []
    diagnostics = []
    for example in examples:
        example_id = str(example["example_id"])
        label = (example.get("core_slot_labels") or {}).get("HEAD")
        base = {
            "example_id": example_id,
            "glamour_id": int(example["glamour_id"]),
            "expected_name": label.get("item_name") if label else None,
            "rank": None,
        }
        status, expected_ids, expected_key = label_status(
            label, available_ids, available_keys
        )
        if status != "eligible" or not records.get(example_id):
            if status == "eligible":
                status = "no_usable_crop"
            row = {**base, "status": status}
            baseline_cases.append(row)
            reranked_cases.append(dict(row))
            continue
        candidates = fuse_crop_results(records[example_id], search, top_k=1000, fusion="max")
        keys = {
            int(row["piece_id"]): appearance_name_key(row.get("name_en"))
            for row in candidates
        }
        baseline_ids = [int(row["piece_id"]) for row in candidates]
        baseline_rank = truth_rank(baseline_ids, expected_ids, expected_key, keys)
        probability = morphology_probability_by_id.get(example_id, 0.5)
        rerank = (
            rerank_candidates_by_covered_probability
            if probability_kind == "covered"
            else rerank_candidates_by_enclosing_probability
        )
        routed = rerank(
            candidates, probability, rerank_k=rerank_k, minimum_confidence=minimum_confidence
        )
        routed_ids = [int(row["piece_id"]) for row in routed]
        routed_rank = truth_rank(routed_ids, expected_ids, expected_key, keys)
        baseline_cases.append({**base, "status": status, "rank": baseline_rank})
        reranked_cases.append({**base, "status": status, "rank": routed_rank})
        diagnostics.append(
            {
                "example_id": example_id,
                f"{probability_kind}_probability": probability,
                "baseline_top_piece_id": baseline_ids[0] if baseline_ids else None,
                "reranked_top_piece_id": routed_ids[0] if routed_ids else None,
                "truth_baseline_rank": baseline_rank,
                "truth_reranked_rank": routed_rank,
            }
        )
    return baseline_cases, reranked_cases, diagnostics


if __name__ == "__main__":
    raise SystemExit(main())
