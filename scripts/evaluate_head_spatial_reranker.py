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
from ffxiv_prism.matching.inference import (
    HEAD_PROMOTED_CANDIDATE_CROP_ROUTES,
    fuse_crop_results,
)
from ffxiv_prism.matching.coverage_profiles import (
    encode_head_parsing_spatial_profiles,
)
from ffxiv_prism.matching.spatial_profiles import (
    encode_dino_row_profiles,
    load_spatial_profile_index,
    rerank_candidates_with_spatial_profile,
)
from ffxiv_prism.paths import CACHE_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.human_parsing import HumanParser
from scripts.evaluate_cached_query_retrieval import load_query_embeddings
from scripts.evaluate_glamour_embedding_model import search_crops
from scripts.evaluate_head_coverage_reranker import (
    label_status,
    select_coverage_records,
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
        description="Re-rank HEAD candidates with grayscale DINO row profiles."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--branch",
        action="append",
        nargs=2,
        metavar=("INDEX_DIR", "QUERY_CACHE"),
        required=True,
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
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument("--blend", type=float, default=0.35)
    parser.add_argument("--branch-rrf-k", type=int, default=1)
    parser.add_argument("--pool-branches", type=int)
    parser.add_argument("--pool-rrf-k", type=int, default=1)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()
    if not 0.0 <= args.blend <= 1.0:
        parser.error("--blend must be between zero and one")
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
    retrieval_records = {
        example_id: [
            row for row in rows if str(row.get("name")) == "head_upper_accessory"
        ]
        for example_id, rows in context["records"].items()
    }
    expected_paths = [str(row["local_path"]) for row in metadata]
    searches = []
    branch_manifest = []
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
                retrieval_records,
                load_query_embeddings(cache_path),
                args.candidate_k,
                device=args.device,
                candidate_crop_routes=HEAD_PROMOTED_CANDIDATE_CROP_ROUTES,
            )
        )
        branch_manifest.append(
            {
                "index_dir": portable_path(index_dir),
                "query_cache": portable_path(cache_path),
            }
        )
    candidate_pool = None
    if args.pool_branches is not None:
        if not 1 <= args.pool_branches <= len(searches):
            parser.error("--pool-branches must select at least one available branch")
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

    spatial_index = load_spatial_profile_index(args.spatial_index)
    spatial_records = select_coverage_records(context["records"])
    ordered_ids = sorted(spatial_records)
    if spatial_index.preprocess.startswith("human-parsing-head-layout-"):
        profiles = encode_head_parsing_spatial_profiles(
            HumanParser(model_id=spatial_index.model_id, device=args.device),
            [spatial_records[value] for value in ordered_ids],
            batch_size=args.batch_size,
            grid_size=int(spatial_index.preprocess.split("-")[-2].split("x")[0]),
        )
    else:
        device = resolve_torch_device(args.device)
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            spatial_index.model_id,
            cache_dir=CACHE_DIR / "huggingface" / "hub",
            local_files_only=True,
        ).to(device)
        profiles = encode_dino_row_profiles(
            model,
            [spatial_records[value] for value in ordered_ids],
            projection=spatial_index.projection,
            batch_size=args.batch_size,
            device=device,
        )
    query_profiles = dict(zip(ordered_ids, profiles, strict=True))
    profile_indices_by_piece: dict[int, list[int]] = defaultdict(list)
    for profile_index, piece_ids in enumerate(spatial_index.piece_ids_by_profile):
        for piece_id in piece_ids:
            profile_indices_by_piece[piece_id].append(profile_index)

    baseline_cases, reranked_cases, diagnostics = rank_cases(
        context["examples"],
        retrieval_records,
        merged,
        metadata,
        spatial_index,
        query_profiles,
        rerank_k=args.rerank_k,
        blend=args.blend,
    )
    payload = {
        "slot": "HEAD",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "metadata": portable_path(args.metadata),
        "branches": branch_manifest,
        "spatial_index": portable_path(args.spatial_index),
        "candidate_k": args.candidate_k,
        "rerank_k": args.rerank_k,
        "blend": args.blend,
        "pool_branches": args.pool_branches,
        "pool_rrf_k": args.pool_rrf_k,
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


def rank_cases(
    examples: Sequence[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    search: dict[str, list[dict[str, Any]]],
    metadata: list[dict[str, Any]],
    spatial_index: Any,
    query_profiles: dict[str, np.ndarray],
    *,
    rerank_k: int,
    blend: float,
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
            "model_gender": example.get("model_gender"),
            "model_race": example.get("model_race"),
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
        candidates = fuse_crop_results(
            records[example_id], search, top_k=1000, fusion="max"
        )
        keys = {
            int(row["piece_id"]): appearance_name_key(row.get("name_en"))
            for row in candidates
        }
        baseline_ids = [int(row["piece_id"]) for row in candidates]
        baseline_rank = truth_rank(baseline_ids, expected_ids, expected_key, keys)
        reranked = candidates
        if example_id in query_profiles:
            reranked = rerank_candidates_with_spatial_profile(
                candidates,
                spatial_index,
                query_profiles[example_id],
                rerank_k=rerank_k,
                blend=blend,
            )
        reranked_ids = [int(row["piece_id"]) for row in reranked]
        reranked_rank = truth_rank(reranked_ids, expected_ids, expected_key, keys)
        baseline_cases.append({**base, "status": status, "rank": baseline_rank})
        reranked_cases.append({**base, "status": status, "rank": reranked_rank})
        diagnostics.append(
            {
                "example_id": example_id,
                "baseline_top_piece_id": baseline_ids[0] if baseline_ids else None,
                "reranked_top_piece_id": reranked_ids[0] if reranked_ids else None,
                "truth_baseline_rank": baseline_rank,
                "truth_reranked_rank": reranked_rank,
            }
        )
    return baseline_cases, reranked_cases, diagnostics


if __name__ == "__main__":
    raise SystemExit(main())
