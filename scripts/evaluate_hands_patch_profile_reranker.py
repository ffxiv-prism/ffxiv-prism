from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.patch_profiles import load_patch_profile_index
from ffxiv_prism.paths import portable_path
from scripts.evaluate_compact_rank_ensemble import (
    _fuse_all_examples,
    _fuse_all_query_crops,
    _rank_cases,
    load_compact_piece_ranks,
)
from scripts.evaluate_compact_rank_gate import find_domain_regressions
from scripts.evaluate_retrieval_funnel import (
    build_fold_context,
    summarize_ranked_cases,
    validate_cutoffs,
)


def percentile_by_piece(scores: Mapping[int, float]) -> dict[int, float]:
    ordered = sorted(scores, key=scores.get)
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: 1.0}
    denominator = len(ordered) - 1
    return {
        piece_id: rank / denominator
        for rank, piece_id in enumerate(ordered)
    }


def rerank_piece_ids(
    baseline_piece_ids: np.ndarray,
    patch_scores: Mapping[int, float],
    *,
    rerank_k: int,
    blend: float,
) -> np.ndarray:
    if rerank_k < 1:
        raise ValueError("rerank_k must be positive")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between zero and one")
    baseline = np.asarray(baseline_piece_ids, dtype=np.int64)
    leading = [int(value) for value in baseline[:rerank_k]]
    if len(leading) < 2:
        return baseline.copy()
    baseline_percentiles = {
        piece_id: 1.0 - rank / (len(leading) - 1)
        for rank, piece_id in enumerate(leading)
    }
    local_scores = {
        piece_id: float(patch_scores[piece_id])
        for piece_id in leading
        if piece_id in patch_scores
    }
    patch_percentiles = percentile_by_piece(local_scores)
    combined = {
        piece_id: (
            (1.0 - blend) * baseline_percentiles[piece_id]
            + blend * patch_percentiles.get(piece_id, 0.0)
        )
        for piece_id in leading
    }
    leading.sort(
        key=lambda piece_id: (
            combined[piece_id],
            baseline_percentiles[piece_id],
        ),
        reverse=True,
    )
    return np.concatenate(
        (np.asarray(leading, dtype=np.int64), baseline[len(leading) :])
    )


def patch_scores_by_piece(
    profile_scores: np.ndarray,
    *,
    metadata: Sequence[Mapping[str, Any]],
    metadata_indices: np.ndarray,
) -> dict[int, float]:
    output: dict[int, float] = {}
    for profile_index in np.flatnonzero(np.isfinite(profile_scores)):
        score = float(profile_scores[int(profile_index)])
        row = metadata[int(metadata_indices[int(profile_index)])]
        for piece in row.get("pieces") or []:
            piece_id = int(piece["piece_id"])
            output[piece_id] = max(output.get(piece_id, -float("inf")), score)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank a fixed HANDS baseline pool with cached local patch scores."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline-search-cache", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--patch-score-cache", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--branch-rrf-k", type=int, default=1)
    parser.add_argument("--crop-rrf-k", type=int, default=1)
    parser.add_argument("--rerank-k", type=int, nargs="+", default=[3, 5, 10, 20, 50])
    parser.add_argument(
        "--blend",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()

    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)
    if any(value < 1 or value > args.candidate_k for value in args.rerank_k):
        parser.error("--rerank-k values must be within candidate-k")
    if any(value < 0.0 or value > 1.0 for value in args.blend):
        parser.error("--blend values must be between zero and one")
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
        slot="HANDS",
    )

    paths, piece_ranks = load_compact_piece_ranks(
        args.baseline_search_cache,
        candidate_k=args.candidate_k,
    )
    path_index = {path: index for index, path in enumerate(paths)}
    crop_rankings = _fuse_all_query_crops(
        [piece_ranks],
        (1.0,),
        branch_rrf_k=args.branch_rrf_k,
        candidate_k=args.candidate_k,
    )
    baseline_by_example = _fuse_all_examples(
        context["records"],
        path_index,
        crop_rankings,
        crop_rrf_k=args.crop_rrf_k,
        candidate_k=args.candidate_k,
    )
    baseline_cases = _rank_cases(
        context["examples"],
        baseline_by_example,
        metadata,
        slot="HANDS",
    )

    patch_index = load_patch_profile_index(args.patch_index)
    with np.load(args.patch_score_cache, allow_pickle=False) as cache:
        example_ids = [str(value) for value in cache["example_ids"].tolist()]
        profile_score_matrix = np.asarray(cache["scores"], dtype=np.float32)
        cached_patch_index = str(cache["patch_index"].tolist())
    if Path(cached_patch_index).resolve() != args.patch_index.resolve():
        raise ValueError("Patch score cache belongs to a different patch index")
    if profile_score_matrix.shape != (
        len(example_ids),
        len(patch_index.profiles),
    ):
        raise ValueError("Patch score cache shape is inconsistent")
    patch_by_example = {
        example_id: patch_scores_by_piece(
            profile_score_matrix[row_index],
            metadata=metadata,
            metadata_indices=patch_index.metadata_indices,
        )
        for row_index, example_id in enumerate(example_ids)
    }

    experiments = []
    for rerank_k in args.rerank_k:
        for blend in args.blend:
            reranked_by_example = {
                example_id: rerank_piece_ids(
                    piece_ids,
                    patch_by_example.get(example_id, {}),
                    rerank_k=rerank_k,
                    blend=blend,
                )
                for example_id, piece_ids in baseline_by_example.items()
            }
            cases = _rank_cases(
                context["examples"],
                reranked_by_example,
                metadata,
                slot="HANDS",
            )
            experiments.append(
                {
                    "rerank_k": rerank_k,
                    "blend": blend,
                    "summary": summarize_ranked_cases(cases, cutoffs),
                    "domain_regressions": find_domain_regressions(
                        baseline_cases,
                        cases,
                        cutoffs,
                    ),
                    "cases": cases,
                }
            )
    safe = [row for row in experiments if not row["domain_regressions"]]
    safe.sort(
        key=lambda row: tuple(
            -int(row["summary"]["hits"][str(cutoff)]) for cutoff in cutoffs
        )
    )
    best_overall = min(
        experiments,
        key=lambda row: tuple(
            -int(row["summary"]["hits"][str(cutoff)]) for cutoff in cutoffs
        ),
    )
    payload = {
        "slot": "HANDS",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "metadata": portable_path(args.metadata),
        "baseline_search_cache": portable_path(args.baseline_search_cache),
        "patch_index": portable_path(args.patch_index),
        "patch_score_cache": portable_path(args.patch_score_cache),
        "candidate_k": args.candidate_k,
        "cutoffs": list(cutoffs),
        "baseline_summary": summarize_ranked_cases(baseline_cases, cutoffs),
        "best_overall": best_overall,
        "safe_policy_count": len(safe),
        "best_safe": safe[0] if safe else None,
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    compact = lambda row: (
        None
        if row is None
        else {key: value for key, value in row.items() if key != "cases"}
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "baseline_summary": payload["baseline_summary"],
                "best_overall": compact(best_overall),
                "safe_policy_count": len(safe),
                "best_safe": compact(safe[0] if safe else None),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
