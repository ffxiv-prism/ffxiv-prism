from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.paths import portable_path
from scripts.evaluate_compact_rank_ensemble import (
    _fuse_all_examples,
    _fuse_all_query_crops,
    _fuse_branch_rankings_by_example,
    _fuse_raw_crops_per_branch,
    _rank_cases,
    constrain_piece_ids_to_pool,
    load_compact_piece_ranks,
)
from scripts.evaluate_retrieval_funnel import (
    SLOTS,
    build_fold_context,
    summarize_ranked_cases,
    validate_cutoffs,
)


def piece_rank(piece_ids: np.ndarray, piece_id: int) -> int | None:
    positions = np.flatnonzero(np.asarray(piece_ids, dtype=np.int64) == int(piece_id))
    return int(positions[0]) + 1 if len(positions) else None


def promotion_gate_decision(
    base_piece_ids: np.ndarray,
    candidate_piece_ids: np.ndarray,
    last_branch_piece_ids: np.ndarray,
    *,
    max_base_rank: int,
    max_last_branch_rank: int,
    min_last_branch_rank_ratio: float,
    max_base_last_branch_rank: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Allow a reranker promotion only when both old and new evidence support it."""

    if max_base_rank < 1 or max_last_branch_rank < 1:
        raise ValueError("rank thresholds must be positive")
    if min_last_branch_rank_ratio < 1.0:
        raise ValueError("min_last_branch_rank_ratio must be at least one")
    if max_base_last_branch_rank is not None and max_base_last_branch_rank < 1:
        raise ValueError("max_base_last_branch_rank must be positive")
    if not len(base_piece_ids) or not len(candidate_piece_ids):
        return False, {"reason": "missing_ranking"}
    base_top1 = int(base_piece_ids[0])
    candidate_top1 = int(candidate_piece_ids[0])
    if base_top1 == candidate_top1:
        return False, {
            "reason": "unchanged_top1",
            "base_top1_piece_id": base_top1,
            "candidate_top1_piece_id": candidate_top1,
        }

    candidate_base_rank = piece_rank(base_piece_ids, candidate_top1)
    candidate_last_rank = piece_rank(last_branch_piece_ids, candidate_top1)
    base_last_rank = piece_rank(last_branch_piece_ids, base_top1)
    ratio_ready = (
        candidate_last_rank is not None
        and (
            base_last_rank is None
            or base_last_rank
            >= candidate_last_rank * min_last_branch_rank_ratio
        )
    )
    accepted = bool(
        candidate_base_rank is not None
        and candidate_base_rank <= max_base_rank
        and candidate_last_rank is not None
        and candidate_last_rank <= max_last_branch_rank
        and ratio_ready
        and (
            max_base_last_branch_rank is None
            or (
                base_last_rank is not None
                and base_last_rank <= max_base_last_branch_rank
            )
        )
    )
    return accepted, {
        "reason": "accepted" if accepted else "insufficient_agreement",
        "base_top1_piece_id": base_top1,
        "candidate_top1_piece_id": candidate_top1,
        "candidate_base_rank": candidate_base_rank,
        "candidate_last_branch_rank": candidate_last_rank,
        "base_last_branch_rank": base_last_rank,
    }


def apply_top1_promotion(
    baseline_piece_ids: np.ndarray,
    candidate_piece_ids: np.ndarray,
    accepted: bool,
) -> np.ndarray:
    """Promote only the candidate winner while preserving the baseline tail."""

    baseline = np.asarray(baseline_piece_ids, dtype=np.int64)
    candidate = np.asarray(candidate_piece_ids, dtype=np.int64)
    if not accepted or not len(baseline) or not len(candidate):
        return baseline.copy()
    promoted = int(candidate[0])
    if promoted == int(baseline[0]):
        return baseline.copy()
    return np.concatenate(
        (
            np.asarray([promoted], dtype=np.int64),
            baseline[baseline != promoted],
        )
    )[: len(baseline)]


def apply_top1_promotion_to_cases(
    baseline_cases: Sequence[Mapping[str, Any]],
    candidate_cases: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, bool],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Update truth ranks for a Top-1 move without rescanning full rankings."""

    candidate_by_id = {
        str(case["example_id"]): case for case in candidate_cases
    }
    output = []
    for baseline_case in baseline_cases:
        example_id = str(baseline_case["example_id"])
        row = dict(baseline_case)
        if not decisions.get(example_id, False):
            output.append(row)
            continue
        candidate_case = candidate_by_id[example_id]
        if candidate_case.get("rank") == 1:
            row["rank"] = 1
            output.append(row)
            continue
        baseline_rank = baseline_case.get("rank")
        promoted_base_rank = diagnostics[example_id].get("candidate_base_rank")
        if baseline_rank is not None and promoted_base_rank is not None:
            row["rank"] = int(baseline_rank) + int(
                int(promoted_base_rank) > int(baseline_rank)
            )
        output.append(row)
    return output


def find_domain_regressions(
    baseline_cases: Sequence[Mapping[str, Any]],
    candidate_cases: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int],
    *,
    dimensions: Sequence[str] = ("model_gender", "model_race"),
) -> list[dict[str, Any]]:
    baseline_by_id = {str(case["example_id"]): case for case in baseline_cases}
    candidate_by_id = {str(case["example_id"]): case for case in candidate_cases}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("baseline and candidate cases must contain identical examples")

    regressions = []
    for dimension in dimensions:
        grouped_ids: dict[str, list[str]] = defaultdict(list)
        for example_id, case in baseline_by_id.items():
            if case.get("status") == "eligible":
                grouped_ids[str(case.get(dimension) or "unknown")].append(example_id)
        for value in sorted(grouped_ids):
            example_ids = grouped_ids[value]
            for cutoff in cutoffs:
                baseline_hits = sum(
                    baseline_by_id[example_id].get("rank") is not None
                    and int(baseline_by_id[example_id]["rank"]) <= cutoff
                    for example_id in example_ids
                )
                candidate_hits = sum(
                    candidate_by_id[example_id].get("rank") is not None
                    and int(candidate_by_id[example_id]["rank"]) <= cutoff
                    for example_id in example_ids
                )
                if candidate_hits < baseline_hits:
                    regressions.append(
                        {
                            "dimension": dimension,
                            "value": value,
                            "cutoff": int(cutoff),
                            "baseline_hits": baseline_hits,
                            "candidate_hits": candidate_hits,
                        }
                    )
    return regressions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Screen a label-agnostic agreement gate for the last branch of a "
            "compact rank ensemble."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--branch-search-cache", action="append", type=Path, required=True
    )
    parser.add_argument("--candidate-last-branch-weight", type=float, default=0.8)
    parser.add_argument(
        "--baseline-branch-weights",
        type=float,
        nargs="+",
        help="Optional full branch-weight vector for the baseline ensemble.",
    )
    parser.add_argument(
        "--candidate-branch-weights",
        type=float,
        nargs="+",
        help="Optional full branch-weight vector for the candidate ensemble.",
    )
    parser.add_argument(
        "--fusion-order",
        choices=("branch-then-crop", "crop-then-branch"),
        default="branch-then-crop",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("full-ranking", "top1-only"),
        default="full-ranking",
        help="Adopt the full candidate ranking or only its promoted Top-1.",
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--slot", choices=SLOTS, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--branch-rrf-k", type=int, default=1)
    parser.add_argument("--crop-rrf-k", type=int, default=1)
    parser.add_argument("--final-pool-branches", type=int, default=2)
    parser.add_argument(
        "--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000]
    )
    parser.add_argument(
        "--max-base-ranks",
        type=int,
        nargs="+",
        default=[2, 3, 5, 10, 20, 50, 100, 200, 500, 1000],
    )
    parser.add_argument(
        "--max-last-branch-ranks",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 10, 20, 50, 1000],
    )
    parser.add_argument(
        "--min-last-branch-rank-ratios",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 10.0],
    )
    parser.add_argument(
        "--max-base-last-branch-ranks",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Optional caps for the baseline Top-1 rank in the last branch; "
            "zero disables this extra conservative agreement condition."
        ),
    )
    args = parser.parse_args()

    branch_count = len(args.branch_search_cache)
    if branch_count < 2:
        parser.error("agreement gating requires at least two branches")
    if not 1 <= args.final_pool_branches <= branch_count:
        parser.error("final pool branch count must be within the ensemble")
    if args.candidate_last_branch_weight <= 0:
        parser.error("candidate last-branch weight must be positive")
    if bool(args.baseline_branch_weights) != bool(args.candidate_branch_weights):
        parser.error(
            "--baseline-branch-weights and --candidate-branch-weights are paired"
        )
    explicit_weights = args.baseline_branch_weights is not None
    if explicit_weights:
        if len(args.baseline_branch_weights) != branch_count or len(
            args.candidate_branch_weights
        ) != branch_count:
            parser.error("Explicit branch weights must align with every cache")
        if any(weight < 0 for weight in args.baseline_branch_weights) or not any(
            args.baseline_branch_weights
        ):
            parser.error("Baseline branch weights must be non-negative and nonzero")
        if any(weight < 0 for weight in args.candidate_branch_weights) or not any(
            args.candidate_branch_weights
        ):
            parser.error("Candidate branch weights must be non-negative and nonzero")
        baseline_weights = tuple(args.baseline_branch_weights)
        candidate_weights = tuple(args.candidate_branch_weights)
    else:
        if args.final_pool_branches >= branch_count:
            parser.error(
                "legacy gate mode requires a pool using fewer than all branches"
            )
        baseline_weights = (1.0,) * args.final_pool_branches + (0.0,) * (
            branch_count - args.final_pool_branches
        )
        candidate_weights = (1.0,) * (branch_count - 1) + (
            args.candidate_last_branch_weight,
        )
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)

    cached = [
        load_compact_piece_ranks(
            path,
            candidate_k=args.candidate_k,
            include_scores=args.fusion_order == "crop-then-branch",
        )
        for path in args.branch_search_cache
    ]
    paths = cached[0][0]
    if any(branch[0] != paths for branch in cached[1:]):
        raise ValueError("Compact branch caches do not contain identical query paths")
    piece_ranks = [branch[1] for branch in cached]
    piece_scores = [branch[2] for branch in cached] if len(cached[0]) == 3 else []
    path_index = {path: index for index, path in enumerate(paths)}

    metadata = [
        row for row in read_jsonl(args.metadata) if str(row.get("slot")) == args.slot
    ]
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
        slot=args.slot,
    )

    if args.fusion_order == "crop-then-branch":
        branch_by_example = _fuse_raw_crops_per_branch(
            piece_ranks,
            piece_scores,
            records_by_example=context["records"],
            path_index=path_index,
            candidate_k=args.candidate_k,
        )
        pool_by_example = _fuse_branch_rankings_by_example(
            branch_by_example[: args.final_pool_branches],
            (1.0,) * args.final_pool_branches,
            records_by_example=context["records"],
            branch_rrf_k=args.branch_rrf_k,
            candidate_k=args.candidate_k,
        )
        base_reranked_by_example = _fuse_branch_rankings_by_example(
            branch_by_example,
            baseline_weights,
            records_by_example=context["records"],
            branch_rrf_k=args.branch_rrf_k,
            candidate_k=args.candidate_k,
        )
        candidate_reranked_by_example = _fuse_branch_rankings_by_example(
            branch_by_example,
            candidate_weights,
            records_by_example=context["records"],
            branch_rrf_k=args.branch_rrf_k,
            candidate_k=args.candidate_k,
        )
        last_branch_by_example = branch_by_example[-1]
    else:
        pool_crop_rankings = _fuse_all_query_crops(
            piece_ranks[: args.final_pool_branches],
            (1.0,) * args.final_pool_branches,
            branch_rrf_k=args.branch_rrf_k,
            candidate_k=args.candidate_k,
        )
        pool_by_example = _fuse_all_examples(
            context["records"],
            path_index,
            pool_crop_rankings,
            crop_rrf_k=args.crop_rrf_k,
            candidate_k=args.candidate_k,
        )
        base_crop_rankings = _fuse_all_query_crops(
            piece_ranks,
            baseline_weights,
            branch_rrf_k=args.branch_rrf_k,
            candidate_k=args.candidate_k,
        )
        base_reranked_by_example = _fuse_all_examples(
            context["records"],
            path_index,
            base_crop_rankings,
            crop_rrf_k=args.crop_rrf_k,
            candidate_k=args.candidate_k,
        )
        candidate_crop_rankings = _fuse_all_query_crops(
            piece_ranks,
            candidate_weights,
            branch_rrf_k=args.branch_rrf_k,
            candidate_k=args.candidate_k,
        )
        candidate_reranked_by_example = _fuse_all_examples(
            context["records"],
            path_index,
            candidate_crop_rankings,
            crop_rrf_k=args.crop_rrf_k,
            candidate_k=args.candidate_k,
        )
        last_branch_by_example = _fuse_all_examples(
            context["records"],
            path_index,
            piece_ranks[-1],
            crop_rrf_k=args.crop_rrf_k,
            candidate_k=args.candidate_k,
        )
    base_by_example = {
        example_id: constrain_piece_ids_to_pool(
            pool_ids,
            base_reranked_by_example.get(
                example_id, np.empty(0, dtype=np.int64)
            ),
        )
        for example_id, pool_ids in pool_by_example.items()
    }
    candidate_by_example = {
        example_id: constrain_piece_ids_to_pool(
            pool_by_example.get(example_id, np.empty(0, dtype=np.int64)),
            candidate_reranked_by_example.get(
                example_id, np.empty(0, dtype=np.int64)
            ),
        )
        for example_id in base_by_example
    }

    baseline_cases = _rank_cases(
        context["examples"], base_by_example, metadata, slot=args.slot
    )
    candidate_cases = _rank_cases(
        context["examples"], candidate_by_example, metadata, slot=args.slot
    )
    candidate_case_by_id = {
        str(case["example_id"]): case for case in candidate_cases
    }

    experiments = []
    policy_count = (
        len(args.max_base_ranks)
        * len(args.max_last_branch_ranks)
        * len(args.min_last_branch_rank_ratios)
        * len(args.max_base_last_branch_ranks)
    )
    for max_base_rank in args.max_base_ranks:
        for max_last_rank in args.max_last_branch_ranks:
            for min_ratio in args.min_last_branch_rank_ratios:
                for raw_base_last_cap in args.max_base_last_branch_ranks:
                    base_last_cap = (
                        None if raw_base_last_cap == 0 else raw_base_last_cap
                    )
                    decisions = {}
                    diagnostics = {}
                    for example_id, base_ids in base_by_example.items():
                        accepted, diagnostic = promotion_gate_decision(
                            base_ids,
                            candidate_by_example.get(
                                example_id, np.empty(0, dtype=np.int64)
                            ),
                            last_branch_by_example.get(
                                example_id, np.empty(0, dtype=np.int64)
                            ),
                            max_base_rank=max_base_rank,
                            max_last_branch_rank=max_last_rank,
                            min_last_branch_rank_ratio=min_ratio,
                            max_base_last_branch_rank=base_last_cap,
                        )
                        decisions[example_id] = accepted
                        diagnostics[example_id] = diagnostic
                    if args.gate_mode == "top1-only":
                        gated_cases = apply_top1_promotion_to_cases(
                            baseline_cases,
                            candidate_cases,
                            decisions,
                            diagnostics,
                        )
                    else:
                        gated_cases = []
                        for baseline_case in baseline_cases:
                            example_id = str(baseline_case["example_id"])
                            selected = (
                                candidate_case_by_id[example_id]
                                if decisions.get(example_id, False)
                                else baseline_case
                            )
                            gated_cases.append(dict(selected))
                    regressions = find_domain_regressions(
                        baseline_cases, gated_cases, cutoffs
                    )
                    row = {
                        "policy": {
                            "max_base_rank": max_base_rank,
                            "max_last_branch_rank": max_last_rank,
                            "min_last_branch_rank_ratio": min_ratio,
                            "max_base_last_branch_rank": base_last_cap,
                        },
                        "accepted_change_count": sum(decisions.values()),
                        "summary": summarize_ranked_cases(gated_cases, cutoffs),
                        "domain_regressions": regressions,
                    }
                    if policy_count == 1:
                        row["cases"] = gated_cases
                        row["gate_diagnostics"] = diagnostics
                    experiments.append(row)

    safe = [row for row in experiments if not row["domain_regressions"]]
    safe.sort(
        key=lambda row: (
            tuple(
                -int(row["summary"]["hits"][str(cutoff)])
                for cutoff in cutoffs
            ),
            int(row["accepted_change_count"]),
        )
    )
    payload = {
        "slot": args.slot,
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "metadata": portable_path(args.metadata),
        "branch_search_caches": [
            portable_path(path) for path in args.branch_search_cache
        ],
        "candidate_last_branch_weight": args.candidate_last_branch_weight,
        "baseline_branch_weights": list(baseline_weights),
        "candidate_branch_weights": list(candidate_weights),
        "fusion_order": args.fusion_order,
        "gate_mode": args.gate_mode,
        "candidate_k": args.candidate_k,
        "branch_rrf_k": args.branch_rrf_k,
        "crop_rrf_k": args.crop_rrf_k,
        "final_pool_branches": args.final_pool_branches,
        "cutoffs": list(cutoffs),
        "baseline_summary": summarize_ranked_cases(baseline_cases, cutoffs),
        "ungated_candidate_summary": summarize_ranked_cases(candidate_cases, cutoffs),
        "safe_policy_count": len(safe),
        "best_safe_policy": safe[0] if safe else None,
        "experiments": experiments,
    }
    if policy_count == 1:
        payload["baseline_cases"] = baseline_cases
        payload["ungated_candidate_cases"] = candidate_cases
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": portable_path(args.output),
                "baseline_summary": payload["baseline_summary"],
                "ungated_candidate_summary": payload["ungated_candidate_summary"],
                "safe_policy_count": payload["safe_policy_count"],
                "best_safe_policy": (
                    {
                        key: value
                        for key, value in payload["best_safe_policy"].items()
                        if key not in {"cases", "gate_diagnostics"}
                    }
                    if payload["best_safe_policy"]
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
