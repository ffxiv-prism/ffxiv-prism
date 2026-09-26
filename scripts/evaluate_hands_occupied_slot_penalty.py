from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.paths import portable_path
from scripts.evaluate_compact_rank_ensemble import (
    _fuse_all_examples,
    _rank_cases,
    load_compact_piece_ranks,
)
from scripts.evaluate_compact_rank_gate import find_domain_regressions
from scripts.evaluate_retrieval_funnel import (
    build_fold_context,
    summarize_ranked_cases,
    validate_cutoffs,
)
from scripts.train_glamour_query_encoder import load_slot_eligibility


def exact_slot_piece_ids(metadata: Sequence[dict[str, Any]]) -> set[int]:
    return {
        int(piece["piece_id"])
        for row in metadata
        for piece in row.get("pieces", [])
        if str(piece.get("link_type")) == "exact_slot"
    }


def rerank_occupied_slot_pieces(
    piece_ids: np.ndarray,
    exact_piece_ids: set[int],
    *,
    occupied_multiplier: float,
    rrf_k: int,
) -> np.ndarray:
    if occupied_multiplier < 0:
        raise ValueError("occupied_multiplier must be non-negative")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    values = np.asarray(piece_ids, dtype=np.int64)
    valid = values >= 0
    output = values.copy()
    active = values[valid]
    if not len(active):
        return output
    ranks = np.arange(1, len(active) + 1, dtype=np.float64)
    multipliers = np.fromiter(
        (1.0 if int(piece_id) in exact_piece_ids else occupied_multiplier for piece_id in active),
        dtype=np.float64,
        count=len(active),
    )
    scores = multipliers / (rrf_k + ranks)
    order = np.lexsort((np.arange(len(active)), -scores))
    output[: len(active)] = active[order]
    return output


def gate_occupied_top1_promotion(
    base_piece_ids: np.ndarray,
    candidate_piece_ids: np.ndarray,
    exact_piece_ids: set[int],
    *,
    max_base_rank: int,
) -> tuple[np.ndarray, bool]:
    base = np.asarray(base_piece_ids, dtype=np.int64)
    candidate = np.asarray(candidate_piece_ids, dtype=np.int64)
    if max_base_rank < 1:
        raise ValueError("max_base_rank must be positive")
    if not len(base) or not len(candidate):
        return base.copy(), False
    base_top1 = int(base[0])
    candidate_top1 = int(candidate[0])
    if (
        base_top1 in exact_piece_ids
        or candidate_top1 not in exact_piece_ids
        or candidate_top1 == base_top1
    ):
        return base.copy(), False
    positions = np.flatnonzero(base == candidate_top1)
    if not len(positions) or int(positions[0]) + 1 > max_base_rank:
        return base.copy(), False
    promoted = np.concatenate(
        (np.asarray([candidate_top1], dtype=np.int64), base[base != candidate_top1])
    )
    return promoted, True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen a rank penalty for multi-slot HANDS occupied-slot pieces."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--search-cache", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    quality_group = parser.add_mutually_exclusive_group(required=True)
    quality_group.add_argument("--quality-manifest", type=Path)
    quality_group.add_argument("--slot-eligibility-manifest", type=Path)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--piece-rrf-k", type=int, default=1)
    parser.add_argument("--crop-rrf-k", type=int, default=1)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument(
        "--occupied-multipliers",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--gate-max-base-ranks", type=int, nargs="+", default=[2, 3, 5, 10]
    )
    args = parser.parse_args()
    if any(value < 0 for value in args.occupied_multipliers):
        parser.error("occupied multipliers must be non-negative")
    if any(value < 1 for value in args.gate_max_base_ranks):
        parser.error("gate max-base ranks must be positive")
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)

    metadata = [
        row for row in read_jsonl(args.metadata) if str(row.get("slot")) == "HANDS"
    ]
    exact_ids = exact_slot_piece_ids(metadata)
    all_ids = {
        int(piece["piece_id"])
        for row in metadata
        for piece in row.get("pieces", [])
    }
    occupied_only_ids = all_ids - exact_ids
    examples = read_jsonl(args.examples)
    if args.quality_manifest:
        quality = {
            str(row["example_id"]): bool(row.get("passed"))
            for row in read_jsonl(args.quality_manifest)
        }
    else:
        eligibility = load_slot_eligibility(args.slot_eligibility_manifest)
        quality = {
            str(example["example_id"]): (
                "HANDS"
                in eligibility.get(str(example["example_id"]), set())
            )
            for example in examples
        }
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality,
        regions_dir=args.regions_dir,
        slot="HANDS",
    )
    paths, piece_ranks = load_compact_piece_ranks(
        args.search_cache, candidate_k=args.candidate_k
    )
    path_index = {path: index for index, path in enumerate(paths)}

    def build_rankings(multiplier: float) -> dict[str, np.ndarray]:
        reranked = [
            rerank_occupied_slot_pieces(
                ranking,
                exact_ids,
                occupied_multiplier=float(multiplier),
                rrf_k=args.piece_rrf_k,
            )
            for ranking in piece_ranks
        ]
        return _fuse_all_examples(
            context["records"],
            path_index,
            reranked,
            crop_rrf_k=args.crop_rrf_k,
            candidate_k=args.candidate_k,
        )

    def evaluate(multiplier: float) -> dict[str, Any]:
        rankings = build_rankings(multiplier)
        cases = _rank_cases(context["examples"], rankings, metadata, slot="HANDS")
        return {
            "occupied_multiplier": float(multiplier),
            "summary": summarize_ranked_cases(cases, cutoffs),
            "cases": cases,
            "rankings": rankings,
        }

    baseline = evaluate(1.0)
    trials = []
    safe = []
    gated_trials = []
    safe_gates = []
    for multiplier in dict.fromkeys(float(value) for value in args.occupied_multipliers):
        result = evaluate(multiplier)
        regressions = find_domain_regressions(
            baseline["cases"], result["cases"], cutoffs
        )
        cutoff_regressions = [
            {
                "cutoff": cutoff,
                "baseline_hits": baseline["summary"]["hits"][str(cutoff)],
                "candidate_hits": result["summary"]["hits"][str(cutoff)],
            }
            for cutoff in cutoffs
            if result["summary"]["hits"][str(cutoff)]
            < baseline["summary"]["hits"][str(cutoff)]
        ]
        compact = {
            "occupied_multiplier": multiplier,
            "summary": result["summary"],
            "domain_regressions": regressions,
            "cutoff_regressions": cutoff_regressions,
        }
        trials.append(compact)
        if not regressions and not cutoff_regressions:
            safe.append({**compact, "cases": result["cases"]})
        if multiplier == 1.0:
            continue
        for max_base_rank in dict.fromkeys(args.gate_max_base_ranks):
            accepted_change_count = 0
            gated_rankings = {}
            for example_id, base_ranking in baseline["rankings"].items():
                promoted, accepted = gate_occupied_top1_promotion(
                    base_ranking,
                    result["rankings"].get(
                        example_id, np.empty(0, dtype=np.int64)
                    ),
                    exact_ids,
                    max_base_rank=max_base_rank,
                )
                gated_rankings[example_id] = promoted
                accepted_change_count += int(accepted)
            gated_cases = _rank_cases(
                context["examples"], gated_rankings, metadata, slot="HANDS"
            )
            gated_summary = summarize_ranked_cases(gated_cases, cutoffs)
            gated_domain_regressions = find_domain_regressions(
                baseline["cases"], gated_cases, cutoffs
            )
            gated_cutoff_regressions = [
                {
                    "cutoff": cutoff,
                    "baseline_hits": baseline["summary"]["hits"][str(cutoff)],
                    "candidate_hits": gated_summary["hits"][str(cutoff)],
                }
                for cutoff in cutoffs
                if gated_summary["hits"][str(cutoff)]
                < baseline["summary"]["hits"][str(cutoff)]
            ]
            gated_compact = {
                "occupied_multiplier": multiplier,
                "max_base_rank": int(max_base_rank),
                "accepted_change_count": accepted_change_count,
                "summary": gated_summary,
                "domain_regressions": gated_domain_regressions,
                "cutoff_regressions": gated_cutoff_regressions,
            }
            gated_trials.append(gated_compact)
            if not gated_domain_regressions and not gated_cutoff_regressions:
                safe_gates.append({**gated_compact, "cases": gated_cases})

    best_safe = max(
        safe,
        key=lambda row: (
            row["summary"]["hits"]["1"],
            row["summary"]["hits"]["10"],
            row["summary"]["hits"]["100"],
            row["summary"]["mrr"],
            row["occupied_multiplier"],
        ),
    ) if safe else None
    best_safe_gate = max(
        safe_gates,
        key=lambda row: (
            row["summary"]["hits"]["1"],
            row["summary"]["hits"]["10"],
            row["summary"]["hits"]["100"],
            row["summary"]["mrr"],
            -row["accepted_change_count"],
            -row["max_base_rank"],
            row["occupied_multiplier"],
        ),
    ) if safe_gates else None
    baseline.pop("rankings")
    payload = {
        "slot": "HANDS",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "metadata": portable_path(args.metadata),
        "search_cache": portable_path(args.search_cache),
        "candidate_k": args.candidate_k,
        "piece_rrf_k": args.piece_rrf_k,
        "crop_rrf_k": args.crop_rrf_k,
        "cutoffs": list(cutoffs),
        "exact_slot_piece_count": len(exact_ids),
        "occupied_only_piece_count": len(occupied_only_ids),
        "baseline": baseline,
        "trials": trials,
        "safe_policy_count": len(safe),
        "best_safe": best_safe,
        "gated_trials": gated_trials,
        "safe_gate_count": len(safe_gates),
        "best_safe_gate": best_safe_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": portable_path(args.output),
                "exact_slot_piece_count": len(exact_ids),
                "occupied_only_piece_count": len(occupied_only_ids),
                "baseline_summary": baseline["summary"],
                "safe_policy_count": len(safe),
                "best_safe": (
                    {
                        "occupied_multiplier": best_safe["occupied_multiplier"],
                        "summary": best_safe["summary"],
                    }
                    if best_safe
                    else None
                ),
                "safe_gate_count": len(safe_gates),
                "best_safe_gate": (
                    {
                        "occupied_multiplier": best_safe_gate["occupied_multiplier"],
                        "max_base_rank": best_safe_gate["max_base_rank"],
                        "accepted_change_count": best_safe_gate[
                            "accepted_change_count"
                        ],
                        "summary": best_safe_gate["summary"],
                    }
                    if best_safe_gate
                    else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
