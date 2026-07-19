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
from ffxiv_prism.matching.inference import crop_weight
from ffxiv_prism.paths import portable_path
from scripts.evaluate_compact_rank_ensemble import (
    _rank_cases,
    load_compact_piece_ranks,
    stable_weighted_rrf,
)
from scripts.evaluate_compact_rank_gate import find_domain_regressions
from scripts.evaluate_retrieval_funnel import (
    build_fold_context,
    summarize_ranked_cases,
    validate_cutoffs,
)


def fuse_examples_with_role_multipliers(
    records_by_example: Mapping[str, Sequence[dict[str, Any]]],
    path_index: Mapping[str, int],
    crop_rankings: Sequence[np.ndarray],
    *,
    role_multipliers: Mapping[str, float],
    crop_rrf_k: int,
    candidate_k: int,
) -> dict[str, np.ndarray]:
    output = {}
    for example_id, records in records_by_example.items():
        piece_lists = []
        weights = []
        for record in records:
            query_index = path_index.get(str(record["path"]))
            role = str(record.get("role", "primary"))
            multiplier = float(role_multipliers.get(role, 0.0))
            weight = crop_weight(record) * multiplier
            if query_index is None or weight <= 0:
                continue
            piece_lists.append(crop_rankings[query_index])
            weights.append(weight)
        if piece_lists:
            output[str(example_id)] = stable_weighted_rrf(
                piece_lists,
                weights,
                rrf_k=crop_rrf_k,
                top_k=candidate_k,
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen HANDS query-crop role multipliers from a frozen rank cache."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--search-cache", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--crop-rrf-k", type=int, default=1)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument(
        "--auxiliary-multipliers",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    )
    parser.add_argument(
        "--context-multipliers",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    )
    args = parser.parse_args()

    if any(value < 0 for value in args.auxiliary_multipliers):
        parser.error("auxiliary multipliers must be non-negative")
    if any(value < 0 for value in args.context_multipliers):
        parser.error("context multipliers must be non-negative")
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)
    metadata = [
        row for row in read_jsonl(args.metadata) if str(row.get("slot")) == "HANDS"
    ]
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
        slot="HANDS",
    )
    paths, piece_ranks = load_compact_piece_ranks(
        args.search_cache, candidate_k=args.candidate_k
    )
    path_index = {path: index for index, path in enumerate(paths)}

    def evaluate(auxiliary: float, context_multiplier: float) -> dict[str, Any]:
        multipliers = {
            "primary": 1.0,
            "auxiliary": float(auxiliary),
            "context": float(context_multiplier),
        }
        rankings = fuse_examples_with_role_multipliers(
            context["records"],
            path_index,
            piece_ranks,
            role_multipliers=multipliers,
            crop_rrf_k=args.crop_rrf_k,
            candidate_k=args.candidate_k,
        )
        cases = _rank_cases(context["examples"], rankings, metadata, slot="HANDS")
        return {
            "role_multipliers": multipliers,
            "summary": summarize_ranked_cases(cases, cutoffs),
            "cases": cases,
        }

    baseline = evaluate(1.0, 1.0)
    trials = []
    safe = []
    seen = set()
    for auxiliary in args.auxiliary_multipliers:
        for context_multiplier in args.context_multipliers:
            key = (float(auxiliary), float(context_multiplier))
            if key in seen:
                continue
            seen.add(key)
            result = evaluate(*key)
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
                "role_multipliers": result["role_multipliers"],
                "summary": result["summary"],
                "domain_regressions": regressions,
                "cutoff_regressions": cutoff_regressions,
            }
            trials.append(compact)
            if not regressions and not cutoff_regressions:
                safe.append({**compact, "cases": result["cases"]})

    best_safe = max(
        safe,
        key=lambda row: (
            row["summary"]["hits"]["1"],
            row["summary"]["hits"]["10"],
            row["summary"]["hits"]["100"],
            row["summary"]["mrr"],
            -abs(row["role_multipliers"]["auxiliary"] - 1.0),
            -abs(row["role_multipliers"]["context"] - 1.0),
        ),
    ) if safe else None

    payload = {
        "slot": "HANDS",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "metadata": portable_path(args.metadata),
        "search_cache": portable_path(args.search_cache),
        "candidate_k": args.candidate_k,
        "crop_rrf_k": args.crop_rrf_k,
        "cutoffs": list(cutoffs),
        "baseline": baseline,
        "trial_count": len(trials),
        "trials": trials,
        "safe_policy_count": len(safe),
        "best_safe": best_safe,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": portable_path(args.output),
                "baseline_summary": baseline["summary"],
                "safe_policy_count": len(safe),
                "best_safe": (
                    {
                        "role_multipliers": best_safe["role_multipliers"],
                        "summary": best_safe["summary"],
                    }
                    if best_safe
                    else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
