from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.paths import portable_path  # noqa: E402
from scripts.evaluate_retrieval_funnel import (  # noqa: E402
    appearance_name_key,
    summarize_ranked_cases,
    validate_cutoffs,
)


def result_is_truth(result: Mapping[str, Any], case: Mapping[str, Any]) -> bool:
    expected_piece_ids = {
        int(value) for value in case.get("eligible_expected_piece_ids") or []
    }
    expected_key = str(case.get("expected_appearance_key") or "")
    return int(result["piece_id"]) in expected_piece_ids or (
        bool(expected_key)
        and appearance_name_key(result.get("appearance_key")) == expected_key
    )


def fuse_primary_pool(
    primary_results: Sequence[Mapping[str, Any]],
    secondary_results: Sequence[Mapping[str, Any]],
    *,
    secondary_weight: float,
    rrf_k: int,
    pool_depth: int,
) -> list[Mapping[str, Any]]:
    if secondary_weight < 0:
        raise ValueError("secondary_weight must not be negative")
    if rrf_k < 1 or pool_depth < 1:
        raise ValueError("rrf_k and pool_depth must be positive")
    primary = list(primary_results)
    depth = min(pool_depth, len(primary))
    secondary_ranks = {
        int(row["piece_id"]): rank
        for rank, row in enumerate(secondary_results, start=1)
    }
    scores = []
    for rank, row in enumerate(primary[:depth], start=1):
        score = 1.0 / (rrf_k + rank)
        secondary_rank = secondary_ranks.get(int(row["piece_id"]))
        if secondary_rank is not None and secondary_weight > 0:
            score += secondary_weight / (rrf_k + secondary_rank)
        scores.append(score)
    order = sorted(range(depth), key=lambda index: -scores[index])
    return [primary[index] for index in order] + primary[depth:]


def fuse_union_pool(
    primary_results: Sequence[Mapping[str, Any]],
    secondary_results: Sequence[Mapping[str, Any]],
    *,
    secondary_weight: float,
    rrf_k: int,
    pool_depth: int,
) -> list[Mapping[str, Any]]:
    if secondary_weight < 0:
        raise ValueError("secondary_weight must not be negative")
    if rrf_k < 1 or pool_depth < 1:
        raise ValueError("rrf_k and pool_depth must be positive")
    exemplar: dict[int, Mapping[str, Any]] = {}
    insertion_order: dict[int, int] = {}
    scores: dict[int, float] = {}
    for branch_weight, rows in (
        (1.0, list(primary_results)[:pool_depth]),
        (secondary_weight, list(secondary_results)[:pool_depth]),
    ):
        if branch_weight <= 0:
            continue
        for rank, row in enumerate(rows, start=1):
            piece_id = int(row["piece_id"])
            if piece_id not in exemplar:
                insertion_order[piece_id] = len(insertion_order)
                exemplar[piece_id] = row
            scores[piece_id] = scores.get(piece_id, 0.0) + branch_weight / (
                rrf_k + rank
            )
    ranked_ids = sorted(
        scores,
        key=lambda piece_id: (-scores[piece_id], insertion_order[piece_id]),
    )
    return [exemplar[piece_id] for piece_id in ranked_ids]


def build_variant_cases(
    primary_cases: Sequence[Mapping[str, Any]],
    secondary_by_id: Mapping[str, Mapping[str, Any]],
    *,
    secondary_weight: float,
    rrf_k: int,
    pool_depth: int,
    pool_mode: str = "primary",
) -> list[dict[str, Any]]:
    output = []
    for case in primary_cases:
        base = {
            key: value
            for key, value in case.items()
            if key not in {"results", "rank"}
        }
        if case.get("status") != "eligible":
            output.append({**base, "rank": None})
            continue
        example_id = str(case["example_id"])
        secondary = secondary_by_id.get(example_id)
        if secondary is None or secondary.get("status") != "eligible":
            raise ValueError(f"Secondary report misses eligible case {example_id}")
        fuse = fuse_primary_pool if pool_mode == "primary" else fuse_union_pool
        ranked = fuse(
            case.get("results") or [],
            secondary.get("results") or [],
            secondary_weight=secondary_weight,
            rrf_k=rrf_k,
            pool_depth=pool_depth,
        )
        rank = next(
            (
                rank
                for rank, result in enumerate(ranked, start=1)
                if result_is_truth(result, case)
            ),
            None,
        )
        output.append({**base, "rank": rank})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a secondary retrieval report as a conservative RRF "
            "reranker of a frozen primary candidate pool."
        )
    )
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--secondary-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--secondary-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    )
    parser.add_argument("--rrf-k", type=int, nargs="+", default=[1, 10, 60])
    parser.add_argument("--pool-depth", type=int, nargs="+", default=[100, 300, 1000])
    parser.add_argument(
        "--pool-mode",
        choices=("primary", "union"),
        default="primary",
        help="Freeze the primary pool or fuse the union of both branch pools.",
    )
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()

    if any(weight < 0 for weight in args.secondary_weights):
        parser.error("secondary weights must not be negative")
    if any(value < 1 for value in (*args.rrf_k, *args.pool_depth)):
        parser.error("rrf-k and pool-depth values must be positive")
    cutoffs = validate_cutoffs(args.cutoffs, max(args.cutoffs))
    primary = json.loads(args.primary_report.read_text(encoding="utf-8"))
    secondary = json.loads(args.secondary_report.read_text(encoding="utf-8"))
    for field in ("slot", "fold", "fold_role"):
        if primary.get(field) != secondary.get(field):
            raise ValueError(f"Report field {field} differs")
    if primary.get("fold_role") != "validation":
        raise ValueError("Development ensemble selection requires validation reports")
    primary_cases = list(primary["cases"])
    secondary_by_id = {
        str(case["example_id"]): case for case in secondary["cases"]
    }
    if len(secondary_by_id) != len(secondary["cases"]):
        raise ValueError("Secondary report has duplicate example IDs")
    if any(
        len(case.get("results") or []) < max(args.pool_depth)
        for case in primary_cases
        if case.get("status") == "eligible"
    ):
        raise ValueError("Primary report does not contain the requested pool depth")

    variants = []
    for pool_depth in args.pool_depth:
        for rrf_k in args.rrf_k:
            for secondary_weight in args.secondary_weights:
                cases = build_variant_cases(
                    primary_cases,
                    secondary_by_id,
                    secondary_weight=secondary_weight,
                    rrf_k=rrf_k,
                    pool_depth=pool_depth,
                    pool_mode=args.pool_mode,
                )
                variants.append(
                    {
                        "pool_depth": pool_depth,
                        "rrf_k": rrf_k,
                        "secondary_weight": secondary_weight,
                        "summary": summarize_ranked_cases(cases, cutoffs),
                        "cases": cases,
                    }
                )
    best = max(
        variants,
        key=lambda variant: (
            int(variant["summary"]["hits"].get("10", 0)),
            int(variant["summary"]["hits"].get("1", 0)),
            int(variant["summary"]["hits"].get("100", 0)),
            float(variant["summary"]["mrr"]),
        ),
    )
    payload = {
        "version": 1,
        "slot": primary["slot"],
        "fold": primary["fold"],
        "fold_role": primary["fold_role"],
        "primary_report": portable_path(args.primary_report),
        "secondary_report": portable_path(args.secondary_report),
        "pool_mode": args.pool_mode,
        "primary_candidate_pool_frozen": args.pool_mode == "primary",
        "blind_test_accessed": False,
        "selection_metric": "validation_top10_hits_then_top1_top100_mrr",
        "baseline_summary": primary["summary"],
        "best_variant": {
            key: value for key, value in best.items() if key != "cases"
        },
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "baseline_summary": primary["summary"],
                "best_variant": payload["best_variant"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
