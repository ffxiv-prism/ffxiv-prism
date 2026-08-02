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

from ffxiv_prism.jsonl import read_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402
from scripts.evaluate_cached_query_retrieval import (  # noqa: E402
    load_query_embeddings,
)
from scripts.evaluate_glamour_embedding_model import search_crops  # noqa: E402
from scripts.evaluate_retrieval_funnel import (  # noqa: E402
    HANDS_CANDIDATE_CROP_ROUTES,
    build_fold_context,
    rank_body_cases,
    validate_cutoffs,
)
from scripts.train_glamour_query_encoder import (  # noqa: E402
    load_slot_eligibility,
)


HANDS_CROP_POLICIES: dict[str, frozenset[str]] = {
    "all": frozenset(
        {
            "hands_left",
            "hands_right",
            "hands_forearm_left",
            "hands_forearm_right",
            "hands_arm_cover_context",
        }
    ),
    "primary_bilateral": frozenset({"hands_left", "hands_right"}),
    "forearm_bilateral": frozenset(
        {"hands_forearm_left", "hands_forearm_right"}
    ),
    "left_side": frozenset({"hands_left", "hands_forearm_left"}),
    "right_side": frozenset({"hands_right", "hands_forearm_right"}),
    "no_context": frozenset(
        {
            "hands_left",
            "hands_right",
            "hands_forearm_left",
            "hands_forearm_right",
        }
    ),
    "primary_plus_context": frozenset(
        {"hands_left", "hands_right", "hands_arm_cover_context"}
    ),
    "forearm_plus_context": frozenset(
        {
            "hands_forearm_left",
            "hands_forearm_right",
            "hands_arm_cover_context",
        }
    ),
    "hands_left": frozenset({"hands_left"}),
    "hands_right": frozenset({"hands_right"}),
    "hands_forearm_left": frozenset({"hands_forearm_left"}),
    "hands_forearm_right": frozenset({"hands_forearm_right"}),
    "hands_arm_cover_context": frozenset({"hands_arm_cover_context"}),
}


def select_policy_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    allowed_names: frozenset[str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(example_id): [
            dict(record)
            for record in rows
            if str(record.get("name")) in allowed_names
        ]
        for example_id, rows in records.items()
    }


def ranks_by_id(
    cases: Sequence[Mapping[str, Any]],
    population_ids: frozenset[str],
) -> dict[str, int | None]:
    case_by_id = {str(case["example_id"]): case for case in cases}
    return {
        example_id: (
            int(case_by_id[example_id]["rank"])
            if example_id in case_by_id
            and case_by_id[example_id].get("status") == "eligible"
            and case_by_id[example_id].get("rank") is not None
            else None
        )
        for example_id in population_ids
    }


def summarize_fixed_population(
    ranks: Mapping[str, int | None],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    denominator = len(ranks)
    finite = [int(rank) for rank in ranks.values() if rank is not None]
    return {
        "eligible_case_count": denominator,
        "retrieved_truth_count": len(finite),
        "missing_truth_rank_count": denominator - len(finite),
        "hits": {
            str(cutoff): sum(rank <= cutoff for rank in finite)
            for cutoff in cutoffs
        },
        "recall": {
            str(cutoff): (
                sum(rank <= cutoff for rank in finite) / denominator
                if denominator
                else 0.0
            )
            for cutoff in cutoffs
        },
        "mrr": (
            sum(1.0 / rank for rank in finite) / denominator
            if denominator
            else 0.0
        ),
    }


def oracle_summary(
    trials: Sequence[Mapping[str, Any]],
    population_ids: frozenset[str],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    best_ranks: dict[str, int | None] = {}
    winning_trials: dict[str, list[str]] = {}
    for example_id in population_ids:
        candidates = [
            (int(trial["ranks"][example_id]), str(trial["trial_id"]))
            for trial in trials
            if trial["ranks"].get(example_id) is not None
        ]
        if not candidates:
            best_ranks[example_id] = None
            winning_trials[example_id] = []
            continue
        best_rank = min(rank for rank, _ in candidates)
        best_ranks[example_id] = best_rank
        winning_trials[example_id] = sorted(
            trial_id for rank, trial_id in candidates if rank == best_rank
        )
    return {
        "summary": summarize_fixed_population(best_ranks, cutoffs),
        "best_rank_by_example": best_ranks,
        "winning_trials_by_example": winning_trials,
    }


def quality_by_example(
    examples: Sequence[Mapping[str, Any]],
    *,
    quality_manifest: Path | None,
    slot_eligibility_manifest: Path | None,
) -> dict[str, bool]:
    if quality_manifest is not None:
        return {
            str(row["example_id"]): bool(row.get("passed"))
            for row in read_jsonl(quality_manifest)
        }
    if slot_eligibility_manifest is None:
        raise ValueError("A quality source is required")
    eligibility = load_slot_eligibility(slot_eligibility_manifest)
    return {
        str(example["example_id"]): (
            "HANDS" in eligibility.get(str(example["example_id"]), set())
        )
        for example in examples
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose HANDS crop and fusion bottlenecks from a frozen query "
            "embedding cache without retraining the model."
        )
    )
    parser.add_argument("--examples", type=Path, required=True)
    quality_group = parser.add_mutually_exclusive_group(required=True)
    quality_group.add_argument("--quality-manifest", type=Path)
    quality_group.add_argument("--slot-eligibility-manifest", type=Path)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument(
        "--candidate-crop-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--candidate-gender-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--candidate-gender-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--rrf-ks", type=int, nargs="+", default=[1, 10, 60])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()

    if any(value < 1 for value in args.rrf_ks):
        parser.error("rrf-ks must contain positive integers")
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)
    examples = read_jsonl(args.examples)
    quality = quality_by_example(
        examples,
        quality_manifest=args.quality_manifest,
        slot_eligibility_manifest=args.slot_eligibility_manifest,
    )
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(args.index_dir / "embeddings.npy", mmap_mode="r")
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality,
        regions_dir=args.regions_dir,
        slot="HANDS",
    )
    query_embeddings = load_query_embeddings(args.query_cache)
    required_paths = {
        str(record["path"])
        for rows in context["records"].values()
        for record in rows
    }
    missing_paths = sorted(required_paths - set(query_embeddings))
    if missing_paths:
        raise ValueError(
            f"Query cache misses {len(missing_paths)} crop paths; "
            f"first={missing_paths[0]}"
        )

    candidate_crop_routes = (
        HANDS_CANDIDATE_CROP_ROUTES if args.candidate_crop_routing else None
    )
    candidate_gender_by_example = (
        {
            str(example["example_id"]): str(example.get("model_gender", ""))
            for example in context["examples"]
        }
        if args.candidate_gender_routing
        else None
    )
    search = search_crops(
        metadata,
        candidate_embeddings,
        context["records"],
        query_embeddings,
        args.candidate_k,
        device=args.device,
        candidate_crop_routes=candidate_crop_routes,
        candidate_gender_by_example=candidate_gender_by_example,
        candidate_gender_fallback=args.candidate_gender_fallback,
    )

    full_cases = rank_body_cases(
        context["examples"],
        context["records"],
        search,
        metadata,
        slot="HANDS",
        fusion="rrf",
        rrf_k=1,
    )
    population_ids = frozenset(
        str(case["example_id"])
        for case in full_cases
        if case.get("status") == "eligible"
    )
    trials: list[dict[str, Any]] = []
    for policy_name, allowed_names in HANDS_CROP_POLICIES.items():
        policy_records = select_policy_records(context["records"], allowed_names)
        configs = [("max", None), *(('rrf', value) for value in args.rrf_ks)]
        for fusion, rrf_k in configs:
            cases = rank_body_cases(
                context["examples"],
                policy_records,
                search,
                metadata,
                slot="HANDS",
                fusion=fusion,
                rrf_k=rrf_k or 1,
            )
            ranks = ranks_by_id(cases, population_ids)
            trial_id = (
                f"{policy_name}:{fusion}"
                if rrf_k is None
                else f"{policy_name}:rrf{rrf_k}"
            )
            trials.append(
                {
                    "trial_id": trial_id,
                    "policy": policy_name,
                    "allowed_crops": sorted(allowed_names),
                    "fusion": fusion,
                    "rrf_k": rrf_k,
                    "summary": summarize_fixed_population(ranks, cutoffs),
                    "ranks": ranks,
                }
            )

    best_trial = max(
        trials,
        key=lambda row: (
            row["summary"]["hits"].get("10", 0),
            row["summary"]["hits"].get("1", 0),
            row["summary"]["mrr"],
        ),
    )
    oracle = oracle_summary(trials, population_ids, cutoffs)
    payload = {
        "version": 1,
        "slot": "HANDS",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "slot_eligibility_manifest": portable_path(
            args.slot_eligibility_manifest
        ),
        "quality_manifest": portable_path(args.quality_manifest),
        "regions_dir": portable_path(args.regions_dir),
        "index_dir": portable_path(args.index_dir),
        "query_cache": portable_path(args.query_cache),
        "candidate_k": args.candidate_k,
        "candidate_crop_routing": args.candidate_crop_routing,
        "candidate_gender_routing": args.candidate_gender_routing,
        "candidate_gender_fallback": args.candidate_gender_fallback,
        "cutoffs": list(cutoffs),
        "population_count": len(population_ids),
        "trial_count": len(trials),
        "best_trial_id": best_trial["trial_id"],
        "best_trial_summary": best_trial["summary"],
        "oracle": oracle,
        "trials": trials,
        "interpretation": (
            "The oracle is a diagnostic upper bound selected per validation "
            "example. It is not a deployable metric and must not be used for "
            "model selection on blind data."
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
                "population_count": len(population_ids),
                "trial_count": len(trials),
                "best_trial_id": best_trial["trial_id"],
                "best_trial_summary": best_trial["summary"],
                "oracle_summary": oracle["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
