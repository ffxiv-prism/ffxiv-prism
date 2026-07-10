from __future__ import annotations

import argparse
import gc
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.collectors.eorzea_collection.glamour_parser import (
    appearance_name_key,
)
from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import CACHE_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.evaluation_split import load_split_assignments
from scripts.evaluate_glamour_embedding_model import search_crops
from scripts.evaluate_glamour_matching import (
    ROLE_PROFILES,
    SLOTS,
    VISIBLE_WEIGHTS,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    collect_crop_records,
    crop_record_weight,
)
from scripts.train_glamour_pair_reranker import PairReranker, rerank_crops
from scripts.train_glamour_query_encoder import (
    encode_records,
    filter_crop_records_by_slots,
    load_query_head_from_checkpoint,
)

DEFAULT_EXAMPLES = Path("data/processed/glamour_examples.jsonl")
DEFAULT_QUALITY = Path("data/processed/front_fullbody_quality_final_v44.jsonl")
DEFAULT_REGIONS = Path("data/debug/outfit-regions-v6-1882")
DEFAULT_INDEX = Path(
    "data/indexes/facebook-dinov2-base-slot-aligned-v3-body-multiscale"
)
DEFAULT_FOLD_DIR = Path("data/interim/item-disjoint-cv-v44-train-only")
DEFAULT_QUERY_TEMPLATE = (
    "data/models/item_disjoint_cv_v44_fold{fold:02d}_"
    "qv60_strict_base_e10.pt"
)
DEFAULT_RERANKER_TEMPLATE = (
    "data/models/item_disjoint_cv_v44_fold{fold:02d}_"
    "qv60_strict_reranker_e7_h1000.pt"
)
DEFAULT_OUTPUT = Path("data/processed/retrieval_funnel_v44_strict.json")
DEFAULT_CUTOFFS = (1, 10, 100, 1000)
HEAD_CANDIDATE_CROP_ROUTES = {
    "head_upper_accessory": ("head_upper_accessory",),
    "head_face_accessory": ("head_face_accessory",),
    "head_context": ("full",),
}
HANDS_CANDIDATE_CROP_ROUTES = {
    "hands_left": ("left_arm_tight", "left_forearm_context"),
    "hands_forearm_left": ("left_arm_tight", "left_forearm_context"),
    "hands_right": ("right_arm_tight", "right_forearm_context"),
    "hands_forearm_right": ("right_arm_tight", "right_forearm_context"),
    "hands_arm_cover_context": ("both_hands", "hands_upper_context"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure index coverage, base-DINOv2 recall, fold-trained query "
            "recall, and reranker conversion on frozen item-disjoint folds."
        )
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--quality-manifest", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument(
        "--slot",
        choices=SLOTS,
        default="BODY",
        help="Equipment slot evaluated by the strict retrieval funnel.",
    )
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--fold-numbers",
        type=int,
        nargs="+",
        help="Optional explicit one-based folds to evaluate.",
    )
    parser.add_argument("--query-template", default=DEFAULT_QUERY_TEMPLATE)
    parser.add_argument(
        "--reranker-template",
        default=DEFAULT_RERANKER_TEMPLATE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument(
        "--candidate-crop-routing",
        action="store_true",
        help="Route semantic query crops to matching candidate crop variants.",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
    )
    args = parser.parse_args()

    fold_numbers = (
        list(args.fold_numbers)
        if args.fold_numbers
        else list(range(1, args.folds + 1))
    )
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)
    device = resolve_torch_device(args.device)
    examples = read_jsonl(args.examples)
    quality_by_id = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
    }
    candidate_crop_routes = (
        HEAD_CANDIDATE_CROP_ROUTES
        if args.candidate_crop_routing and args.slot == "HEAD"
        else None
    )
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(
        args.index_dir / "embeddings.npy",
        mmap_mode="r",
    )
    contexts = [
        build_fold_context(
            fold=fold,
            fold_manifest=args.fold_dir / f"fold-{fold:02d}.json",
            all_examples=examples,
            quality_by_id=quality_by_id,
            regions_dir=args.regions_dir,
            slot=args.slot,
        )
        for fold in fold_numbers
    ]

    from transformers import AutoModel

    print(f"device={device} stage=base_dinov2")
    base_model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    for context in contexts:
        embeddings = encode_records(
            base_model,
            None,
            context["records"],
            args.batch_size,
            device,
        )
        search = search_crops(
            metadata,
            candidate_embeddings,
            context["records"],
            embeddings,
            args.candidate_k,
            device=str(device),
            candidate_crop_routes=candidate_crop_routes,
        )
        context["base_cases"] = rank_body_cases(
            context["examples"],
            context["records"],
            search,
            metadata,
            slot=args.slot,
        )
        print_stage(context["fold"], "base_dinov2", context["base_cases"], cutoffs)
        del embeddings, search
        release_cuda()
    del base_model
    release_cuda()

    for context in contexts:
        fold = int(context["fold"])
        query_checkpoint = Path(args.query_template.format(fold=fold))
        reranker_checkpoint = Path(args.reranker_template.format(fold=fold))
        print(f"fold={fold} stage=fold_query checkpoint={query_checkpoint}")
        query_model = AutoModel.from_pretrained(
            args.model_id,
            cache_dir=CACHE_DIR / "huggingface" / "hub",
            local_files_only=True,
        ).to(device)
        query_head = load_query_head_from_checkpoint(
            query_checkpoint,
            query_model,
            device,
        )
        query_embeddings = encode_records(
            query_model,
            query_head,
            context["records"],
            args.batch_size,
            device,
        )
        query_search = search_crops(
            metadata,
            candidate_embeddings,
            context["records"],
            query_embeddings,
            args.candidate_k,
            device=str(device),
            candidate_crop_routes=candidate_crop_routes,
        )
        context["query_cases"] = rank_body_cases(
            context["examples"],
            context["records"],
            query_search,
            metadata,
            slot=args.slot,
        )
        print_stage(fold, "fold_query", context["query_cases"], cutoffs)
        del query_search

        reranker_payload = torch.load(
            reranker_checkpoint,
            map_location=device,
            weights_only=True,
        )
        reranker = PairReranker(
            int(reranker_payload["dimension"]),
            per_slot_heads=bool(
                reranker_payload.get("per_slot_heads", False)
            ),
            dropout=float(reranker_payload.get("dropout", 0.1)),
        ).to(device)
        reranker.load_state_dict(reranker_payload["state_dict"])
        reranked_search = rerank_crops(
            reranker,
            metadata,
            candidate_embeddings,
            context["records"],
            query_embeddings,
            args.candidate_k,
            args.candidate_k,
            device,
            candidate_view_aggregate="max",
        )
        context["reranker_cases"] = rank_body_cases(
            context["examples"],
            context["records"],
            reranked_search,
            metadata,
            slot=args.slot,
        )
        print_stage(fold, "reranker", context["reranker_cases"], cutoffs)
        context["query_checkpoint"] = query_checkpoint
        context["reranker_checkpoint"] = reranker_checkpoint
        del (
            query_embeddings,
            reranked_search,
            reranker,
            reranker_payload,
            query_head,
            query_model,
        )
        release_cuda()

    fold_payloads = [
        compact_fold_payload(context, cutoffs)
        for context in contexts
    ]
    pooled_cases = [
        case
        for fold_payload in fold_payloads
        for case in fold_payload["cases"]
    ]
    payload = {
        "evaluation_type": "strict_item_disjoint_retrieval_funnel",
        "slot": args.slot,
        "model_id": args.model_id,
        "examples": portable_path(args.examples),
        "quality_manifest": portable_path(args.quality_manifest),
        "regions_dir": portable_path(args.regions_dir),
        "index_dir": portable_path(args.index_dir),
        "fold_dir": portable_path(args.fold_dir),
        "candidate_k": args.candidate_k,
        "candidate_crop_routing": args.candidate_crop_routing,
        "cutoffs": list(cutoffs),
        "fixed_fusion": {
            "role_profile": "balanced",
            "tag_penalty_profile": "none",
            "aggregate": "max",
            "candidate_view_aggregate": "max",
        },
        "folds": [
            {
                key: value
                for key, value in fold_payload.items()
                if key != "cases"
            }
            for fold_payload in fold_payloads
        ],
        "pooled": summarize_funnel_cases(pooled_cases, cutoffs),
        "cases": pooled_cases,
    }
    write_json(args.output, payload)
    print(
        f"pooled={payload['pooled']['eligible_case_count']} "
        f"output={args.output}"
    )
    return 0


def validate_cutoffs(
    cutoffs: Sequence[int],
    candidate_k: int,
) -> tuple[int, ...]:
    normalized = tuple(sorted({int(value) for value in cutoffs}))
    if not normalized or normalized[0] < 1:
        raise ValueError("cutoffs must contain positive integers")
    if normalized[-1] > candidate_k:
        raise ValueError("largest cutoff cannot exceed candidate-k")
    return normalized


def build_fold_context(
    *,
    fold: int,
    fold_manifest: Path,
    all_examples: list[dict[str, Any]],
    quality_by_id: dict[str, bool],
    regions_dir: Path,
    slot: str,
) -> dict[str, Any]:
    assignments = load_split_assignments(fold_manifest)
    selected_ids = {
        glamour_id
        for glamour_id, role in assignments.items()
        if role == "validation"
    }
    split_examples = [
        example
        for example in all_examples
        if str(example["glamour_id"]) in selected_ids
    ]
    missing_quality = [
        str(example["example_id"])
        for example in split_examples
        if str(example["example_id"]) not in quality_by_id
    ]
    if missing_quality:
        raise ValueError(
            f"Fold {fold} has {len(missing_quality)} images missing quality labels"
        )
    examples = [
        example
        for example in split_examples
        if quality_by_id[str(example["example_id"])]
    ]
    records = filter_crop_records_by_slots(
        collect_crop_records(examples, regions_dir),
        [slot],
    )
    return {
        "fold": fold,
        "fold_manifest": fold_manifest,
        "examples": examples,
        "records": records,
    }


def rank_body_cases(
    examples: Sequence[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    search_results_by_crop: dict[str, list[dict[str, Any]]],
    metadata: list[dict[str, Any]],
    slot: str = "BODY",
    fusion: str = "max",
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    if fusion not in {"max", "rrf"}:
        raise ValueError(f"Unsupported fusion: {fusion}")
    if rrf_k < 1:
        raise ValueError("rrf-k must be positive")
    available_ids = available_pieces_by_slot(metadata)[slot]
    available_keys = available_appearance_keys_by_slot(metadata)[slot]
    role_weights = ROLE_PROFILES["balanced"]
    cases = []
    for example in examples:
        label = (example.get("core_slot_labels") or {}).get(slot)
        base = {
            "example_id": str(example["example_id"]),
            "glamour_id": int(example["glamour_id"]),
            "model_gender": example.get("model_gender"),
            "model_race": example.get("model_race"),
            "expected_name": label.get("item_name") if label else None,
            "rank": None,
        }
        if not label or label.get("piece_match_type") == "empty":
            cases.append({**base, "status": "empty_label"})
            continue
        expected_ids = {
            int(value)
            for value in label.get("piece_ids", [])
        }
        if not expected_ids:
            cases.append({**base, "status": "missing_truth_piece_id"})
            continue
        expected_key = appearance_name_key(label.get("item_name"))
        eligible_ids = expected_ids & available_ids
        appearance_eligible = bool(
            expected_key and expected_key in available_keys
        )
        if not eligible_ids and not appearance_eligible:
            cases.append({**base, "status": "truth_not_in_index"})
            continue
        slot_records = [
            record
            for record in records.get(str(example["example_id"]), [])
            if record["slot"] == slot
            and VISIBLE_WEIGHTS.get(record["visibility"], 0.0) > 0.0
        ]
        if not slot_records:
            cases.append({**base, "status": "no_usable_crop"})
            continue

        scores: dict[int, float] = {}
        keys_by_piece: dict[int, str] = {}
        for record in slot_records:
            weight = crop_record_weight(record, role_weights, {})
            if weight <= 0:
                continue
            for result_offset, result in enumerate(
                search_results_by_crop.get(str(record["path"]), []),
                start=1,
            ):
                piece_id = int(result["piece_id"])
                if fusion == "rrf":
                    result_rank = int(result.get("rank", result_offset))
                    contribution = weight / (rrf_k + result_rank)
                    scores[piece_id] = scores.get(piece_id, 0.0) + contribution
                else:
                    contribution = weight * float(result["score"])
                    scores[piece_id] = max(
                        scores.get(piece_id, 0.0),
                        contribution,
                    )
                keys_by_piece[piece_id] = appearance_name_key(
                    result.get("name_en")
                )
        ranked_ids = sorted(
            scores,
            key=lambda piece_id: scores[piece_id],
            reverse=True,
        )
        rank = next(
            (
                offset
                for offset, piece_id in enumerate(ranked_ids, start=1)
                if piece_id in eligible_ids
                or keys_by_piece.get(piece_id) == expected_key
            ),
            None,
        )
        cases.append(
            {
                **base,
                "status": "eligible",
                "rank": rank,
            }
        )
    return cases


def summarize_ranked_cases(
    cases: Sequence[dict[str, Any]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    eligible = [case for case in cases if case["status"] == "eligible"]
    ranks = [
        int(case["rank"])
        for case in eligible
        if case.get("rank") is not None
    ]
    count = len(eligible)
    return {
        "eligible_case_count": count,
        "hits": {
            str(cutoff): sum(rank <= cutoff for rank in ranks)
            for cutoff in cutoffs
        },
        "recall": {
            str(cutoff): (
                sum(rank <= cutoff for rank in ranks) / count
                if count
                else 0.0
            )
            for cutoff in cutoffs
        },
        "mrr": (
            sum(1.0 / rank for rank in ranks) / count
            if count
            else 0.0
        ),
    }


def combine_stage_cases(
    base_cases: Sequence[dict[str, Any]],
    query_cases: Sequence[dict[str, Any]],
    reranker_cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    def keyed(
        cases: Sequence[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return {str(case["example_id"]): case for case in cases}

    query_by_id = keyed(query_cases)
    reranker_by_id = keyed(reranker_cases)
    output = []
    for base in base_cases:
        example_id = str(base["example_id"])
        query = query_by_id[example_id]
        reranker = reranker_by_id[example_id]
        if not (
            base["status"] == query["status"] == reranker["status"]
        ):
            raise ValueError(f"Stage status mismatch for {example_id}")
        output.append(
            {
                key: value
                for key, value in base.items()
                if key != "rank"
            }
            | {
                "base_rank": base.get("rank"),
                "query_rank": query.get("rank"),
                "reranker_rank": reranker.get("rank"),
            }
        )
    return output


def summarize_funnel_cases(
    cases: Sequence[dict[str, Any]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    eligible = [case for case in cases if case["status"] == "eligible"]
    index_denominator = sum(
        case["status"] in {"eligible", "truth_not_in_index"}
        for case in cases
    )
    index_hits = sum(case["status"] == "eligible" for case in cases)

    def stage_cases(rank_key: str) -> list[dict[str, Any]]:
        return [
            {
                "status": case["status"],
                "rank": case.get(rank_key),
            }
            for case in cases
        ]

    query_pool_hits = sum(
        case.get("query_rank") is not None
        and int(case["query_rank"]) <= max(cutoffs)
        for case in eligible
    )
    reranker_top1 = sum(case.get("reranker_rank") == 1 for case in eligible)
    return {
        "case_count": len(cases),
        "status_counts": dict(Counter(case["status"] for case in cases)),
        "eligible_case_count": len(eligible),
        "index_coverage": {
            "hits": index_hits,
            "denominator": index_denominator,
            "rate": (
                index_hits / index_denominator
                if index_denominator
                else 0.0
            ),
        },
        "base_dinov2": summarize_ranked_cases(
            stage_cases("base_rank"),
            cutoffs,
        ),
        "fold_query": summarize_ranked_cases(
            stage_cases("query_rank"),
            cutoffs,
        ),
        "reranker": summarize_ranked_cases(
            stage_cases("reranker_rank"),
            cutoffs,
        ),
        "reranker_top1_conversion_from_query_pool": {
            "hits": reranker_top1,
            "query_pool_hits": query_pool_hits,
            "rate": (
                reranker_top1 / query_pool_hits
                if query_pool_hits
                else 0.0
            ),
        },
    }


def compact_fold_payload(
    context: dict[str, Any],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    cases = combine_stage_cases(
        context["base_cases"],
        context["query_cases"],
        context["reranker_cases"],
    )
    return {
        "fold": context["fold"],
        "fold_manifest": portable_path(context["fold_manifest"]),
        "query_checkpoint": portable_path(context["query_checkpoint"]),
        "reranker_checkpoint": portable_path(context["reranker_checkpoint"]),
        "summary": summarize_funnel_cases(cases, cutoffs),
        "cases": cases,
    }


def print_stage(
    fold: int,
    stage: str,
    cases: Sequence[dict[str, Any]],
    cutoffs: Sequence[int],
) -> None:
    summary = summarize_ranked_cases(cases, cutoffs)
    recalls = " ".join(
        f"r@{cutoff}={summary['recall'][str(cutoff)]:.3f}"
        for cutoff in cutoffs
    )
    print(
        f"fold={fold} stage={stage} "
        f"eligible={summary['eligible_case_count']} {recalls}"
    )


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
