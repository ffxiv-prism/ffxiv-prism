from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import CACHE_DIR, PROCESSED_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.evaluation_split import load_split_assignments
from scripts.evaluate_glamour_matching import (
    ROLE_PROFILES,
    SLOTS,
    TAG_PENALTY_PROFILES,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    collect_crop_records,
    evaluate_config,
    summarize,
)
from scripts.evaluate_glamour_embedding_model import search_crops
from scripts.train_glamour_pair_reranker import PairReranker, rerank_crops
from scripts.train_glamour_query_encoder import (
    encode_records,
    filter_crop_records_by_slots,
    load_query_head_from_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen query/reranker checkpoints on a non-blind split. "
            "blind_test is intentionally unsupported."
        )
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROCESSED_DIR / "glamour_examples.jsonl",
    )
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument(
        "--slots",
        nargs="+",
        choices=SLOTS,
        default=list(SLOTS),
    )
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--example-model-gender")
    parser.add_argument("--example-model-races", nargs="+")
    parser.add_argument("--candidate-model-gender")
    parser.add_argument("--candidate-races", nargs="+")
    parser.add_argument(
        "--candidate-image-parts",
        nargs="+",
        help="Optional candidate image-part allowlist, for example body or front.",
    )
    parser.add_argument(
        "--candidate-sources",
        nargs="+",
        help="Optional supplemental_candidate_source allowlist.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROCESSED_DIR / "glamour_split_v2.json",
    )
    parser.add_argument(
        "--split-role",
        choices=("train", "validation", "development_test", "confirmation_test"),
        default="validation",
    )
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--reranker-checkpoint", type=Path, required=True)
    parser.add_argument("--tuning-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retrieval-image-k", type=int, default=1000)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--candidate-view-aggregate",
        choices=("max", "top2_mean"),
        default="max",
    )
    parser.add_argument("--role-profile", choices=sorted(ROLE_PROFILES))
    parser.add_argument(
        "--tag-penalty-profile",
        choices=sorted(TAG_PENALTY_PROFILES),
    )
    parser.add_argument("--aggregate", choices=("max", "sum"))
    args = parser.parse_args()

    assignments = load_split_assignments(args.split_manifest)
    selected_ids = {
        glamour_id
        for glamour_id, role in assignments.items()
        if role == args.split_role
    }
    all_examples = read_jsonl(args.examples)
    split_examples = [
        example
        for example in all_examples
        if str(example["glamour_id"]) in selected_ids
        and (
            not args.example_model_gender
            or str(example.get("model_gender") or "").lower()
            == args.example_model_gender.lower()
        )
        and (
            not args.example_model_races
            or str(example.get("model_race") or "").lower()
            in {value.lower() for value in args.example_model_races}
        )
    ]
    quality_by_id = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
    }
    missing = {
        str(example["example_id"])
        for example in split_examples
        if str(example["example_id"]) not in quality_by_id
    }
    if missing:
        raise ValueError(
            f"Quality manifest does not cover {len(missing)} split images"
        )
    examples = [
        example
        for example in split_examples
        if quality_by_id[str(example["example_id"])]
    ]
    if not examples:
        raise ValueError("Quality manifest retained no split images")
    records = filter_crop_records_by_slots(
        collect_crop_records(examples, args.regions_dir),
        args.slots,
    )

    device = resolve_torch_device(args.device)
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(
        args.index_dir / "embeddings.npy",
        mmap_mode="r",
    )
    if (
        args.candidate_model_gender
        or args.candidate_races
        or args.candidate_image_parts
        or args.candidate_sources
    ):
        candidate_races = {
            value.lower()
            for value in (args.candidate_races or [])
        }
        candidate_image_parts = {
            value.lower()
            for value in (args.candidate_image_parts or [])
        }
        candidate_sources = {
            value.lower()
            for value in (args.candidate_sources or [])
        }
        candidate_indices = [
            index
            for index, row in enumerate(metadata)
            if (
                not args.candidate_model_gender
                or str(row.get("model_gender") or "").lower()
                == args.candidate_model_gender.lower()
            )
            and (
                not candidate_races
                or str(row.get("race") or "").lower() in candidate_races
            )
            and (
                not candidate_image_parts
                or str(row.get("image_part") or "").lower()
                in candidate_image_parts
            )
            and (
                not candidate_sources
                or str(row.get("supplemental_candidate_source") or "").lower()
                in candidate_sources
            )
        ]
        if not candidate_indices:
            raise ValueError("Candidate model filter retained no index rows")
        metadata = [metadata[index] for index in candidate_indices]
        candidate_embeddings = np.asarray(
            candidate_embeddings[candidate_indices],
            dtype=np.float32,
        )
    from transformers import AutoModel

    query_model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    query_head = load_query_head_from_checkpoint(
        args.query_checkpoint,
        query_model,
        device,
    )
    query_embeddings = encode_records(
        query_model,
        query_head,
        records,
        args.batch_size,
        device,
    )
    base_search = search_crops(
        metadata,
        candidate_embeddings,
        records,
        query_embeddings,
        args.candidate_k,
        device=str(device),
    )

    reranker_payload = torch.load(
        args.reranker_checkpoint,
        map_location=device,
        weights_only=True,
    )
    reranker = PairReranker(
        int(reranker_payload["dimension"]),
        per_slot_heads=bool(reranker_payload.get("per_slot_heads", False)),
    ).to(device)
    reranker.load_state_dict(reranker_payload["state_dict"])
    search = rerank_crops(
        reranker,
        metadata,
        candidate_embeddings,
        records,
        query_embeddings,
        args.retrieval_image_k,
        args.candidate_k,
        device,
        candidate_view_aggregate=args.candidate_view_aggregate,
    )

    tuning = json.loads(args.tuning_report.read_text(encoding="utf-8"))
    tuning_selected = tuning["evaluation"]["validation"]
    selected_role_profile = (
        args.role_profile or tuning_selected["role_profile"]
    )
    selected_tag_profile = (
        args.tag_penalty_profile
        or tuning_selected["tag_penalty_profile"]
    )
    selected_aggregate = args.aggregate or tuning_selected["aggregate"]
    config_sweep = []
    for role_profile in sorted(ROLE_PROFILES):
        for tag_profile in sorted(TAG_PENALTY_PROFILES):
            for aggregate in ("max", "sum"):
                config_outcomes = evaluate_config(
                    examples,
                    records,
                    search,
                    available_pieces_by_slot(metadata),
                    available_appearance_keys_by_slot(metadata),
                    args.top_k,
                    ROLE_PROFILES[role_profile],
                    TAG_PENALTY_PROFILES[tag_profile],
                    aggregate,
                )
                config_summary = summarize(config_outcomes, args.top_k)
                config_summary.update(
                    role_profile=role_profile,
                    tag_penalty_profile=tag_profile,
                    aggregate=aggregate,
                )
                config_sweep.append(config_summary)
    outcomes = evaluate_config(
        examples,
        records,
        search,
        available_pieces_by_slot(metadata),
        available_appearance_keys_by_slot(metadata),
        args.top_k,
        ROLE_PROFILES[selected_role_profile],
        TAG_PENALTY_PROFILES[selected_tag_profile],
        selected_aggregate,
    )
    base_outcomes = evaluate_config(
        examples,
        records,
        base_search,
        available_pieces_by_slot(metadata),
        available_appearance_keys_by_slot(metadata),
        args.top_k,
        ROLE_PROFILES[selected_role_profile],
        TAG_PENALTY_PROFILES[selected_tag_profile],
        selected_aggregate,
    )
    summary = summarize(outcomes, args.top_k)
    summary.update(
        role_profile=selected_role_profile,
        tag_penalty_profile=selected_tag_profile,
        aggregate=selected_aggregate,
    )
    base_summary = summarize(base_outcomes, args.top_k)
    comparison = compare_outcomes(base_outcomes, outcomes)
    payload = {
        "evaluation_type": "frozen_checkpoint_non_blind",
        "split_role": args.split_role,
        "slots": list(args.slots),
        "example_model_gender": args.example_model_gender,
        "example_model_races": args.example_model_races,
        "candidate_model_gender": args.candidate_model_gender,
        "candidate_races": args.candidate_races,
        "candidate_image_parts": args.candidate_image_parts,
        "candidate_sources": args.candidate_sources,
        "candidate_view_aggregate": args.candidate_view_aggregate,
        "selected_config": {
            "role_profile": selected_role_profile,
            "tag_penalty_profile": selected_tag_profile,
            "aggregate": selected_aggregate,
            "source": (
                "explicit"
                if args.role_profile
                or args.tag_penalty_profile
                or args.aggregate
                else "tuning_report"
            ),
        },
        "config_sweep": config_sweep,
        "evaluation": {
            args.split_role: summary,
            f"{args.split_role}_cases": outcomes,
        },
        "retrieval_diagnostics": {
            "cosine_baseline": base_summary,
            "reranker_comparison": comparison,
        },
        "quality_manifest": portable_path(args.quality_manifest),
        "query_checkpoint": portable_path(args.query_checkpoint),
        "reranker_checkpoint": portable_path(args.reranker_checkpoint),
        "tuning_report": portable_path(args.tuning_report),
        "split_manifest": portable_path(args.split_manifest),
    }
    write_json(args.output, payload)
    print(
        f"split={args.split_role} eligible={summary['eligible_case_count']} "
        f"appearance@1={summary['appearance_recall_at_1']:.3f} "
        f"appearance@{args.top_k}={summary['appearance_recall_at_k']:.3f} "
        f"output={args.output}"
    )
    return 0


def compare_outcomes(
    base_outcomes: list[dict[str, object]],
    reranked_outcomes: list[dict[str, object]],
) -> dict[str, object]:
    def case_key(row: dict[str, object]) -> tuple[str, str]:
        return str(row.get("example_id")), str(row.get("slot"))

    base_by_key = {case_key(row): row for row in base_outcomes}
    eligible_pairs = [
        (base_by_key[case_key(row)], row)
        for row in reranked_outcomes
        if case_key(row) in base_by_key
        and row.get("status") == "eligible"
        and base_by_key[case_key(row)].get("status") == "eligible"
    ]
    top1_rescued = []
    top1_degraded = []
    topk_rescued = []
    topk_degraded = []
    for base, reranked in eligible_pairs:
        key = {
            "example_id": reranked.get("example_id"),
            "glamour_id": reranked.get("glamour_id"),
            "slot": reranked.get("slot"),
            "model_race": reranked.get("model_race"),
            "model_gender": reranked.get("model_gender"),
            "expected_name": reranked.get("expected_name"),
            "base_appearance_rank": base.get("appearance_rank"),
            "reranked_appearance_rank": reranked.get("appearance_rank"),
        }
        base_top1 = bool(base.get("appearance_hit_at_1"))
        reranked_top1 = bool(reranked.get("appearance_hit_at_1"))
        base_topk = bool(base.get("appearance_hit_at_k"))
        reranked_topk = bool(reranked.get("appearance_hit_at_k"))
        if reranked_top1 and not base_top1:
            top1_rescued.append(key)
        elif base_top1 and not reranked_top1:
            top1_degraded.append(key)
        if reranked_topk and not base_topk:
            topk_rescued.append(key)
        elif base_topk and not reranked_topk:
            topk_degraded.append(key)
    return {
        "eligible_pair_count": len(eligible_pairs),
        "top1_rescued_count": len(top1_rescued),
        "top1_degraded_count": len(top1_degraded),
        "topk_rescued_count": len(topk_rescued),
        "topk_degraded_count": len(topk_degraded),
        "top1_rescued_cases": top1_rescued,
        "top1_degraded_cases": top1_degraded,
        "topk_rescued_cases": topk_rescued,
        "topk_degraded_cases": topk_degraded,
    }


if __name__ == "__main__":
    raise SystemExit(main())
