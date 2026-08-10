from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.paths import CACHE_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from scripts.evaluate_glamour_embedding_model import read_jsonl, search_crops
from scripts.evaluate_retrieval_funnel import (
    HANDS_CANDIDATE_CROP_ROUTES,
    HEAD_CANDIDATE_CROP_ROUTES,
    SLOTS,
    build_fold_context,
    rank_body_cases,
    release_cuda,
    summarize_ranked_cases,
    validate_cutoffs,
)
from scripts.train_glamour_query_encoder import (
    encode_records,
    load_slot_eligibility,
    load_query_head_from_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one slot/fold query encoder on a frozen candidate index; "
            "omit --query-checkpoint for a zero-shot backbone."
        )
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--examples", type=Path, required=True)
    quality_group = parser.add_mutually_exclusive_group(required=True)
    quality_group.add_argument("--quality-manifest", type=Path)
    quality_group.add_argument("--slot-eligibility-manifest", type=Path)
    parser.add_argument("--slot", choices=SLOTS, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument(
        "--fold-role",
        choices=("train", "validation", "blind_test"),
        default="validation",
        help="Evaluate the selected manifest role (default: validation).",
    )
    parser.add_argument("--query-checkpoint", type=Path)
    parser.add_argument(
        "--pose-normalize-hands",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--query-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument(
        "--candidate-crop-routing",
        action="store_true",
        help="Route semantic query crops to slot-aligned candidate crops.",
    )
    parser.add_argument(
        "--candidate-gender-routing",
        action="store_true",
        help="Search candidate renders matching the query model gender.",
    )
    parser.add_argument(
        "--candidate-gender-fallback",
        action="store_true",
        help=(
            "With gender routing, retain opposite-gender rows only for pieces "
            "that have no render in the query gender."
        ),
    )
    parser.add_argument(
        "--fusion",
        choices=("max", "rrf"),
        default="max",
        help="Fuse evidence across query crops by cosine max or weighted RRF.",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--include-top-results",
        type=int,
        default=0,
        help=(
            "Retain this many fused candidates per eligible case for error "
            "analysis or multi-view aggregation (default: compact reports)."
        ),
    )
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)
    examples = read_jsonl(args.examples)
    if args.quality_manifest:
        quality_by_id = {
            str(row["example_id"]): bool(row.get("passed"))
            for row in read_jsonl(args.quality_manifest)
        }
    else:
        eligibility = load_slot_eligibility(args.slot_eligibility_manifest)
        quality_by_id = {
            str(example["example_id"]): (
                args.slot
                in eligibility.get(str(example["example_id"]), set())
            )
            for example in examples
        }
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(args.index_dir / "embeddings.npy", mmap_mode="r")
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality_by_id,
        regions_dir=args.regions_dir,
        slot=args.slot,
        fold_role=args.fold_role,
    )
    candidate_crop_routes = None
    if args.candidate_crop_routing:
        candidate_crop_routes = {
            "HEAD": HEAD_CANDIDATE_CROP_ROUTES,
            "HANDS": HANDS_CANDIDATE_CROP_ROUTES,
        }.get(args.slot)
        if candidate_crop_routes is None:
            raise ValueError(
                f"Candidate crop routing is not defined for {args.slot}"
            )
    candidate_gender_by_example = None
    if args.candidate_gender_routing:
        candidate_gender_by_example = {
            str(example["example_id"]): str(example.get("model_gender", ""))
            for example in context["examples"]
        }

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    query_head = (
        load_query_head_from_checkpoint(
            args.query_checkpoint,
            model,
            device,
        )
        if args.query_checkpoint is not None
        else None
    )
    query_embeddings = encode_records(
        model,
        query_head,
        context["records"],
        args.batch_size,
        device,
        pose_normalize_hands=args.pose_normalize_hands,
    )
    if args.query_cache:
        paths = sorted(query_embeddings)
        args.query_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.query_cache,
            paths=np.asarray(paths),
            embeddings=np.stack([query_embeddings[path] for path in paths]).astype(
                np.float32,
                copy=False,
            ),
            checkpoint=np.asarray(
                str(args.query_checkpoint)
                if args.query_checkpoint is not None
                else ""
            ),
        )

    search = search_crops(
        metadata,
        candidate_embeddings,
        context["records"],
        query_embeddings,
        args.candidate_k,
        device=str(device),
        candidate_crop_routes=candidate_crop_routes,
        candidate_gender_by_example=candidate_gender_by_example,
        candidate_gender_fallback=args.candidate_gender_fallback,
    )
    cases = rank_body_cases(
        context["examples"],
        context["records"],
        search,
        metadata,
        slot=args.slot,
        fusion=args.fusion,
        rrf_k=args.rrf_k,
        include_top_results=args.include_top_results,
    )
    summary = summarize_ranked_cases(cases, cutoffs)
    payload: dict[str, Any] = {
        "slot": args.slot,
        "fold": args.fold,
        "fold_role": args.fold_role,
        "fold_manifest": portable_path(args.fold_manifest),
        "index_dir": portable_path(args.index_dir),
        "query_checkpoint": portable_path(args.query_checkpoint),
        "quality_manifest": portable_path(args.quality_manifest),
        "slot_eligibility_manifest": portable_path(
            args.slot_eligibility_manifest
        ),
        "pose_normalize_hands": args.pose_normalize_hands,
        "query_cache": portable_path(args.query_cache),
        "candidate_k": args.candidate_k,
        "candidate_crop_routing": args.candidate_crop_routing,
        "candidate_gender_routing": args.candidate_gender_routing,
        "candidate_gender_fallback": args.candidate_gender_fallback,
        "fusion": args.fusion,
        "rrf_k": args.rrf_k,
        "included_top_results": args.include_top_results,
        "cutoffs": list(cutoffs),
        "summary": summary,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(args.output), "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )
    del model, query_embeddings, search
    release_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
