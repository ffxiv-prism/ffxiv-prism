from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.paths import portable_path
from scripts.evaluate_glamour_embedding_model import (
    build_encoder,
    load_crop_embedding_cache,
    load_existing_candidate_embeddings,
    load_or_build_candidate_embeddings,
    load_or_build_crop_embeddings,
    read_jsonl,
    search_crops,
)
from scripts.evaluate_retrieval_funnel import (
    HANDS_CANDIDATE_CROP_ROUTES,
    HEAD_CANDIDATE_CROP_ROUTES,
    SLOTS,
    build_fold_context,
    rank_body_cases,
    summarize_ranked_cases,
    validate_cutoffs,
)
from scripts.train_glamour_query_encoder import load_slot_eligibility


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a zero-shot visual encoder on one strict retrieval fold."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--encoder-kind",
        choices=("clip", "dinov2", "dinov3", "auto"),
        default="auto",
    )
    parser.add_argument("--preprocess", choices=("color", "gray", "edge"), default="color")
    parser.add_argument("--pooling", choices=("cls", "mean", "cls_mean"), default="cls")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--crop-cache", type=Path, required=True)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Encode and persist the fold query cache without running retrieval.",
    )
    parser.add_argument(
        "--rebuild-crop-cache",
        action="store_true",
        help="Ignore a valid crop cache and encode it again.",
    )
    parser.add_argument("--examples", type=Path, required=True)
    quality_group = parser.add_mutually_exclusive_group(required=True)
    quality_group.add_argument("--quality-manifest", type=Path)
    quality_group.add_argument("--slot-eligibility-manifest", type=Path)
    parser.add_argument("--slot", choices=SLOTS, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--candidate-crop-routing", action="store_true")
    parser.add_argument("--candidate-gender-routing", action="store_true")
    parser.add_argument("--fusion", choices=("max", "rrf"), default="max")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()

    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)
    metadata = [
        row
        for row in read_jsonl(args.metadata)
        if str(row.get("slot")) == args.slot
    ]
    if not metadata:
        raise ValueError(f"No {args.slot} rows found in {args.metadata}")
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
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality_by_id,
        regions_dir=args.regions_dir,
        slot=args.slot,
    )
    query_embeddings = None if args.rebuild_crop_cache else load_crop_embedding_cache(
        context["records"],
        args.crop_cache,
        model_id=args.model_id,
        preprocess=args.preprocess,
        pooling=args.pooling,
    )
    if args.cache_only and query_embeddings is not None:
        print(
            json.dumps(
                {
                    "query_cache": portable_path(args.crop_cache),
                    "crop_count": len(query_embeddings),
                    "reused": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    candidate_embeddings = (
        None
        if args.cache_only
        else load_existing_candidate_embeddings(metadata, args.index_dir)
    )
    encoder = None
    if query_embeddings is None or candidate_embeddings is None:
        encoder = build_encoder(
            args.model_id,
            args.encoder_kind,
            args.device,
            args.preprocess,
            pooling=args.pooling,
        )
    if candidate_embeddings is None and not args.cache_only:
        candidate_embeddings = load_or_build_candidate_embeddings(
            encoder,
            metadata,
            args.index_dir,
            args.batch_size,
        )
    if query_embeddings is None:
        query_embeddings = load_or_build_crop_embeddings(
            encoder,
            context["records"],
            args.crop_cache,
            args.batch_size,
            rebuild=args.rebuild_crop_cache,
        )
    if args.cache_only:
        print(
            json.dumps(
                {
                    "query_cache": portable_path(args.crop_cache),
                    "crop_count": len(query_embeddings),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    candidate_crop_routes = None
    if args.candidate_crop_routing:
        candidate_crop_routes = {
            "HEAD": HEAD_CANDIDATE_CROP_ROUTES,
            "HANDS": HANDS_CANDIDATE_CROP_ROUTES,
        }.get(args.slot)
        if candidate_crop_routes is None:
            raise ValueError(f"Candidate crop routing is not defined for {args.slot}")
    candidate_gender_by_example = None
    if args.candidate_gender_routing:
        candidate_gender_by_example = {
            str(example["example_id"]): str(example.get("model_gender", ""))
            for example in context["examples"]
        }

    search = search_crops(
        metadata,
        candidate_embeddings,
        context["records"],
        query_embeddings,
        args.candidate_k,
        device=args.device,
        candidate_crop_routes=candidate_crop_routes,
        candidate_gender_by_example=candidate_gender_by_example,
    )
    cases = rank_body_cases(
        context["examples"],
        context["records"],
        search,
        metadata,
        slot=args.slot,
        fusion=args.fusion,
        rrf_k=args.rrf_k,
    )
    summary = summarize_ranked_cases(cases, cutoffs)
    payload = {
        "model_id": args.model_id,
        "encoder_kind": args.encoder_kind,
        "preprocess": args.preprocess,
        "pooling": args.pooling,
        "slot": args.slot,
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "metadata": portable_path(args.metadata),
        "index_dir": portable_path(args.index_dir),
        "crop_cache": portable_path(args.crop_cache),
        "quality_manifest": portable_path(args.quality_manifest),
        "slot_eligibility_manifest": portable_path(
            args.slot_eligibility_manifest
        ),
        "candidate_k": args.candidate_k,
        "candidate_crop_routing": args.candidate_crop_routing,
        "candidate_gender_routing": args.candidate_gender_routing,
        "fusion": args.fusion,
        "rrf_k": args.rrf_k,
        "cutoffs": list(cutoffs),
        "summary": summary,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
