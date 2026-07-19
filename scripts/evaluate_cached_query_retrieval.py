from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.paths import portable_path
from scripts.evaluate_glamour_embedding_model import read_jsonl, search_crops
from scripts.evaluate_retrieval_funnel import (
    HANDS_CANDIDATE_CROP_ROUTES,
    HEAD_CANDIDATE_CROP_ROUTES,
    SLOTS,
    build_fold_context,
    rank_body_cases,
    summarize_ranked_cases,
    validate_cutoffs,
)


def load_query_embeddings(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        paths = [str(value) for value in cache["paths"].tolist()]
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(paths) != len(embeddings):
        raise ValueError(
            f"Invalid query cache {path}: {len(paths)} paths vs "
            f"embedding shape {embeddings.shape}"
        )
    return dict(zip(paths, embeddings, strict=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate a frozen query cache with a different crop-fusion policy "
            "without loading or training a model."
        )
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--slot", choices=SLOTS, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--candidate-crop-routing", action="store_true")
    parser.add_argument("--candidate-gender-routing", action="store_true")
    parser.add_argument("--fusion", choices=("max", "rrf"), default="max")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()

    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)
    examples = read_jsonl(args.examples)
    quality_by_id = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
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
    )
    query_embeddings = load_query_embeddings(args.query_cache)
    missing_paths = sorted(
        {
            str(record["path"])
            for records in context["records"].values()
            for record in records
            if str(record["path"]) not in query_embeddings
        }
    )
    if missing_paths:
        raise ValueError(
            f"Query cache is missing {len(missing_paths)} required crop paths; "
            f"first={missing_paths[0]}"
        )

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
        "slot": args.slot,
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "index_dir": portable_path(args.index_dir),
        "query_cache": portable_path(args.query_cache),
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
