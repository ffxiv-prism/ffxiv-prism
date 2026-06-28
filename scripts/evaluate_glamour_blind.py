from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import CACHE_DIR, PROCESSED_DIR, portable_path
from ffxiv_prism.vision.evaluation_split import load_split_assignments
from scripts.evaluate_glamour_matching import (
    ROLE_PROFILES,
    TAG_PENALTY_PROFILES,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    collect_crop_records,
    evaluate_config,
    summarize,
)
from scripts.train_glamour_pair_reranker import (
    PairReranker,
    rerank_crops,
)
from scripts.train_glamour_query_encoder import (
    encode_records,
    load_training_checkpoint,
)

DEFAULT_INDEX = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_REGIONS = Path("data/debug/outfit-regions-v3-624")
DEFAULT_QUERY = Path("data/models/glamour_query_encoder_v5_160g_grouped.pt")
DEFAULT_RERANKER = Path(
    "data/models/glamour_pair_reranker_v2_conservative.pt"
)
DEFAULT_TUNING_REPORT = (
    PROCESSED_DIR / "glamour_pair_reranker_eval_v2_conservative.json"
)
DEFAULT_SPLIT = PROCESSED_DIR / "glamour_split_v2.json"
DEFAULT_OUTPUT = PROCESSED_DIR / "glamour_pair_reranker_blind_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one-time evaluation on the currently sealed blind groups."
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROCESSED_DIR / "glamour_examples.jsonl",
    )
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument(
        "--quality-manifest",
        type=Path,
        help=(
            "Optional complete visual-review manifest. Only passed images are "
            "scored, while every reviewed sealed group is consumed."
        ),
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--query-checkpoint", type=Path, default=DEFAULT_QUERY)
    parser.add_argument("--reranker-checkpoint", type=Path, default=DEFAULT_RERANKER)
    parser.add_argument("--tuning-report", type=Path, default=DEFAULT_TUNING_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retrieval-image-k", type=int, default=400)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--consume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark evaluated blind groups as confirmation_test after success.",
    )
    args = parser.parse_args()

    assignments = load_split_assignments(args.split_manifest)
    blind_ids = {
        glamour_id
        for glamour_id, role in assignments.items()
        if role == "blind_test"
    }
    if not blind_ids:
        raise ValueError("No sealed blind_test groups remain")
    all_examples = read_jsonl(args.examples)
    blind_examples = [
        example
        for example in all_examples
        if str(example["glamour_id"]) in blind_ids
    ]
    if not blind_examples:
        raise ValueError("Blind groups have no image examples")
    examples = blind_examples
    quality_excluded_ids: set[str] = set()
    if args.quality_manifest is not None:
        quality_rows = read_jsonl(args.quality_manifest)
        quality_by_id = {
            str(row["example_id"]): bool(row.get("passed"))
            for row in quality_rows
        }
        blind_example_ids = {
            str(example["example_id"])
            for example in blind_examples
        }
        missing_reviews = blind_example_ids - set(quality_by_id)
        if missing_reviews:
            raise ValueError(
                "Quality manifest does not cover every blind image: "
                f"{sorted(missing_reviews)}"
            )
        examples = [
            example
            for example in blind_examples
            if quality_by_id[str(example["example_id"])]
        ]
        quality_excluded_ids = {
            str(example["example_id"])
            for example in blind_examples
            if not quality_by_id[str(example["example_id"])]
        }
        if not examples:
            raise ValueError("Quality manifest retained no blind images")
    records = collect_crop_records(examples, args.regions_dir)

    device = torch.device(args.device)
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(
        args.index_dir / "embeddings.npy",
        mmap_mode="r",
    )
    from transformers import AutoModel

    query_model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    load_training_checkpoint(
        args.query_checkpoint,
        query_model,
        None,
        device,
    )
    query_embeddings = encode_records(
        query_model,
        None,
        records,
        args.batch_size,
        device,
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
    )

    tuning = json.loads(args.tuning_report.read_text(encoding="utf-8"))
    selected = tuning["evaluation"]["validation"]
    outcomes = evaluate_config(
        examples,
        records,
        search,
        available_pieces_by_slot(metadata),
        available_appearance_keys_by_slot(metadata),
        args.top_k,
        ROLE_PROFILES[selected["role_profile"]],
        TAG_PENALTY_PROFILES[selected["tag_penalty_profile"]],
        selected["aggregate"],
    )
    summary = summarize(outcomes, args.top_k)
    summary.update(
        role_profile=selected["role_profile"],
        tag_penalty_profile=selected["tag_penalty_profile"],
        aggregate=selected["aggregate"],
    )
    payload = {
        **summary,
        "evaluation_type": "one_time_blind_confirmation",
        "sealed_blind_glamour_ids": sorted(blind_ids),
        "evaluated_glamour_ids": sorted(
            {str(example["glamour_id"]) for example in examples}
        ),
        "quality_excluded_glamour_ids": sorted(
            blind_ids
            - {str(example["glamour_id"]) for example in examples}
        ),
        "quality_manifest": portable_path(args.quality_manifest),
        "quality_retained_image_count": len(examples),
        "quality_excluded_image_count": len(quality_excluded_ids),
        "query_checkpoint": portable_path(args.query_checkpoint),
        "reranker_checkpoint": portable_path(args.reranker_checkpoint),
        "tuning_report": portable_path(args.tuning_report),
        "split_manifest": portable_path(args.split_manifest),
        "cases": outcomes,
    }
    write_json(args.output, payload)
    if args.consume:
        consume_blind_groups(
            args.split_manifest,
            blind_ids,
            args.output,
        )
    print(
        f"blind_glamours={len(blind_ids)} "
        f"eligible={summary['eligible_case_count']} "
        f"appearance@1={summary['appearance_recall_at_1']:.3f} "
        f"appearance@{args.top_k}={summary['appearance_recall_at_k']:.3f} "
        f"mrr={summary['appearance_mrr']:.3f} output={args.output}"
    )
    for slot, row in summary["by_slot"].items():
        print(
            f"  {slot}: app@1={row['appearance_recall_at_1']:.3f} "
            f"app@{args.top_k}={row['appearance_recall_at_k']:.3f}"
        )
    return 0


def consume_blind_groups(
    split_manifest: Path,
    blind_ids: set[str],
    report_path: Path,
) -> None:
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    assignments = payload["assignments"]
    for glamour_id in blind_ids:
        if assignments.get(glamour_id) != "blind_test":
            raise ValueError(f"Blind assignment changed during evaluation: {glamour_id}")
        assignments[glamour_id] = "confirmation_test"
    payload["counts"] = dict(sorted(Counter(assignments.values()).items()))
    payload.setdefault("blind_evaluations", []).append(
        {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "group_count": len(blind_ids),
            "report": portable_path(report_path),
        }
    )
    write_json(split_manifest, payload)


if __name__ == "__main__":
    raise SystemExit(main())
