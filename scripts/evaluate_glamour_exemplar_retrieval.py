from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.collectors.eorzea_collection.glamour_parser import (
    appearance_name_key,
)
from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import CACHE_DIR, PROCESSED_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
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
from scripts.train_glamour_query_encoder import (
    encode_records,
    filter_crop_records_by_slots,
    load_query_head_from_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate train-only high-quality player-image exemplars as a "
            "BODY retrieval branch. Blind evaluation is intentionally blocked."
        )
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indexes/facebook-dinov2-base-slot-aligned-v2"),
        help="Canonical metadata used only for fixed evaluation eligibility.",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROCESSED_DIR / "glamour_examples.jsonl",
    )
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROCESSED_DIR / "glamour_split_v2.json",
    )
    parser.add_argument(
        "--split-role",
        choices=("validation", "development_test", "confirmation_test"),
        default="validation",
    )
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--role-profile", choices=sorted(ROLE_PROFILES), default="primary")
    parser.add_argument(
        "--tag-penalty-profile",
        choices=sorted(TAG_PENALTY_PROFILES),
        default="mild",
    )
    parser.add_argument("--aggregate", choices=("max", "sum"), default="max")
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assignments = load_split_assignments(args.split_manifest)
    quality_by_id = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
    }
    all_examples = read_jsonl(args.examples)
    accepted_examples = [
        example
        for example in all_examples
        if quality_by_id.get(str(example["example_id"]), False)
    ]
    train_examples = [
        example
        for example in accepted_examples
        if assignments.get(str(example["glamour_id"])) == "train"
    ]
    evaluation_examples = [
        example
        for example in accepted_examples
        if assignments.get(str(example["glamour_id"])) == args.split_role
    ]
    if not train_examples or not evaluation_examples:
        raise ValueError("Train or evaluation split has no accepted examples")

    selected_examples = [*train_examples, *evaluation_examples]
    records = filter_crop_records_by_slots(
        collect_crop_records(selected_examples, args.regions_dir),
        ["BODY"],
    )
    device = resolve_torch_device(args.device)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    query_head = load_query_head_from_checkpoint(
        args.query_checkpoint,
        model,
        device,
    )
    embeddings = encode_records(
        model,
        query_head,
        records,
        args.batch_size,
        device,
    )

    train_records = {
        str(example["example_id"]): records.get(str(example["example_id"]), [])
        for example in train_examples
    }
    evaluation_records = {
        str(example["example_id"]): records.get(str(example["example_id"]), [])
        for example in evaluation_examples
    }
    exemplars = build_exemplars(train_examples, train_records, embeddings)
    search = search_exemplars(
        exemplars,
        evaluation_records,
        embeddings,
        candidate_k=args.candidate_k,
        device=device,
    )

    canonical_metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    outcomes = evaluate_config(
        evaluation_examples,
        evaluation_records,
        search,
        available_pieces_by_slot(canonical_metadata),
        available_appearance_keys_by_slot(canonical_metadata),
        args.top_k,
        ROLE_PROFILES[args.role_profile],
        TAG_PENALTY_PROFILES[args.tag_penalty_profile],
        args.aggregate,
    )
    summary = summarize(outcomes, args.top_k)
    summary.update(
        role_profile=args.role_profile,
        tag_penalty_profile=args.tag_penalty_profile,
        aggregate=args.aggregate,
    )
    payload = {
        "evaluation_type": "train_only_player_exemplar_retrieval",
        "split_role": args.split_role,
        "evaluation": {
            args.split_role: summary,
            f"{args.split_role}_cases": outcomes,
        },
        "train_example_count": len(train_examples),
        "train_exemplar_crop_count": len(exemplars),
        "train_appearance_count": len(
            {str(row["appearance_key"]) for row in exemplars}
        ),
        "quality_manifest": portable_path(args.quality_manifest),
        "query_checkpoint": portable_path(args.query_checkpoint),
        "split_manifest": portable_path(args.split_manifest),
        "regions_dir": portable_path(args.regions_dir),
    }
    write_json(args.output, payload)
    print(
        f"split={args.split_role} exemplars={len(exemplars)} "
        f"eligible={summary['eligible_case_count']} "
        f"appearance@1={summary['appearance_recall_at_1']:.3f} "
        f"appearance@{args.top_k}={summary['appearance_recall_at_k']:.3f} "
        f"output={args.output}"
    )
    return 0


def build_exemplars(
    examples: list[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    embeddings: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    exemplars = []
    for example in examples:
        example_id = str(example["example_id"])
        label = example.get("core_slot_labels", {}).get("BODY") or {}
        name = str(label.get("item_name") or "").strip()
        appearance_key = appearance_name_key(name)
        piece_ids = [
            int(value)
            for value in label.get("piece_ids") or []
        ]
        if not appearance_key or not piece_ids:
            continue
        for record in records.get(example_id, []):
            path = str(record["path"])
            if path not in embeddings:
                continue
            exemplars.append(
                {
                    "example_id": example_id,
                    "glamour_id": str(example["glamour_id"]),
                    "path": path,
                    "embedding": np.asarray(
                        embeddings[path],
                        dtype=np.float32,
                    ),
                    "appearance_key": appearance_key,
                    "name_en": name,
                    "piece_ids": piece_ids,
                    "model_race": example.get("model_race"),
                    "model_gender": example.get("model_gender"),
                }
            )
    return exemplars


def search_exemplars(
    exemplars: list[dict[str, Any]],
    query_records: dict[str, list[dict[str, Any]]],
    query_embeddings: dict[str, np.ndarray],
    *,
    candidate_k: int,
    device: torch.device | str = "cpu",
) -> dict[str, list[dict[str, Any]]]:
    if not exemplars:
        raise ValueError("No train exemplars with BODY labels were found")
    torch_device = torch.device(device)
    matrix = torch.as_tensor(
        np.stack([row["embedding"] for row in exemplars]),
        device=torch_device,
    )
    results = {}
    for rows in query_records.values():
        for record in rows:
            path = str(record["path"])
            if path in results or path not in query_embeddings:
                continue
            scores_tensor = matrix @ torch.as_tensor(
                np.asarray(query_embeddings[path], dtype=np.float32),
                device=torch_device,
            )
            order = torch.argsort(
                scores_tensor,
                descending=True,
            ).cpu().tolist()
            scores = scores_tensor.cpu().tolist()
            unique_glamour_neighbors = []
            seen_glamour_appearances = set()
            for index in order:
                exemplar = exemplars[index]
                key = (
                    str(exemplar["glamour_id"]),
                    str(exemplar["appearance_key"]),
                )
                if key in seen_glamour_appearances:
                    continue
                seen_glamour_appearances.add(key)
                unique_glamour_neighbors.append(exemplar)
                if len(unique_glamour_neighbors) >= 20:
                    break
            vote_counts = {
                count: Counter(
                    str(row["appearance_key"])
                    for row in unique_glamour_neighbors[:count]
                )
                for count in (5, 10, 20)
            }
            best_by_appearance: dict[str, dict[str, Any]] = {}
            for index in order:
                exemplar = exemplars[index]
                appearance_key = str(exemplar["appearance_key"])
                if appearance_key in best_by_appearance:
                    continue
                best_by_appearance[appearance_key] = {
                    "rank": 0,
                    "score": float(scores[index]),
                    "piece_id": int(exemplar["piece_ids"][0]),
                    "gearset_slug": None,
                    "name_en": exemplar["name_en"],
                    "name_zh": None,
                    "link_type": "train_player_exemplar",
                    "source_example_id": exemplar["example_id"],
                    "source_glamour_id": exemplar["glamour_id"],
                    "source_crop_path": exemplar["path"],
                    "model_race": exemplar.get("model_race"),
                    "model_gender": exemplar.get("model_gender"),
                    "neighbor_vote_fraction_5": (
                        vote_counts[5][appearance_key]
                        / max(min(5, len(unique_glamour_neighbors)), 1)
                    ),
                    "neighbor_vote_fraction_10": (
                        vote_counts[10][appearance_key]
                        / max(min(10, len(unique_glamour_neighbors)), 1)
                    ),
                    "neighbor_vote_fraction_20": (
                        vote_counts[20][appearance_key]
                        / max(min(20, len(unique_glamour_neighbors)), 1)
                    ),
                }
                if len(best_by_appearance) >= candidate_k:
                    break
            ranked = list(best_by_appearance.values())
            for rank, row in enumerate(ranked, start=1):
                row["rank"] = rank
            results[path] = ranked
    return results


if __name__ == "__main__":
    raise SystemExit(main())
