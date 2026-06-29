from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import PROCESSED_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
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

DEFAULT_FIRST_INDEX = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_FIRST_CROPS = PROCESSED_DIR / "glamour_crop_embeddings_facebook-dinov2-base-color.npz"
DEFAULT_SECOND_INDEX = Path("data/indexes/openai-clip-vit-base-patch32-color")
DEFAULT_SECOND_CROPS = PROCESSED_DIR / "glamour_crop_embeddings_openai-clip-vit-base-patch32-color.npz"
DEFAULT_OUTPUT = PROCESSED_DIR / "glamour_matching_eval_results_dinov2-base-clip-zscore-fusion.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate score-level fusion of two cached glamour embedding models."
    )
    parser.add_argument("--first-index", type=Path, default=DEFAULT_FIRST_INDEX)
    parser.add_argument("--first-crops", type=Path, default=DEFAULT_FIRST_CROPS)
    parser.add_argument("--second-index", type=Path, default=DEFAULT_SECOND_INDEX)
    parser.add_argument("--second-crops", type=Path, default=DEFAULT_SECOND_CROPS)
    parser.add_argument("--first-weights", default="0.25,0.5,0.75")
    parser.add_argument("--examples", type=Path, default=PROCESSED_DIR / "glamour_examples.jsonl")
    parser.add_argument("--regions-dir", type=Path, default=Path("data/debug/outfit-regions/overlap-final-201"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--sweep-crop-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    args.device = str(resolve_torch_device(args.device))

    weights = parse_weights(args.first_weights)
    first = load_source(args.first_index, args.first_crops)
    second = load_source(args.second_index, args.second_crops)
    validate_source_alignment(first, second)

    metadata = first["metadata"]
    examples = read_jsonl(args.examples)
    crop_records = collect_crop_records(examples, args.regions_dir)
    available_piece_ids = available_pieces_by_slot(metadata)
    available_appearance_keys = available_appearance_keys_by_slot(metadata)
    configs = crop_configs(args.sweep_crop_profiles)

    evaluated: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for first_weight in weights:
        search_results = build_fused_search_results(
            metadata=metadata,
            first_embeddings=first["embeddings"],
            first_crop_embeddings=first["crop_embeddings"],
            second_embeddings=second["embeddings"],
            second_crop_embeddings=second["crop_embeddings"],
            crop_records=crop_records,
            first_weight=first_weight,
            candidate_k=args.candidate_k,
            device=args.device,
        )
        for role_profile, tag_profile, aggregate in configs:
            outcomes = evaluate_config(
                examples,
                crop_records,
                search_results,
                available_piece_ids,
                available_appearance_keys,
                args.top_k,
                ROLE_PROFILES[role_profile],
                TAG_PENALTY_PROFILES[tag_profile],
                aggregate,
            )
            summary = summarize(outcomes, args.top_k)
            summary.update(
                first_weight=first_weight,
                second_weight=1.0 - first_weight,
                role_profile=role_profile,
                tag_penalty_profile=tag_profile,
                aggregate=aggregate,
            )
            evaluated.append((summary, outcomes))

        best_for_weight = select_best(
            [item for item in evaluated if item[0]["first_weight"] == first_weight]
        )[0]
        print_summary(best_for_weight)

    best_summary, best_outcomes = select_best(evaluated)
    payload = {
        **best_summary,
        "fusion": "per-crop candidate z-score weighted sum",
        "first_index": portable_path(args.first_index),
        "first_crops": portable_path(args.first_crops),
        "second_index": portable_path(args.second_index),
        "second_crops": portable_path(args.second_crops),
        "examples": portable_path(args.examples),
        "regions_dir": portable_path(args.regions_dir),
        "candidate_k": args.candidate_k,
        "sweep_results": [summary for summary, _ in evaluated],
        "cases": best_outcomes,
    }
    write_json(args.output, payload)
    print(f"best_output={args.output}")
    print_summary(best_summary)
    return 0


def parse_weights(value: str) -> list[float]:
    weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not weights:
        raise ValueError("At least one fusion weight is required")
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("Fusion weights must be between 0 and 1")
    return weights


def load_source(index_dir: Path, crop_cache: Path) -> dict[str, Any]:
    metadata = read_jsonl(index_dir / "metadata.jsonl")
    embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")
    if len(metadata) != len(embeddings):
        raise ValueError(f"Metadata/embedding count mismatch in {index_dir}")
    cached = np.load(crop_cache, allow_pickle=False)
    crop_embeddings = {
        str(path): embedding
        for path, embedding in zip(cached["paths"].tolist(), cached["embeddings"], strict=True)
    }
    return {
        "metadata": metadata,
        "embeddings": embeddings,
        "crop_embeddings": crop_embeddings,
    }


def validate_source_alignment(first: dict[str, Any], second: dict[str, Any]) -> None:
    first_paths = [row["local_path"] for row in first["metadata"]]
    second_paths = [row["local_path"] for row in second["metadata"]]
    if first_paths != second_paths:
        raise ValueError("Fusion indexes must contain the same candidate images in the same order")
    if first["crop_embeddings"].keys() != second["crop_embeddings"].keys():
        raise ValueError("Fusion crop caches must contain the same crop paths")


def crop_configs(sweep: bool) -> list[tuple[str, str, str]]:
    if not sweep:
        return [("overlap", "mild", "sum")]
    return [
        (role_profile, tag_profile, aggregate)
        for role_profile in sorted(ROLE_PROFILES)
        for tag_profile in sorted(TAG_PENALTY_PROFILES)
        for aggregate in ("max", "sum")
    ]


def build_fused_search_results(
    *,
    metadata: list[dict[str, Any]],
    first_embeddings: np.ndarray,
    first_crop_embeddings: dict[str, np.ndarray],
    second_embeddings: np.ndarray,
    second_crop_embeddings: dict[str, np.ndarray],
    crop_records: dict[str, list[dict[str, Any]]],
    first_weight: float,
    candidate_k: int,
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    path_slots = {
        str(record["path"]): record["slot"]
        for records in crop_records.values()
        for record in records
        if str(record["path"]) in first_crop_embeddings
    }
    results: dict[str, list[dict[str, Any]]] = {}
    torch_device = resolve_torch_device(device)

    for slot in SLOTS:
        indices = np.asarray(
            [index for index, row in enumerate(metadata) if row.get("slot") == slot],
            dtype=np.int64,
        )
        paths = sorted(path for path, crop_slot in path_slots.items() if crop_slot == slot)
        if not len(indices) or not paths:
            continue

        first_scores = score_matrix(
            first_embeddings,
            first_crop_embeddings,
            indices,
            paths,
            torch_device,
        )
        second_scores = score_matrix(
            second_embeddings,
            second_crop_embeddings,
            indices,
            paths,
            torch_device,
        )
        fused_scores = first_weight * first_scores + (1.0 - first_weight) * second_scores
        image_k = min(candidate_k * 4, len(indices))
        values, offsets = torch.topk(fused_scores, k=image_k, dim=0)
        values_array = values.cpu().numpy()
        offsets_array = offsets.cpu().numpy()

        for column, path in enumerate(paths):
            results[path] = piece_results(
                metadata,
                indices,
                offsets_array[:, column],
                values_array[:, column],
                candidate_k,
            )
    return results


def score_matrix(
    candidate_embeddings: np.ndarray,
    crop_embeddings: dict[str, np.ndarray],
    indices: np.ndarray,
    paths: list[str],
    device: torch.device,
) -> torch.Tensor:
    candidates = torch.tensor(np.asarray(candidate_embeddings[indices]), device=device)
    queries = torch.tensor(np.stack([crop_embeddings[path] for path in paths]), device=device)
    scores = candidates @ queries.T
    means = scores.mean(dim=0, keepdim=True)
    deviations = scores.std(dim=0, correction=0, keepdim=True).clamp_min(1e-6)
    return (scores - means) / deviations


def piece_results(
    metadata: list[dict[str, Any]],
    indices: np.ndarray,
    offsets: np.ndarray,
    values: np.ndarray,
    candidate_k: int,
) -> list[dict[str, Any]]:
    best_by_piece: dict[int, dict[str, Any]] = {}
    for offset, value in zip(offsets, values, strict=True):
        row = metadata[int(indices[int(offset)])]
        for piece in row["pieces"]:
            piece_id = int(piece["piece_id"])
            if piece_id in best_by_piece:
                continue
            best_by_piece[piece_id] = {
                "rank": 0,
                "score": float(value),
                **piece,
                "image_row_id": row["image_row_id"],
                "local_path": row["local_path"],
                "source_image_url": row["source_image_url"],
                "model_gender": row.get("model_gender"),
                "race": row.get("race"),
                "dye_state": row.get("dye_state"),
            }
            if len(best_by_piece) >= candidate_k:
                break
        if len(best_by_piece) >= candidate_k:
            break
    rows = list(best_by_piece.values())
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def select_best(
    evaluated: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return max(
        evaluated,
        key=lambda item: (
            item[0]["appearance_recall_at_1"],
            item[0]["appearance_mrr"],
            item[0]["appearance_recall_at_k"],
            item[0]["exact_recall_at_1"],
        ),
    )


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"weights={summary['first_weight']:.2f}/{summary['second_weight']:.2f} "
        f"eligible={summary['eligible_case_count']} "
        f"appearance@1={summary['appearance_recall_at_1']:.3f} "
        f"appearance@{summary['top_k']}={summary['appearance_recall_at_k']:.3f} "
        f"mrr={summary['appearance_mrr']:.3f} "
        f"profile={summary['role_profile']}/{summary['tag_penalty_profile']}/{summary['aggregate']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
