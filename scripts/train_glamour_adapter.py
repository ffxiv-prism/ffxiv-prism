from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import PROCESSED_DIR, portable_path
from ffxiv_prism.vision.adaptation import SlotResidualAdapter
from scripts.evaluate_glamour_embedding_model import search_crops
from scripts.evaluate_glamour_fusion import (
    DEFAULT_SECOND_CROPS,
    DEFAULT_SECOND_INDEX,
    build_fused_search_results,
    parse_weights,
)
from scripts.evaluate_glamour_matching import (
    ROLE_PROFILES,
    SLOTS,
    TAG_PENALTY_PROFILES,
    VISIBLE_WEIGHTS,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    collect_crop_records,
    evaluate_config,
    summarize,
)

DEFAULT_INDEX = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_CROP_CACHE = PROCESSED_DIR / "glamour_crop_embeddings_facebook-dinov2-base-color.npz"
DEFAULT_REGIONS = Path("data/debug/outfit-regions/overlap-final-201")
DEFAULT_CHECKPOINT = Path("data/models/glamour_adapter_dinov2_base.pt")
DEFAULT_OUTPUT = PROCESSED_DIR / "glamour_adapter_eval_dinov2_base.json"


@dataclass(frozen=True)
class TrainingSample:
    path: str
    positives: tuple[int, ...]
    weight: float


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a slot-conditioned low-rank adapter for screenshot-to-render retrieval."
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--crop-cache", type=Path, default=DEFAULT_CROP_CACHE)
    parser.add_argument("--examples", type=Path, default=PROCESSED_DIR / "glamour_examples.jsonl")
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--identity-penalty", type=float, default=0.05)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fusion-second-index", type=Path, default=DEFAULT_SECOND_INDEX)
    parser.add_argument("--fusion-second-crops", type=Path, default=DEFAULT_SECOND_CROPS)
    parser.add_argument("--fusion-first-weights", default="0.5,0.6,0.7,0.75,0.8")
    parser.add_argument("--skip-fusion", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(args.index_dir / "embeddings.npy", mmap_mode="r")
    crop_embeddings = load_crop_embeddings(args.crop_cache)
    examples = read_jsonl(args.examples)
    crop_records = collect_crop_records(examples, args.regions_dir)

    glamour_ids = sorted({str(example["glamour_id"]) for example in examples})
    split_map = deterministic_glamour_split(glamour_ids, args.seed)
    split_examples = {
        split: [
            example
            for example in examples
            if split_map[str(example["glamour_id"])] == split
        ]
        for split in ("train", "validation", "test")
    }
    split_records = {
        split: {
            example["example_id"]: crop_records.get(example["example_id"], [])
            for example in split_examples[split]
        }
        for split in split_examples
    }

    candidate_state = build_candidate_state(metadata, candidate_embeddings, device)
    samples = {
        split: build_samples(
            split_examples[split],
            split_records[split],
            crop_embeddings,
            candidate_state["piece_to_local"],
        )
        for split in split_examples
    }
    print_split_summary(split_map, split_examples, samples)

    dimension = int(candidate_embeddings.shape[1])
    adapter = SlotResidualAdapter(dimension, args.rank).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation = (-1.0, -1.0)
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            adapter,
            optimizer,
            samples["train"],
            crop_embeddings,
            candidate_state["tensors"],
            args.batch_size,
            args.temperature,
            args.identity_penalty,
            device,
        )
        validation = crop_recall(
            adapter,
            samples["validation"],
            crop_embeddings,
            candidate_state["tensors"],
            args.batch_size,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_sample_count": validation["sample_count"],
            "validation_recall_at_1": validation["recall_at_1"],
            "validation_recall_at_10": validation["recall_at_10"],
            "validation_by_slot": validation["by_slot"],
        }
        history.append(row)
        score = (validation["recall_at_1"], validation["recall_at_10"])
        improved = score > best_validation
        if improved:
            best_validation = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0 or stale_epochs == 0:
            print(
                f"epoch={epoch} loss={train_loss:.4f} "
                f"val_crop@1={validation['recall_at_1']:.3f} "
                f"val_crop@10={validation['recall_at_10']:.3f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Adapter training did not produce a checkpoint")
    adapter.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "state_dict": best_state,
            "dimension": dimension,
            "rank": args.rank,
            "slots": list(SLOTS),
            "seed": args.seed,
            "best_epoch": best_epoch,
            "index_dir": portable_path(args.index_dir),
            "crop_cache": portable_path(args.crop_cache),
        },
        args.checkpoint,
    )

    validation_and_test_records = {
        **split_records["validation"],
        **split_records["test"],
    }
    relevant_paths = {
        str(record["path"])
        for records in validation_and_test_records.values()
        for record in records
        if str(record["path"]) in crop_embeddings
    }
    identity_crops = {path: crop_embeddings[path] for path in relevant_paths}
    adapted_crops = project_crop_embeddings(
        adapter,
        validation_and_test_records,
        identity_crops,
        args.batch_size,
        device,
    )
    baseline_search = search_crops(
        metadata,
        candidate_embeddings,
        validation_and_test_records,
        identity_crops,
        args.candidate_k,
        device=args.device,
    )
    adapted_search = search_crops(
        metadata,
        candidate_embeddings,
        validation_and_test_records,
        adapted_crops,
        args.candidate_k,
        device=args.device,
    )
    available_piece_ids = available_pieces_by_slot(metadata)
    available_appearance_keys = available_appearance_keys_by_slot(metadata)
    baseline_eval = select_on_validation_and_evaluate_test(
        split_examples,
        split_records,
        baseline_search,
        available_piece_ids,
        available_appearance_keys,
        args.top_k,
    )
    adapted_eval = select_on_validation_and_evaluate_test(
        split_examples,
        split_records,
        adapted_search,
        available_piece_ids,
        available_appearance_keys,
        args.top_k,
    )
    adapted_fusion_eval = None
    cascade_eval = None
    if not args.skip_fusion:
        adapted_fusion_eval = evaluate_adapted_fusion(
            metadata=metadata,
            first_candidate_embeddings=candidate_embeddings,
            first_crop_embeddings=adapted_crops,
            crop_records=validation_and_test_records,
            split_examples=split_examples,
            split_records=split_records,
            available_piece_ids=available_piece_ids,
            available_appearance_keys=available_appearance_keys,
            second_index=args.fusion_second_index,
            second_crops=args.fusion_second_crops,
            first_weights=parse_weights(args.fusion_first_weights),
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            device=args.device,
        )
        cascade_eval = evaluate_cascade(
            adapted_fusion_eval,
            adapted_eval,
            args.top_k,
        )
    test_crop_metrics = crop_recall(
        adapter,
        samples["test"],
        crop_embeddings,
        candidate_state["tensors"],
        args.batch_size,
        device,
    )

    payload = {
        "adapter": {
            "type": "slot_conditioned_low_rank_residual",
            "rank": args.rank,
            "dimension": dimension,
            "best_epoch": best_epoch,
            "checkpoint": portable_path(args.checkpoint),
        },
        "training": {
            "seed": args.seed,
            "temperature": args.temperature,
            "identity_penalty": args.identity_penalty,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "history": history,
        },
        "split": {
            split: sorted(
                glamour_id
                for glamour_id, assigned in split_map.items()
                if assigned == split
            )
            for split in ("train", "validation", "test")
        },
        "baseline": baseline_eval,
        "adapted": adapted_eval,
        "adapted_fusion": adapted_fusion_eval,
        "cascade": cascade_eval,
        "adapted_test_crop_metrics": test_crop_metrics,
        "index_dir": portable_path(args.index_dir),
        "crop_cache": portable_path(args.crop_cache),
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
    }
    write_json(args.output, payload)
    print_comparison(baseline_eval["test"], adapted_eval["test"])
    if adapted_fusion_eval is not None:
        print_comparison(
            baseline_eval["test"],
            adapted_fusion_eval["test"],
            label="adapted_fusion",
        )
    if cascade_eval is not None:
        print_comparison(
            baseline_eval["test"],
            cascade_eval["test"],
            label="cascade",
        )
    print(f"checkpoint={args.checkpoint} output={args.output}")
    return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deterministic_glamour_split(
    glamour_ids: Iterable[str],
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, str]:
    ids = sorted(
        {str(glamour_id) for glamour_id in glamour_ids},
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest(),
    )
    if len(ids) < 3:
        raise ValueError("At least three glamour groups are required")
    train_count = max(1, round(len(ids) * train_fraction))
    validation_count = max(1, round(len(ids) * validation_fraction))
    if train_count + validation_count >= len(ids):
        validation_count = 1
        train_count = len(ids) - 2
    split = {}
    for index, glamour_id in enumerate(ids):
        if index < train_count:
            split[glamour_id] = "train"
        elif index < train_count + validation_count:
            split[glamour_id] = "validation"
        else:
            split[glamour_id] = "test"
    return split


def load_crop_embeddings(path: Path) -> dict[str, np.ndarray]:
    cached = np.load(path, allow_pickle=False)
    return {
        str(crop_path): embedding
        for crop_path, embedding in zip(
            cached["paths"].tolist(),
            cached["embeddings"],
            strict=True,
        )
    }


def build_candidate_state(
    metadata: list[dict[str, Any]],
    embeddings: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    tensors = {}
    piece_to_local: dict[str, dict[int, tuple[int, ...]]] = {}
    for slot in SLOTS:
        global_indices = [
            index for index, row in enumerate(metadata) if row.get("slot") == slot
        ]
        tensors[slot] = torch.tensor(
            np.asarray(embeddings[global_indices]),
            device=device,
        )
        local_by_piece: dict[int, list[int]] = defaultdict(list)
        for local_index, global_index in enumerate(global_indices):
            for piece in metadata[global_index]["pieces"]:
                local_by_piece[int(piece["piece_id"])].append(local_index)
        piece_to_local[slot] = {
            piece_id: tuple(indices)
            for piece_id, indices in local_by_piece.items()
        }
    return {"tensors": tensors, "piece_to_local": piece_to_local}


def build_samples(
    examples: list[dict[str, Any]],
    crop_records: dict[str, list[dict[str, Any]]],
    crop_embeddings: dict[str, np.ndarray],
    piece_to_local: dict[str, dict[int, tuple[int, ...]]],
) -> dict[str, list[TrainingSample]]:
    samples: dict[str, list[TrainingSample]] = {slot: [] for slot in SLOTS}
    for example in examples:
        labels = example.get("core_slot_labels", {})
        for record in crop_records.get(example["example_id"], []):
            slot = record["slot"]
            label = labels.get(slot)
            path = str(record["path"])
            if (
                not label
                or not label.get("piece_ids")
                or path not in crop_embeddings
                or VISIBLE_WEIGHTS.get(record["visibility"], 0.0) <= 0.0
            ):
                continue
            positives = sorted(
                {
                    candidate_index
                    for piece_id in label["piece_ids"]
                    for candidate_index in piece_to_local[slot].get(
                        int(piece_id),
                        (),
                    )
                }
            )
            if not positives:
                continue
            visibility_weight = VISIBLE_WEIGHTS[record["visibility"]]
            confidence = max(0.35, min(1.0, float(record.get("confidence", 0.0))))
            samples[slot].append(
                TrainingSample(
                    path=path,
                    positives=tuple(positives),
                    weight=visibility_weight * confidence,
                )
            )
    return samples


def train_epoch(
    adapter: SlotResidualAdapter,
    optimizer: torch.optim.Optimizer,
    samples: dict[str, list[TrainingSample]],
    crop_embeddings: dict[str, np.ndarray],
    candidates: dict[str, torch.Tensor],
    batch_size: int,
    temperature: float,
    identity_penalty: float,
    device: torch.device,
) -> float:
    adapter.train()
    losses = []
    for slot in SLOTS:
        order = torch.randperm(len(samples[slot])).tolist()
        for start in range(0, len(order), batch_size):
            batch = [samples[slot][index] for index in order[start : start + batch_size]]
            queries = torch.tensor(
                np.stack([crop_embeddings[sample.path] for sample in batch]),
                device=device,
            )
            projected = adapter(queries, slot)
            logits = projected @ candidates[slot].T / temperature
            denominator = torch.logsumexp(logits, dim=1)
            numerator = torch.stack(
                [
                    torch.logsumexp(
                        logits[index, list(sample.positives)],
                        dim=0,
                    )
                    for index, sample in enumerate(batch)
                ]
            )
            weights = torch.tensor(
                [sample.weight for sample in batch],
                device=device,
            )
            contrastive = ((denominator - numerator) * weights).sum() / weights.sum()
            identity = (1.0 - (projected * queries).sum(dim=1)).mean()
            loss = contrastive + identity_penalty * identity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


@torch.inference_mode()
def crop_recall(
    adapter: SlotResidualAdapter,
    samples: dict[str, list[TrainingSample]],
    crop_embeddings: dict[str, np.ndarray],
    candidates: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    adapter.eval()
    count = hits_at_1 = hits_at_10 = 0
    by_slot = {}
    for slot in SLOTS:
        slot_count = slot_hits_at_1 = slot_hits_at_10 = 0
        for start in range(0, len(samples[slot]), batch_size):
            batch = samples[slot][start : start + batch_size]
            queries = torch.tensor(
                np.stack([crop_embeddings[sample.path] for sample in batch]),
                device=device,
            )
            projected = adapter(queries, slot)
            top = torch.topk(
                projected @ candidates[slot].T,
                k=min(10, len(candidates[slot])),
                dim=1,
            ).indices.cpu().tolist()
            for sample, indices in zip(batch, top, strict=True):
                positive_set = set(sample.positives)
                count += 1
                hits_at_1 += bool(indices and indices[0] in positive_set)
                hits_at_10 += bool(positive_set.intersection(indices))
                slot_count += 1
                slot_hits_at_1 += bool(indices and indices[0] in positive_set)
                slot_hits_at_10 += bool(positive_set.intersection(indices))
        by_slot[slot] = {
            "sample_count": slot_count,
            "recall_at_1": slot_hits_at_1 / slot_count if slot_count else 0.0,
            "recall_at_10": slot_hits_at_10 / slot_count if slot_count else 0.0,
        }
    return {
        "sample_count": count,
        "recall_at_1": hits_at_1 / count if count else 0.0,
        "recall_at_10": hits_at_10 / count if count else 0.0,
        "by_slot": by_slot,
    }


@torch.inference_mode()
def project_crop_embeddings(
    adapter: SlotResidualAdapter,
    crop_records: dict[str, list[dict[str, Any]]],
    crop_embeddings: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    adapter.eval()
    path_slots = {
        str(record["path"]): record["slot"]
        for records in crop_records.values()
        for record in records
        if str(record["path"]) in crop_embeddings
    }
    projected = {}
    for slot in SLOTS:
        paths = sorted(path for path, value in path_slots.items() if value == slot)
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            queries = torch.tensor(
                np.stack([crop_embeddings[path] for path in batch_paths]),
                device=device,
            )
            outputs = adapter(queries, slot).cpu().numpy()
            projected.update(zip(batch_paths, outputs, strict=True))
    return projected


def select_on_validation_and_evaluate_test(
    split_examples: dict[str, list[dict[str, Any]]],
    split_records: dict[str, dict[str, list[dict[str, Any]]]],
    search_results: dict[str, list[dict[str, Any]]],
    available_piece_ids: dict[str, set[int]],
    available_appearance_keys: dict[str, set[str]],
    top_k: int,
) -> dict[str, Any]:
    evaluated = []
    for role_profile in sorted(ROLE_PROFILES):
        for tag_profile in sorted(TAG_PENALTY_PROFILES):
            for aggregate in ("max", "sum"):
                outcomes = evaluate_config(
                    split_examples["validation"],
                    split_records["validation"],
                    search_results,
                    available_piece_ids,
                    available_appearance_keys,
                    top_k,
                    ROLE_PROFILES[role_profile],
                    TAG_PENALTY_PROFILES[tag_profile],
                    aggregate,
                )
                summary = summarize(outcomes, top_k)
                summary.update(
                    role_profile=role_profile,
                    tag_penalty_profile=tag_profile,
                    aggregate=aggregate,
                )
                evaluated.append(summary)
    best_validation = max(
        evaluated,
        key=lambda summary: (
            summary["appearance_recall_at_1"],
            summary["appearance_mrr"],
            summary["appearance_recall_at_k"],
        ),
    )
    validation_outcomes = evaluate_config(
        split_examples["validation"],
        split_records["validation"],
        search_results,
        available_piece_ids,
        available_appearance_keys,
        top_k,
        ROLE_PROFILES[best_validation["role_profile"]],
        TAG_PENALTY_PROFILES[best_validation["tag_penalty_profile"]],
        best_validation["aggregate"],
    )
    test_outcomes = evaluate_config(
        split_examples["test"],
        split_records["test"],
        search_results,
        available_piece_ids,
        available_appearance_keys,
        top_k,
        ROLE_PROFILES[best_validation["role_profile"]],
        TAG_PENALTY_PROFILES[best_validation["tag_penalty_profile"]],
        best_validation["aggregate"],
    )
    test_summary = summarize(test_outcomes, top_k)
    test_summary.update(
        role_profile=best_validation["role_profile"],
        tag_penalty_profile=best_validation["tag_penalty_profile"],
        aggregate=best_validation["aggregate"],
    )
    return {
        "validation": best_validation,
        "validation_cases": validation_outcomes,
        "test": test_summary,
        "test_cases": test_outcomes,
    }


def evaluate_cascade(
    precision_model: dict[str, Any],
    recall_model: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    validation_cases = cascade_outcomes(
        precision_model["validation_cases"],
        recall_model["validation_cases"],
        top_k,
    )
    test_cases = cascade_outcomes(
        precision_model["test_cases"],
        recall_model["test_cases"],
        top_k,
    )
    return {
        "strategy": "precision_top1_then_recall_tail",
        "validation": summarize(validation_cases, top_k),
        "validation_cases": validation_cases,
        "test": summarize(test_cases, top_k),
        "test_cases": test_cases,
    }


def cascade_outcomes(
    precision_outcomes: list[dict[str, Any]],
    recall_outcomes: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    recall_by_case = {
        (outcome["example_id"], outcome["slot"]): outcome
        for outcome in recall_outcomes
    }
    combined = []
    for precision in precision_outcomes:
        key = (precision["example_id"], precision["slot"])
        recall = recall_by_case[key]
        outcome = dict(precision)
        if precision["status"] != "eligible":
            combined.append(outcome)
            continue
        sequence = [
            *precision.get("results", [])[:1],
            *recall.get("results", []),
            *precision.get("results", [])[1:],
        ]
        seen = set()
        results = []
        for result in sequence:
            identity = result.get("appearance_key") or f"piece:{result['piece_id']}"
            if identity in seen:
                continue
            seen.add(identity)
            row = dict(result)
            row["rank"] = len(results) + 1
            results.append(row)
            if len(results) >= top_k:
                break
        expected_ids = {
            int(piece_id)
            for piece_id in outcome.get("eligible_expected_piece_ids", [])
        }
        expected_key = outcome.get("expected_appearance_key")
        exact_rank = next(
            (
                result["rank"]
                for result in results
                if int(result["piece_id"]) in expected_ids
            ),
            None,
        )
        appearance_rank = next(
            (
                result["rank"]
                for result in results
                if result.get("appearance_key") == expected_key
            ),
            None,
        )
        outcome.update(
            results=results,
            rank=exact_rank,
            appearance_rank=appearance_rank,
            exact_hit_at_1=exact_rank == 1,
            exact_hit_at_k=exact_rank is not None and exact_rank <= top_k,
            appearance_hit_at_1=appearance_rank == 1,
            appearance_hit_at_k=(
                appearance_rank is not None and appearance_rank <= top_k
            ),
            hit_at_1=appearance_rank == 1,
            hit_at_k=appearance_rank is not None and appearance_rank <= top_k,
        )
        combined.append(outcome)
    return combined


def evaluate_adapted_fusion(
    *,
    metadata: list[dict[str, Any]],
    first_candidate_embeddings: np.ndarray,
    first_crop_embeddings: dict[str, np.ndarray],
    crop_records: dict[str, list[dict[str, Any]]],
    split_examples: dict[str, list[dict[str, Any]]],
    split_records: dict[str, dict[str, list[dict[str, Any]]]],
    available_piece_ids: dict[str, set[int]],
    available_appearance_keys: dict[str, set[str]],
    second_index: Path,
    second_crops: Path,
    first_weights: list[float],
    candidate_k: int,
    top_k: int,
    device: str,
) -> dict[str, Any]:
    second_metadata = read_jsonl(second_index / "metadata.jsonl")
    if [row["local_path"] for row in metadata] != [
        row["local_path"] for row in second_metadata
    ]:
        raise ValueError("Adapted fusion indexes are not candidate-aligned")
    second_candidate_embeddings = np.load(
        second_index / "embeddings.npy",
        mmap_mode="r",
    )
    second_crop_embeddings = load_crop_embeddings(second_crops)
    evaluated = []
    for first_weight in first_weights:
        search_results = build_fused_search_results(
            metadata=metadata,
            first_embeddings=first_candidate_embeddings,
            first_crop_embeddings=first_crop_embeddings,
            second_embeddings=second_candidate_embeddings,
            second_crop_embeddings=second_crop_embeddings,
            crop_records=crop_records,
            first_weight=first_weight,
            candidate_k=candidate_k,
            device=device,
        )
        result = select_on_validation_and_evaluate_test(
            split_examples,
            split_records,
            search_results,
            available_piece_ids,
            available_appearance_keys,
            top_k,
        )
        result["first_weight"] = first_weight
        result["second_weight"] = 1.0 - first_weight
        evaluated.append(result)
        validation = result["validation"]
        print(
            f"adapted_fusion weights={first_weight:.2f}/{1.0-first_weight:.2f} "
            f"val@1={validation['appearance_recall_at_1']:.3f} "
            f"val@10={validation['appearance_recall_at_k']:.3f}",
            flush=True,
        )
    best = max(
        evaluated,
        key=lambda result: (
            result["validation"]["appearance_recall_at_1"],
            result["validation"]["appearance_mrr"],
            result["validation"]["appearance_recall_at_k"],
        ),
    )
    return {
        **best,
        "first_index": portable_path(DEFAULT_INDEX),
        "second_index": portable_path(second_index),
        "second_crops": portable_path(second_crops),
        "sweep_results": [
            {
                "first_weight": result["first_weight"],
                "second_weight": result["second_weight"],
                "validation": result["validation"],
            }
            for result in evaluated
        ],
    }


def print_split_summary(
    split_map: dict[str, str],
    split_examples: dict[str, list[dict[str, Any]]],
    samples: dict[str, dict[str, list[TrainingSample]]],
) -> None:
    for split in ("train", "validation", "test"):
        glamour_count = sum(value == split for value in split_map.values())
        sample_count = sum(len(rows) for rows in samples[split].values())
        print(
            f"{split}: glamours={glamour_count} "
            f"images={len(split_examples[split])} crops={sample_count}"
        )


def print_comparison(
    baseline: dict[str, Any],
    adapted: dict[str, Any],
    label: str = "adapted",
) -> None:
    print(
        f"held_out_test model={label} "
        f"baseline@1={baseline['appearance_recall_at_1']:.3f} "
        f"adapted@1={adapted['appearance_recall_at_1']:.3f} "
        f"baseline@10={baseline['appearance_recall_at_k']:.3f} "
        f"adapted@10={adapted['appearance_recall_at_k']:.3f} "
        f"baseline_mrr={baseline['appearance_mrr']:.3f} "
        f"adapted_mrr={adapted['appearance_mrr']:.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
