from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import CACHE_DIR, PROCESSED_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.evaluation_domains import build_domain_balance_weights
from ffxiv_prism.vision.evaluation_quality import filter_examples_by_quality
from ffxiv_prism.vision.evaluation_split import (
    development_split_map,
    load_split_assignments,
)
from scripts.evaluate_glamour_matching import (
    SLOTS,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    collect_crop_records,
)
from scripts.train_glamour_adapter import (
    build_candidate_state,
    build_samples,
    select_on_validation_and_evaluate_test,
)
from scripts.train_glamour_query_encoder import (
    encode_records,
    filter_crop_records_by_slots,
    group_training_samples,
    load_query_head_from_checkpoint,
    prune_training_samples,
    set_seed,
)

DEFAULT_INDEX = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_REGIONS = Path("data/debug/outfit-regions-v3-624")
DEFAULT_QUERY_CHECKPOINT = Path(
    "data/models/glamour_query_encoder_v5_160g_grouped.pt"
)
DEFAULT_SPLIT = PROCESSED_DIR / "glamour_split_v2.json"
DEFAULT_CHECKPOINT = Path("data/models/glamour_pair_reranker.pt")
DEFAULT_OUTPUT = PROCESSED_DIR / "glamour_pair_reranker_eval.json"
DEFAULT_QUERY_CACHE = PROCESSED_DIR / "glamour_query_embeddings_v5_624.npz"


def combine_group_weights(
    *weight_maps: dict[str, float],
) -> dict[str, float]:
    group_ids = {
        group_id
        for weight_map in weight_maps
        for group_id in weight_map
    }
    return {
        group_id: float(
            np.prod(
                [
                    weight_map.get(group_id, 1.0)
                    for weight_map in weight_maps
                ]
            )
        )
        for group_id in group_ids
    }


from ffxiv_prism.matching.runtime.reranker import PairReranker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a nonlinear query-candidate reranker on mined hard negatives."
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROCESSED_DIR / "glamour_examples.jsonl",
    )
    parser.add_argument(
        "--quality-manifest",
        type=Path,
        help=(
            "Optional JSONL quality manifest. Only examples with passed=true "
            "are used for training and evaluation."
        ),
    )
    parser.add_argument(
        "--slots",
        nargs="+",
        choices=SLOTS,
        default=list(SLOTS),
        help="Train and evaluate only the selected equipment slots.",
    )
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--query-checkpoint",
        type=Path,
        default=DEFAULT_QUERY_CHECKPOINT,
    )
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Continue reranker and optimizer state from a segmented run.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--fixed-epochs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train for exactly --epochs without inspecting validation metrics "
            "between epochs. This avoids selecting a checkpoint on the public "
            "validation split."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hard-image-k", type=int, default=128)
    parser.add_argument("--retrieval-image-k", type=int, default=400)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-crops-per-example-slot", type=int, default=2)
    parser.add_argument(
        "--query-aggregate",
        choices=("logmeanexp", "max"),
        default="logmeanexp",
        help=(
            "Aggregate multiple query crops/views during training. max aligns "
            "the loss with the strongest-evidence inference configuration."
        ),
    )
    parser.add_argument(
        "--candidate-view-aggregate",
        choices=("max", "top2_mean"),
        default="max",
        help=(
            "Collapse clean-render crop variants per piece by max score, or "
            "average the two best distinct candidate crop types."
        ),
    )
    parser.add_argument(
        "--per-slot-heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a shared feature trunk with independent output heads per equipment slot.",
    )
    parser.add_argument(
        "--group-by-glamour",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Aggregate all training views sharing one glamour label.",
    )
    parser.add_argument(
        "--domain-balanced",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply capped inverse-sqrt race/gender training weights.",
    )
    parser.add_argument("--domain-balance-max-weight", type=float, default=3.0)
    parser.add_argument(
        "--replay-quality-manifest",
        type=Path,
        help=(
            "Optional older quality manifest whose accepted train groups are "
            "upweighted to reduce forgetting during data expansion."
        ),
    )
    parser.add_argument(
        "--replay-weight",
        type=float,
        default=1.0,
        help="Training weight for groups accepted by --replay-quality-manifest.",
    )
    args = parser.parse_args()
    if args.replay_weight <= 0:
        parser.error("--replay-weight must be positive")
    if args.replay_weight != 1.0 and args.replay_quality_manifest is None:
        parser.error(
            "--replay-weight requires --replay-quality-manifest"
        )

    set_seed(args.seed)
    device = resolve_torch_device(args.device)
    args.device = str(device)
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(
        args.index_dir / "embeddings.npy",
        mmap_mode="r",
    )
    examples = read_jsonl(args.examples)
    if args.quality_manifest is not None:
        examples = filter_examples_by_quality(
            examples,
            read_jsonl(args.quality_manifest),
        )
        if not examples:
            raise ValueError("Quality manifest accepted no training examples")
    records = collect_crop_records(examples, args.regions_dir)
    records = filter_crop_records_by_slots(records, args.slots)
    assignments = load_split_assignments(args.split_manifest)
    split_map = development_split_map(
        {str(example["glamour_id"]) for example in examples},
        assignments,
    )
    development_examples = [
        example
        for example in examples
        if str(example["glamour_id"]) in split_map
    ]
    split_examples = {
        split: [
            example
            for example in development_examples
            if split_map[str(example["glamour_id"])] == split
        ]
        for split in ("train", "validation", "test")
    }
    split_records = {
        split: {
            example["example_id"]: records.get(example["example_id"], [])
            for example in split_examples[split]
        }
        for split in split_examples
    }
    development_records = {
        **split_records["train"],
        **split_records["validation"],
        **split_records["test"],
    }

    query_embeddings = load_or_build_query_embeddings(
        args,
        development_records,
        device,
    )
    candidate_state = build_candidate_state(
        metadata,
        candidate_embeddings,
        device,
    )
    samples = build_samples(
        split_examples["train"],
        split_records["train"],
        query_embeddings,
        candidate_state["piece_to_local"],
    )
    samples = prune_training_samples(
        samples,
        split_records["train"],
        args.max_crops_per_example_slot,
    )
    training_group_ids = (
        {
            str(example["example_id"]): str(example["glamour_id"])
            for example in split_examples["train"]
        }
        if args.group_by_glamour
        else None
    )
    domain_weights, domain_balance_report = (
        build_domain_balance_weights(
            split_examples["train"],
            group_by_glamour=args.group_by_glamour,
            max_weight=args.domain_balance_max_weight,
        )
        if args.domain_balanced
        else ({}, None)
    )
    replay_weights: dict[str, float] = {}
    replay_report = None
    if args.replay_quality_manifest is not None:
        replay_example_ids = {
            str(row["example_id"])
            for row in read_jsonl(args.replay_quality_manifest)
            if row.get("passed")
        }
        replay_group_ids = {
            (
                str(example["glamour_id"])
                if args.group_by_glamour
                else str(example["example_id"])
            )
            for example in split_examples["train"]
            if str(example["example_id"]) in replay_example_ids
        }
        replay_weights = {
            group_id: args.replay_weight
            for group_id in replay_group_ids
        }
        replay_report = {
            "quality_manifest": portable_path(
                args.replay_quality_manifest
            ),
            "weight": args.replay_weight,
            "weighted_group_count": len(replay_group_ids),
        }
    combined_group_weights = combine_group_weights(
        domain_weights,
        replay_weights,
    )
    grouped_samples = group_training_samples(
        samples,
        split_records["train"],
        group_ids=training_group_ids,
        group_weights=combined_group_weights,
    )

    reranker = PairReranker(
        int(candidate_embeddings.shape[1]),
        per_slot_heads=args.per_slot_heads,
        dropout=args.dropout,
    ).to(device)
    resume_payload = None
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(
            args.resume_checkpoint,
            map_location=device,
            weights_only=True,
        )
        reranker.load_state_dict(resume_payload["state_dict"])

    optimizer = torch.optim.AdamW(
        reranker.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if resume_payload and resume_payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])

    available_piece_ids = available_pieces_by_slot(metadata)
    available_appearance_keys = available_appearance_keys_by_slot(metadata)
    best_state = None
    start_epoch = int((resume_payload or {}).get("epochs_completed", 0))
    best_epoch = start_epoch
    best_score = (-1.0, -1.0, -1.0)
    stale_epochs = 0
    history = list((resume_payload or {}).get("history", []))
    if args.fixed_epochs and args.epochs == 0:
        best_state = copy.deepcopy(reranker.state_dict())

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        loss = train_epoch(
            reranker,
            optimizer,
            grouped_samples,
            query_embeddings,
            candidate_state["tensors"],
            args.hard_image_k,
            args.batch_size,
            device,
            query_aggregate=args.query_aggregate,
        )
        if args.fixed_epochs:
            best_epoch = epoch
            best_state = copy.deepcopy(reranker.state_dict())
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "validation": None,
                }
            )
            print(
                f"epoch={epoch} loss={loss:.4f} "
                "validation=deferred fixed_epochs=true",
                flush=True,
            )
        else:
            validation_search = rerank_crops(
                reranker,
                metadata,
                candidate_embeddings,
                split_records["validation"],
                query_embeddings,
                args.retrieval_image_k,
                args.candidate_k,
                device,
                candidate_view_aggregate=args.candidate_view_aggregate,
            )
            selection = select_on_validation_and_evaluate_test(
                {
                    "validation": split_examples["validation"],
                    "test": split_examples["validation"],
                },
                {
                    "validation": split_records["validation"],
                    "test": split_records["validation"],
                },
                validation_search,
                available_piece_ids,
                available_appearance_keys,
                args.top_k,
            )
            validation = selection["validation"]
            score = (
                validation["appearance_recall_at_1"],
                validation["appearance_mrr"],
                validation["appearance_recall_at_k"],
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(reranker.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "validation": validation,
                }
            )
            print(
                f"epoch={epoch} loss={loss:.4f} "
                f"val@1={validation['appearance_recall_at_1']:.3f} "
                f"val@{args.top_k}={validation['appearance_recall_at_k']:.3f} "
                f"mrr={validation['appearance_mrr']:.3f} best_epoch={best_epoch}",
                flush=True,
            )
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Reranker training produced no checkpoint")
    reranker.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "state_dict": best_state,
            "dimension": reranker.dimension,
            "per_slot_heads": reranker.per_slot_heads,
            "group_by_glamour": args.group_by_glamour,
            "domain_balanced": args.domain_balanced,
            "query_aggregate": args.query_aggregate,
            "fixed_epochs": args.fixed_epochs,
            "dropout": args.dropout,
            "candidate_view_aggregate": args.candidate_view_aggregate,
            "domain_balance": domain_balance_report,
            "replay": replay_report,
            "query_checkpoint": portable_path(args.query_checkpoint),
            "best_epoch": best_epoch,
            "seed": args.seed,
            "optimizer_state_dict": optimizer.state_dict(),
            "epochs_completed": history[-1]["epoch"] if history else start_epoch,
            "history": history,
        },
        args.checkpoint,
    )
    if args.skip_evaluation:
        payload = {
            "model": {
                "checkpoint": portable_path(args.checkpoint),
                "query_checkpoint": portable_path(args.query_checkpoint),
                "best_epoch": best_epoch,
            },
            "training": {
                "fixed_epochs": args.fixed_epochs,
                "resume_checkpoint": portable_path(args.resume_checkpoint),
                "slots": list(args.slots),
                "history": history,
            },
            "evaluation": {},
            "index_dir": portable_path(args.index_dir),
            "split_manifest": portable_path(args.split_manifest),
        }
        write_json(args.output, payload)
        print(f"checkpoint={args.checkpoint} output={args.output} evaluation=skipped")
        return 0


    relevant_records = {
        **split_records["validation"],
        **split_records["test"],
    }
    search = rerank_crops(
        reranker,
        metadata,
        candidate_embeddings,
        relevant_records,
        query_embeddings,
        args.retrieval_image_k,
        args.candidate_k,
        device,
        candidate_view_aggregate=args.candidate_view_aggregate,
    )
    evaluation = select_on_validation_and_evaluate_test(
        split_examples,
        split_records,
        search,
        available_piece_ids,
        available_appearance_keys,
        args.top_k,
    )
    payload = {
        "model": {
            "type": "nonlinear_pair_reranker",
            "checkpoint": portable_path(args.checkpoint),
            "query_checkpoint": portable_path(args.query_checkpoint),
            "parameter_count": sum(
                parameter.numel() for parameter in reranker.parameters()
            ),
            "best_epoch": best_epoch,
            "per_slot_heads": reranker.per_slot_heads,
        },
        "training": {
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "hard_image_k": args.hard_image_k,
            "group_by_glamour": args.group_by_glamour,
            "query_aggregate": args.query_aggregate,
            "fixed_epochs": args.fixed_epochs,
            "dropout": args.dropout,
            "candidate_view_aggregate": args.candidate_view_aggregate,
            "domain_balanced": args.domain_balanced,
            "domain_balance": domain_balance_report,
            "replay": replay_report,
            "quality_manifest": portable_path(args.quality_manifest),
            "quality_example_count": len(examples),
            "slots": list(args.slots),
            "history": history,
        },
        "evaluation": evaluation,
        "split_manifest": portable_path(args.split_manifest),
        "regions_dir": portable_path(args.regions_dir),
        "index_dir": portable_path(args.index_dir),
        "retrieval_image_k": args.retrieval_image_k,
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
    }
    write_json(args.output, payload)
    test = evaluation["test"]
    print(
        f"test@1={test['appearance_recall_at_1']:.3f} "
        f"test@{args.top_k}={test['appearance_recall_at_k']:.3f} "
        f"mrr={test['appearance_mrr']:.3f} "
        f"checkpoint={args.checkpoint} output={args.output}"
    )
    return 0


def load_or_build_query_embeddings(
    args: argparse.Namespace,
    records: dict[str, list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    paths = sorted(
        {
            str(record["path"])
            for rows in records.values()
            for record in rows
        }
    )
    if args.query_cache.is_file():
        with np.load(args.query_cache, allow_pickle=False) as cached:
            cached_paths = [str(value) for value in cached["paths"].tolist()]
            cached_checkpoint = str(cached["checkpoint"].tolist())
            cached_embeddings = (
                np.asarray(cached["embeddings"]).copy()
                if cached_paths == paths
                and cached_checkpoint == str(args.query_checkpoint)
                else None
            )
        if cached_embeddings is not None:
            return {
                path: embedding
                for path, embedding in zip(
                    paths,
                    cached_embeddings,
                    strict=True,
                )
            }

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
        batch_size=32,
        device=device,
    )
    args.query_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.query_cache,
        paths=np.asarray(paths),
        embeddings=np.stack([embeddings[path] for path in paths]),
        checkpoint=np.asarray(str(args.query_checkpoint)),
    )
    return embeddings


def train_epoch(
    reranker: PairReranker,
    optimizer: torch.optim.Optimizer,
    samples: dict[str, list[Any]],
    query_embeddings: dict[str, np.ndarray],
    candidates: dict[str, torch.Tensor],
    hard_image_k: int,
    batch_size: int,
    device: torch.device,
    *,
    query_aggregate: str = "logmeanexp",
) -> float:
    reranker.train()
    losses = []
    for slot in SLOTS:
        order = torch.randperm(len(samples[slot])).tolist()
        for start in range(0, len(order), batch_size):
            batch = [
                samples[slot][index]
                for index in order[start : start + batch_size]
            ]
            sample_losses = []
            for sample in batch:
                queries = torch.tensor(
                    np.stack(
                        [query_embeddings[path] for path in sample.paths]
                    ),
                    device=device,
                )
                base = queries @ candidates[slot].T
                aggregate_base = base.max(dim=0).values
                hard = torch.topk(
                    aggregate_base,
                    k=min(hard_image_k, aggregate_base.numel()),
                ).indices.tolist()
                candidate_indices = list(
                    dict.fromkeys([*hard, *sample.positives])
                )
                candidate_tensor = candidates[slot][candidate_indices]
                crop_scores = []
                for query in queries:
                    crop_scores.append(
                        reranker(
                            query.expand(len(candidate_tensor), -1),
                            candidate_tensor,
                            slot,
                        )
                    )
                scores = aggregate_crop_scores(
                    torch.stack(crop_scores),
                    query_aggregate,
                )
                positive_set = set(sample.positives)
                positive_positions = [
                    index
                    for index, candidate_index in enumerate(candidate_indices)
                    if candidate_index in positive_set
                ]
                numerator = torch.logsumexp(
                    scores[positive_positions],
                    dim=0,
                )
                denominator = torch.logsumexp(scores, dim=0)
                sample_losses.append(
                    (denominator - numerator) * sample.weight
                )
            loss = torch.stack(sample_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def aggregate_crop_scores(
    crop_scores: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    if crop_scores.ndim < 1 or crop_scores.shape[0] < 1:
        raise ValueError("crop_scores must contain at least one query")
    if strategy == "max":
        return crop_scores.max(dim=0).values
    if strategy == "logmeanexp":
        return torch.logsumexp(crop_scores, dim=0) - np.log(
            crop_scores.shape[0]
        )
    raise ValueError(f"Unsupported query aggregation: {strategy}")


from ffxiv_prism.matching.runtime.reranker import rerank_crops


from ffxiv_prism.matching.runtime.reranker import aggregate_candidate_views


if __name__ == "__main__":
    raise SystemExit(main())
