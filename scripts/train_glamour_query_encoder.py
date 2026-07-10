from __future__ import annotations

import argparse
import copy
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
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
from scripts.evaluate_glamour_embedding_model import (
    DINO_MEAN,
    DINO_STD,
    search_crops,
)
from scripts.evaluate_glamour_matching import (
    ROLE_PROFILES,
    SLOTS,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    collect_crop_records,
    crop_record_weight,
)
from scripts.train_glamour_adapter import (
    build_candidate_state,
    build_samples,
    deterministic_glamour_split,
    select_on_validation_and_evaluate_test,
)

DEFAULT_INDEX = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_RAW_REGIONS = Path("data/debug/outfit-regions/overlap-final-201")
DEFAULT_FOREGROUND_REGIONS = Path("data/debug/global-foreground-v1")
DEFAULT_OUTPUT = PROCESSED_DIR / "glamour_query_encoder_eval.json"
DEFAULT_CHECKPOINT = Path("data/models/glamour_query_encoder.pt")
DEFAULT_SPLIT_MANIFEST = PROCESSED_DIR / "glamour_split_v2.json"


@dataclass(frozen=True)
class GroupedTrainingSample:
    paths: tuple[str, ...]
    positives: tuple[int, ...]
    weight: float


class SlotAttentionPooler(nn.Module):
    """Learn a small slot-specific residual from local DINO patch tokens."""

    def __init__(self, dimension: int, slots: Sequence[str] = SLOTS) -> None:
        super().__init__()
        self.dimension = dimension
        self.queries = nn.ParameterDict(
            {
                slot: nn.Parameter(torch.randn(dimension) * 0.02)
                for slot in slots
            }
        )
        self.gates = nn.ParameterDict(
            {
                slot: nn.Parameter(torch.tensor(-3.0))
                for slot in slots
            }
        )
        self.output = nn.ModuleDict(
            {
                slot: nn.Linear(dimension, dimension, bias=False)
                for slot in slots
            }
        )
        for layer in self.output.values():
            nn.init.zeros_(layer.weight)

    def forward(self, tokens: torch.Tensor, slot: str) -> torch.Tensor:
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        attention = torch.softmax(
            patches @ self.queries[slot] / np.sqrt(self.dimension),
            dim=1,
        )
        local = (attention.unsqueeze(-1) * patches).sum(dim=1)
        residual = self.output[slot](local)
        gate = torch.sigmoid(self.gates[slot])
        return torch.nn.functional.normalize(cls + gate * residual, dim=-1)


class LowRankResidualPooler(nn.Module):
    """Identity-initialized low-rank residual over the frozen CLS token."""

    def __init__(self, dimension: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("adapter rank must be positive")
        self.rank = rank
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens: torch.Tensor, slot: str) -> torch.Tensor:
        del slot
        cls = tokens[:, 0]
        residual = self.up(torch.nn.functional.gelu(self.down(cls)))
        return torch.nn.functional.normalize(cls + residual, dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the final DINOv2 query-encoder blocks against fixed "
            "canonical-render embeddings."
        )
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--preprocess",
        choices=("color", "gray"),
        default="color",
        help="Apply the same color or grayscale preprocessing to query crops.",
    )
    parser.add_argument(
        "--candidate-race",
        help="Restrict canonical candidate images to one normalized race value.",
    )
    parser.add_argument(
        "--candidate-model-gender",
        help="Restrict canonical candidate images to one normalized gender value.",
    )
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
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_RAW_REGIONS)
    parser.add_argument(
        "--alternate-regions-dir",
        type=Path,
        default=DEFAULT_FOREGROUND_REGIONS,
    )
    parser.add_argument("--alternate-probability", type=float, default=0.35)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Continue from a compatible query-encoder checkpoint.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument(
        "--adapter-rank",
        type=int,
        default=0,
        help=(
            "Freeze DINOv2 and train an identity-initialized low-rank "
            "residual query head. Zero keeps last-block fine-tuning."
        ),
    )
    parser.add_argument(
        "--slot-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--fixed-epochs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train for exactly --epochs and defer validation evaluation until "
            "the checkpoint is frozen."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-k", type=int, default=20)
    parser.add_argument(
        "--hard-negative-margin",
        type=float,
        default=0.5,
        help="Margin in temperature-scaled logit units.",
    )
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--max-crops-per-example-slot",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--group-by-glamour",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Aggregate all training views of one glamour before applying the "
            "equipment-label loss, reducing occluded-view label noise."
        ),
    )
    parser.add_argument(
        "--domain-balanced",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply capped inverse-sqrt race/gender weights to training groups. "
            "Validation remains unweighted."
        ),
    )
    parser.add_argument(
        "--domain-balance-max-weight",
        type=float,
        default=3.0,
        help="Maximum raw upweight applied to a rare race/gender domain.",
    )
    parser.add_argument(
        "--foreground-review-queue",
        type=Path,
        default=None,
        help=(
            "Optional global-foreground review queue. Alternate crops with "
            "automatic quality flags are excluded unless visually accepted."
        ),
    )
    parser.add_argument(
        "--vision-reviews",
        type=Path,
        default=PROCESSED_DIR / "foreground_vision_reviews_v1.jsonl",
        help="Optional human/vision-model image and slot usability reviews.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Save the training checkpoint without the final retrieval evaluation.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.alternate_probability <= 1.0:
        raise ValueError("alternate-probability must be between 0 and 1")
    if args.epochs < 0:
        parser.error("--epochs must not be negative")
    if args.epochs == 0 and not args.fixed_epochs:
        parser.error("--epochs 0 requires --fixed-epochs")
    if args.adapter_rank < 0:
        parser.error("--adapter-rank must not be negative")
    if args.adapter_rank and args.slot_attention:
        parser.error("--adapter-rank cannot be combined with --slot-attention")

    set_seed(args.seed)
    device = resolve_torch_device(args.device)
    args.device = str(device)
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(
        args.index_dir / "embeddings.npy",
        mmap_mode="r",
    )
    if args.candidate_race or args.candidate_model_gender:
        candidate_indices = [
            index
            for index, row in enumerate(metadata)
            if (
                not args.candidate_race
                or str(row.get("race") or "").lower()
                == args.candidate_race.lower()
            )
            and (
                not args.candidate_model_gender
                or str(row.get("model_gender") or "").lower()
                == args.candidate_model_gender.lower()
            )
        ]
        if not candidate_indices:
            raise ValueError("Candidate race/gender filter accepted no index rows")
        metadata = [metadata[index] for index in candidate_indices]
        candidate_embeddings = np.asarray(
            candidate_embeddings[candidate_indices],
            dtype=np.float32,
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
    alternate_records = collect_crop_records(examples, args.alternate_regions_dir)
    records = filter_crop_records_by_slots(records, args.slots)
    alternate_records = filter_crop_records_by_slots(
        alternate_records,
        args.slots,
    )
    review_queue = args.foreground_review_queue
    if review_queue is None:
        candidate_queue = args.alternate_regions_dir / "vision_review_queue.jsonl"
        review_queue = candidate_queue if candidate_queue.is_file() else None
    alternate_allowlist = (
        load_foreground_allowlist(review_queue)
        if review_queue is not None and review_queue.is_file()
        else None
    )
    alternate_paths = align_alternate_paths(
        records,
        alternate_records,
        allowed_example_ids=alternate_allowlist,
    )

    if args.split_manifest.is_file():
        split_map = development_split_map(
            {str(example["glamour_id"]) for example in examples},
            load_split_assignments(args.split_manifest),
        )
    else:
        split_map = deterministic_glamour_split(
            {str(example["glamour_id"]) for example in examples},
            args.seed,
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
    vision_reviews = (
        load_vision_reviews(args.vision_reviews)
        if args.vision_reviews.is_file()
        else {}
    )
    split_records["train"] = apply_training_reviews(
        split_records["train"],
        vision_reviews,
    )

    candidate_state = build_candidate_state(
        metadata,
        candidate_embeddings,
        device,
    )
    placeholder_embeddings = {
        str(record["path"]): np.empty(0, dtype=np.float32)
        for rows in records.values()
        for record in rows
    }
    samples = build_samples(
        split_examples["train"],
        split_records["train"],
        placeholder_embeddings,
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
    grouped_samples = group_training_samples(
        samples,
        split_records["train"],
        group_ids=training_group_ids,
        group_weights=domain_weights,
    )

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    if args.adapter_rank:
        for parameter in model.parameters():
            parameter.requires_grad = False
        trainable_names: set[str] = set()
        pooler: nn.Module | None = LowRankResidualPooler(
            int(candidate_embeddings.shape[1]),
            args.adapter_rank,
        ).to(device)
    else:
        trainable_names = unfreeze_last_blocks(model, args.unfreeze_blocks)
        pooler = (
            SlotAttentionPooler(int(candidate_embeddings.shape[1])).to(device)
            if args.slot_attention
            else None
        )
    resume_payload = None
    if args.resume_checkpoint is not None:
        resume_payload = load_training_checkpoint(
            args.resume_checkpoint,
            model,
            pooler,
            device,
        )
    optimizer = torch.optim.AdamW(
        [
            *[
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            *([] if pooler is None else list(pooler.parameters())),
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if resume_payload and resume_payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
    )

    available_piece_ids = available_pieces_by_slot(metadata)
    available_appearance_keys = available_appearance_keys_by_slot(metadata)
    best_state = None
    best_pooler_state = None
    start_epoch = int((resume_payload or {}).get("epochs_completed", 0))
    best_epoch = start_epoch
    best_score = (-1.0, -1.0, -1.0)
    stale_epochs = 0
    history = list((resume_payload or {}).get("history", []))
    if args.fixed_epochs and args.epochs == 0:
        best_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.state_dict().items()
            if name in trainable_names
        }
        if pooler is not None:
            best_pooler_state = copy.deepcopy(pooler.state_dict())
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        loss = train_epoch(
            model=model,
            pooler=pooler,
            optimizer=optimizer,
            scaler=scaler,
            samples=grouped_samples,
            candidates=candidate_state["tensors"],
            alternate_paths=alternate_paths,
            alternate_probability=args.alternate_probability,
            batch_size=args.batch_size,
            temperature=args.temperature,
            hard_negative_weight=args.hard_negative_weight,
            hard_negative_k=args.hard_negative_k,
            hard_negative_margin=args.hard_negative_margin,
            device=device,
            augment=args.augment,
            preprocess=args.preprocess,
        )
        if args.fixed_epochs:
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
                if name in trainable_names
            }
            if pooler is not None:
                best_pooler_state = copy.deepcopy(pooler.state_dict())
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
            validation_embeddings = encode_records(
                model,
                pooler,
                split_records["validation"],
                args.batch_size,
                device,
                preprocess=args.preprocess,
            )
            validation_search = search_crops(
                metadata,
                candidate_embeddings,
                split_records["validation"],
                validation_embeddings,
                args.candidate_k,
                device=args.device,
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
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                    if name in trainable_names
                }
                if pooler is not None:
                    best_pooler_state = copy.deepcopy(pooler.state_dict())
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
        raise RuntimeError("Query encoder training produced no checkpoint")
    current = model.state_dict()
    current.update(best_state)
    model.load_state_dict(current)
    if pooler is not None:
        if best_pooler_state is None:
            raise RuntimeError("Slot-attention training produced no pooler state")
        pooler.load_state_dict(best_pooler_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "model_id": args.model_id,
            "unfreeze_blocks": args.unfreeze_blocks,
            "trainable_state_dict": best_state,
            "trainable_parameter_names": sorted(trainable_names),
            "slot_attention": args.slot_attention,
            "adapter_rank": args.adapter_rank,
            "query_head_type": (
                "low_rank_adapter"
                if args.adapter_rank
                else ("slot_attention" if args.slot_attention else "none")
            ),
            "pooler_state_dict": (
                None if pooler is None else pooler.state_dict()
            ),
            "best_epoch": best_epoch,
            "fixed_epochs": args.fixed_epochs,
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
                "model_id": args.model_id,
                "checkpoint": portable_path(args.checkpoint),
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


    evaluation = {}
    for branch, branch_records in (
        ("raw", split_records),
        (
            "foreground",
            split_records_for_examples(
                split_examples,
                alternate_records,
            ),
        ),
    ):
        relevant_records = {
            **branch_records["validation"],
            **branch_records["test"],
        }
        embeddings = encode_records(
            model,
            pooler,
            relevant_records,
            args.batch_size,
            device,
            preprocess=args.preprocess,
        )
        search = search_crops(
            metadata,
            candidate_embeddings,
            relevant_records,
            embeddings,
            args.candidate_k,
            device=args.device,
        )
        evaluation[branch] = select_on_validation_and_evaluate_test(
            split_examples,
            branch_records,
            search,
            available_piece_ids,
            available_appearance_keys,
            args.top_k,
        )

    payload = {
        "model": {
            "type": "dinov2_query_encoder_last_blocks",
            "model_id": args.model_id,
            "unfreeze_blocks": args.unfreeze_blocks,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            + (
                0
                if pooler is None
                else sum(parameter.numel() for parameter in pooler.parameters())
            ),
            "slot_attention": args.slot_attention,
            "adapter_rank": args.adapter_rank,
            "checkpoint": portable_path(args.checkpoint),
            "best_epoch": best_epoch,
        },
        "training": {
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "preprocess": args.preprocess,
            "augment": args.augment,
            "fixed_epochs": args.fixed_epochs,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "hard_negative_weight": args.hard_negative_weight,
            "hard_negative_k": args.hard_negative_k,
            "hard_negative_margin": args.hard_negative_margin,
            "resume_checkpoint": portable_path(args.resume_checkpoint),
            "alternate_probability": args.alternate_probability,
            "max_crops_per_example_slot": args.max_crops_per_example_slot,
            "group_by_glamour": args.group_by_glamour,
            "domain_balanced": args.domain_balanced,
            "domain_balance": domain_balance_report,
            "foreground_review_queue": portable_path(review_queue),
            "vision_reviews": portable_path(args.vision_reviews),
            "quality_manifest": portable_path(args.quality_manifest),
            "quality_example_count": len(examples),
            "candidate_race": args.candidate_race,
            "candidate_model_gender": args.candidate_model_gender,
            "slots": list(args.slots),
            "alternate_path_count": len(alternate_paths),
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
        "evaluation": evaluation,
        "index_dir": portable_path(args.index_dir),
        "regions_dir": portable_path(args.regions_dir),
        "alternate_regions_dir": portable_path(args.alternate_regions_dir),
        "split_manifest": portable_path(args.split_manifest),
    }
    write_json(args.output, payload)
    for branch, result in evaluation.items():
        test = result["test"]
        print(
            f"branch={branch} test@1={test['appearance_recall_at_1']:.3f} "
            f"test@{args.top_k}={test['appearance_recall_at_k']:.3f} "
            f"mrr={test['appearance_mrr']:.3f}"
        )
    print(f"checkpoint={args.checkpoint} output={args.output}")
    return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_crop_records_by_slots(
    records: dict[str, list[dict[str, Any]]],
    slots: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    allowed = set(slots)
    return {
        example_id: [
            record
            for record in rows
            if str(record.get("slot")) in allowed
        ]
        for example_id, rows in records.items()
    }


def unfreeze_last_blocks(
    model: torch.nn.Module,
    block_count: int,
) -> set[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if block_count < 1:
        raise ValueError("At least one encoder block must be trainable")
    blocks = model.encoder.layer
    if block_count > len(blocks):
        raise ValueError("Cannot unfreeze more blocks than the encoder contains")
    for block in blocks[-block_count:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.layernorm.parameters():
        parameter.requires_grad = True
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    pooler: nn.Module | None,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=True)
    state = model.state_dict()
    state.update(payload["trainable_state_dict"])
    model.load_state_dict(state)
    saved_attention = bool(payload.get("slot_attention", False))
    if saved_attention != isinstance(pooler, SlotAttentionPooler):
        raise ValueError(
            "Checkpoint slot-attention setting does not match this run"
        )
    saved_adapter_rank = int(payload.get("adapter_rank", 0))
    actual_adapter_rank = (
        pooler.rank
        if isinstance(pooler, LowRankResidualPooler)
        else 0
    )
    if saved_adapter_rank != actual_adapter_rank:
        raise ValueError(
            "Checkpoint low-rank adapter setting does not match this run"
        )
    if pooler is not None:
        pooler_state = payload.get("pooler_state_dict")
        if pooler_state is None:
            raise ValueError("Checkpoint has no slot-attention pooler state")
        pooler.load_state_dict(pooler_state)
    return payload


def load_query_head_from_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> nn.Module | None:
    payload = torch.load(path, map_location=device, weights_only=True)
    adapter_rank = int(payload.get("adapter_rank", 0))
    if adapter_rank:
        pooler: nn.Module | None = LowRankResidualPooler(
            int(model.config.hidden_size),
            adapter_rank,
        ).to(device)
    elif bool(payload.get("slot_attention", False)):
        pooler = SlotAttentionPooler(int(model.config.hidden_size)).to(device)
    else:
        pooler = None
    load_training_checkpoint(path, model, pooler, device)
    return pooler


def align_alternate_paths(
    primary: dict[str, list[dict[str, Any]]],
    alternate: dict[str, list[dict[str, Any]]],
    *,
    allowed_example_ids: set[str] | None = None,
) -> dict[str, str]:
    alternate_by_key = {
        (example_id, record["name"]): str(record["path"])
        for example_id, rows in alternate.items()
        for record in rows
    }
    return {
        str(record["path"]): alternate_by_key[(example_id, record["name"])]
        for example_id, rows in primary.items()
        for record in rows
        if (example_id, record["name"]) in alternate_by_key
        and (allowed_example_ids is None or example_id in allowed_example_ids)
    }


def load_foreground_allowlist(path: Path) -> set[str]:
    allowed = set()
    for row in read_jsonl(path):
        review = row.get("vision_review", {})
        decision = review.get("decision")
        flags = row.get("automatic_stats", {}).get("quality_flags", [])
        if decision == "accept" or (not flags and decision not in {"raw_only", "reject"}):
            allowed.add(str(row["example_id"]))
    return allowed


def load_vision_reviews(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["example_id"]): row
        for row in read_jsonl(path)
    }


def apply_training_reviews(
    records: dict[str, list[dict[str, Any]]],
    reviews: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    filtered = {}
    for example_id, rows in records.items():
        review = reviews.get(example_id)
        if review is None:
            filtered[example_id] = rows
            continue
        if review.get("decision") == "reject":
            filtered[example_id] = []
            continue
        usability = review.get("slot_usability", {})
        filtered[example_id] = [
            record
            for record in rows
            if usability.get(record["slot"], "usable") != "unusable"
        ]
    return filtered


def prune_training_samples(
    samples: dict[str, list[Any]],
    records: dict[str, list[dict[str, Any]]],
    max_per_example_slot: int,
) -> dict[str, list[Any]]:
    if max_per_example_slot < 1:
        raise ValueError("max-crops-per-example-slot must be positive")
    record_by_path = {
        str(record["path"]): record
        for rows in records.values()
        for record in rows
    }
    example_by_path = {
        str(record["path"]): example_id
        for example_id, rows in records.items()
        for record in rows
    }
    selected: dict[str, list[Any]] = {slot: [] for slot in SLOTS}
    for slot in SLOTS:
        grouped: dict[str, list[Any]] = defaultdict(list)
        for sample in samples[slot]:
            example_id = example_by_path.get(sample.path)
            if example_id is not None:
                grouped[example_id].append(sample)
        for rows in grouped.values():
            ranked = sorted(
                rows,
                key=lambda sample: (
                    sample.weight
                    * crop_record_weight(
                        record_by_path[sample.path],
                        ROLE_PROFILES["balanced"],
                        {},
                    ),
                    record_by_path[sample.path]["role"] == "primary",
                ),
                reverse=True,
            )
            chosen = ranked[:max_per_example_slot]
            if slot == "HANDS" and max_per_example_slot >= 3:
                context = next(
                    (
                        sample
                        for sample in ranked
                        if record_by_path[sample.path]["name"]
                        == "hands_arm_cover_context"
                    ),
                    None,
                )
                if context is not None and context not in chosen:
                    chosen[-1] = context
            selected[slot].extend(chosen)
    return selected


def group_training_samples(
    samples: dict[str, list[Any]],
    records: dict[str, list[dict[str, Any]]],
    group_ids: dict[str, str] | None = None,
    group_weights: Mapping[str, float] | None = None,
) -> dict[str, list[GroupedTrainingSample]]:
    example_by_path = {
        str(record["path"]): example_id
        for example_id, rows in records.items()
        for record in rows
    }
    grouped_output: dict[str, list[GroupedTrainingSample]] = {
        slot: [] for slot in SLOTS
    }
    for slot in SLOTS:
        grouped: dict[str, list[Any]] = defaultdict(list)
        for sample in samples[slot]:
            example_id = example_by_path.get(sample.path)
            if example_id is not None:
                group_id = (
                    group_ids.get(example_id, example_id)
                    if group_ids is not None
                    else example_id
                )
                grouped[group_id].append(sample)
        for group_id, rows in grouped.items():
            positive_sets = {sample.positives for sample in rows}
            if len(positive_sets) != 1:
                raise ValueError("Grouped crop positives are inconsistent")
            grouped_output[slot].append(
                GroupedTrainingSample(
                    paths=tuple(sample.path for sample in rows),
                    positives=rows[0].positives,
                    weight=(
                        float(np.mean([sample.weight for sample in rows]))
                        * (
                            float(group_weights.get(group_id, 1.0))
                            if group_weights is not None
                            else 1.0
                        )
                    ),
                )
            )
    return grouped_output


def split_records_for_examples(
    split_examples: dict[str, list[dict[str, Any]]],
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        split: {
            example["example_id"]: records.get(example["example_id"], [])
            for example in examples
        }
        for split, examples in split_examples.items()
    }


def train_epoch(
    *,
    model: torch.nn.Module,
    pooler: SlotAttentionPooler | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    samples: dict[str, Sequence[GroupedTrainingSample]],
    candidates: dict[str, torch.Tensor],
    alternate_paths: dict[str, str],
    alternate_probability: float,
    batch_size: int,
    temperature: float,
    hard_negative_weight: float,
    hard_negative_k: int,
    hard_negative_margin: float,
    device: torch.device,
    augment: bool,
    preprocess: str,
) -> float:
    model.train()
    losses = []
    for slot in SLOTS:
        order = torch.randperm(len(samples[slot])).tolist()
        for start in range(0, len(order), batch_size):
            batch = [
                samples[slot][index]
                for index in order[start : start + batch_size]
            ]
            paths = []
            group_ranges = []
            for sample in batch:
                group_start = len(paths)
                paths.extend(
                    choose_training_path(
                        path,
                        alternate_paths,
                        alternate_probability,
                    )
                    for path in sample.paths
                )
                group_ranges.append((group_start, len(paths)))
            pixels = load_pixel_batch(
                paths,
                augment=augment,
                preprocess=preprocess,
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                tokens = model(pixel_values=pixels).last_hidden_state
                queries = (
                    pooler(tokens, slot)
                    if pooler is not None
                    else torch.nn.functional.normalize(tokens[:, 0], dim=-1)
                )
                crop_logits = queries @ candidates[slot].T / temperature
                logits = torch.stack(
                    [
                        torch.logsumexp(
                            crop_logits[group_start:group_end],
                            dim=0,
                        )
                        - np.log(max(1, group_end - group_start))
                        for group_start, group_end in group_ranges
                    ]
                )
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
                contrastive = (
                    (denominator - numerator) * weights
                ).sum() / weights.sum()
                loss = contrastive
                if hard_negative_weight > 0:
                    hard_losses = []
                    for index, sample in enumerate(batch):
                        negative_logits = logits[index].clone()
                        negative_logits[list(sample.positives)] = -torch.inf
                        hard_count = min(
                            hard_negative_k,
                            negative_logits.numel() - len(sample.positives),
                        )
                        hard = torch.topk(
                            negative_logits,
                            k=max(1, hard_count),
                        ).values
                        hard_score = torch.logsumexp(hard, dim=0) - np.log(
                            max(1, hard_count)
                        )
                        positive_score = numerator[index]
                        hard_losses.append(
                            torch.nn.functional.softplus(
                                hard_score
                                - positive_score
                                + hard_negative_margin
                            )
                        )
                    hard_tensor = torch.stack(hard_losses)
                    hard_loss = (hard_tensor * weights).sum() / weights.sum()
                    loss = contrastive + hard_negative_weight * hard_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    *[
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    *([] if pooler is None else list(pooler.parameters())),
                ],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def choose_training_path(
    path: str,
    alternate_paths: dict[str, str],
    alternate_probability: float,
) -> str:
    alternate = alternate_paths.get(path)
    if alternate and random.random() < alternate_probability:
        return alternate
    return path


@torch.inference_mode()
def encode_records(
    model: torch.nn.Module,
    pooler: SlotAttentionPooler | None,
    records: dict[str, list[dict[str, Any]]],
    batch_size: int,
    device: torch.device,
    *,
    preprocess: str = "color",
) -> dict[str, np.ndarray]:
    model.eval()
    if pooler is not None:
        pooler.eval()
    embeddings = {}
    path_slots = {
        str(record["path"]): record["slot"]
        for rows in records.values()
        for record in rows
    }
    for slot in SLOTS:
        paths = sorted(
            path
            for path, path_slot in path_slots.items()
            if path_slot == slot
        )
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            pixels = load_pixel_batch(
                batch_paths,
                augment=False,
                preprocess=preprocess,
            ).to(device)
            tokens = model(pixel_values=pixels).last_hidden_state
            outputs = (
                pooler(tokens, slot)
                if pooler is not None
                else torch.nn.functional.normalize(tokens[:, 0], dim=-1)
            )
            embeddings.update(
                zip(
                    batch_paths,
                    outputs.float().cpu().numpy(),
                    strict=True,
                )
            )
    return embeddings


def load_pixel_batch(
    paths: Sequence[str],
    *,
    augment: bool,
    preprocess: str = "color",
) -> torch.Tensor:
    arrays = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        if augment:
            image = augment_query(image)
        if preprocess == "gray":
            grayscale = ImageOps.grayscale(image).convert("RGB")
            image.close()
            image = grayscale
        elif preprocess != "color":
            image.close()
            raise ValueError(f"unsupported query preprocess: {preprocess}")
        resized = image.resize((224, 224), Image.Resampling.BICUBIC)
        image.close()
        array = np.asarray(resized, dtype=np.float32) / 255.0
        resized.close()
        arrays.append(
            np.transpose(
                (array - DINO_MEAN) / DINO_STD,
                (2, 0, 1),
            )
        )
    return torch.from_numpy(np.stack(arrays))


def augment_query(image: Image.Image) -> Image.Image:
    output = image
    if random.random() < 0.5:
        mirrored = ImageOps.mirror(output)
        output.close()
        output = mirrored
    if random.random() < 0.7:
        color = ImageEnhance.Color(output).enhance(random.uniform(0.75, 1.15))
        output.close()
        output = color
        contrast = ImageEnhance.Contrast(output).enhance(random.uniform(0.85, 1.15))
        output.close()
        output = contrast
    return output


if __name__ == "__main__":
    raise SystemExit(main())
