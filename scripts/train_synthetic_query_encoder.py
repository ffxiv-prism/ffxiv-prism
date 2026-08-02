from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import CACHE_DIR, PROCESSED_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.evaluation_split import load_split_assignments
from scripts.evaluate_glamour_embedding_model import DINO_MEAN, DINO_STD
from scripts.evaluate_glamour_matching import SLOTS
from scripts.train_glamour_adapter import build_candidate_state
from scripts.train_glamour_query_encoder import (
    SlotAttentionPooler,
    load_slot_eligibility,
    load_query_head_from_checkpoint,
    set_seed,
    unfreeze_last_blocks,
)

DEFAULT_INDEX = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_CHECKPOINT = Path("data/models/synthetic_query_encoder.pt")
DEFAULT_OUTPUT = PROCESSED_DIR / "synthetic_query_encoder_eval.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the query encoder on screenshot-like augmentations of "
            "strict fold-train canonical equipment renders. All race/gender "
            "renders of one allowed piece are positives, discouraging "
            "mannequin identity features without leaking validation pieces."
        )
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument(
        "--slot-eligibility-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--slots", nargs="+", choices=SLOTS, default=list(SLOTS)
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-validation-images", type=int, default=1200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the strict fold piece scope and report synthetic sample "
            "counts without loading or updating the model."
        ),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_torch_device(args.device)
    args.device = str(device)
    all_metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    selected_slots = set(args.slots)
    selected_indices = [
        index
        for index, row in enumerate(all_metadata)
        if str(row.get("slot")) in selected_slots
    ]
    metadata = [all_metadata[index] for index in selected_indices]
    all_candidate_embeddings = np.load(
        args.index_dir / "embeddings.npy",
        mmap_mode="r",
    )
    candidate_embeddings = np.asarray(
        all_candidate_embeddings[selected_indices], dtype=np.float32
    )
    candidate_state = build_candidate_state(
        metadata,
        candidate_embeddings,
        device,
    )
    examples = read_jsonl(args.examples)
    eligible_slots = load_slot_eligibility(args.slot_eligibility_manifest)
    assignments = load_split_assignments(args.fold_manifest)
    allowed_piece_ids = collect_piece_ids_for_role(
        examples,
        eligible_slots,
        assignments,
        selected_slots,
        role="train",
    )
    validation_piece_ids = collect_piece_ids_for_role(
        examples,
        eligible_slots,
        assignments,
        selected_slots,
        role="validation",
    )
    validation_overlap = {
        slot: sorted(allowed_piece_ids[slot] & validation_piece_ids[slot])
        for slot in selected_slots
    }
    if any(validation_overlap.values()):
        raise ValueError(
            "Synthetic query scope leaks validation piece IDs: "
            f"{validation_overlap}"
        )
    samples = build_catalog_samples(
        metadata,
        candidate_state["piece_to_local"],
        allowed_piece_ids=allowed_piece_ids,
        forbidden_piece_ids=validation_piece_ids,
    )
    empty_slots = [slot for slot in selected_slots if not samples[slot]]
    if empty_slots:
        raise ValueError(
            f"No fold-train canonical samples for requested slots: {empty_slots}"
        )
    train_samples, validation_samples = split_catalog_samples(
        samples,
        validation_percent=args.validation_percent,
    )
    train_samples = limit_samples(
        train_samples,
        args.max_train_images,
        args.seed,
    )
    validation_samples = limit_samples(
        validation_samples,
        args.max_validation_images,
        args.seed + 1,
    )
    if args.dry_run:
        payload = {
            "version": 1,
            "status": "ready",
            "training_started": False,
            "slots": list(args.slots),
            "allowed_piece_counts": {
                slot: len(allowed_piece_ids[slot])
                for slot in sorted(selected_slots)
            },
            "validation_piece_overlap_counts": {
                slot: len(validation_overlap[slot])
                for slot in sorted(selected_slots)
            },
            "train_image_count": sum(map(len, train_samples.values())),
            "validation_image_count": sum(
                map(len, validation_samples.values())
            ),
            "candidate_count": len(metadata),
            "sources": {
                "index_dir": portable_path(args.index_dir),
                "examples": portable_path(args.examples),
                "slot_eligibility_manifest": portable_path(
                    args.slot_eligibility_manifest
                ),
                "fold_manifest": portable_path(args.fold_manifest),
            },
        }
        write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    trainable_names = unfreeze_last_blocks(model, args.unfreeze_blocks)
    pooler = (
        load_query_head_from_checkpoint(
            args.resume_checkpoint,
            model,
            device,
        )
        if args.resume_checkpoint is not None
        else None
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if pooler is not None:
        trainable_parameters.extend(pooler.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
    )

    best_state = None
    best_pooler_state = None
    best_epoch = 0
    best_score = (-1.0, -1.0)
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model=model,
            pooler=pooler,
            optimizer=optimizer,
            scaler=scaler,
            samples=train_samples,
            candidates=candidate_state["tensors"],
            batch_size=args.batch_size,
            temperature=args.temperature,
            device=device,
        )
        validation = evaluate_augmented_catalog(
            model=model,
            pooler=pooler,
            samples=validation_samples,
            candidates=candidate_state["tensors"],
            batch_size=args.batch_size,
            device=device,
        )
        score = (
            validation["piece_recall_at_1"],
            validation["piece_recall_at_10"],
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
                if name in trainable_names
            }
            best_pooler_state = (
                copy.deepcopy(pooler.state_dict()) if pooler is not None else None
            )
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
            f"catalog_piece@1={validation['piece_recall_at_1']:.3f} "
            f"catalog_piece@10={validation['piece_recall_at_10']:.3f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Synthetic query training produced no checkpoint")
    current = model.state_dict()
    current.update(best_state)
    model.load_state_dict(current)
    if pooler is not None and best_pooler_state is not None:
        pooler.load_state_dict(best_pooler_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "model_id": args.model_id,
            "unfreeze_blocks": args.unfreeze_blocks,
            "trainable_state_dict": best_state,
            "trainable_parameter_names": sorted(trainable_names),
            "slot_attention": isinstance(pooler, SlotAttentionPooler),
            "adapter_rank": int(getattr(pooler, "rank", 0)),
            "pooler_state_dict": (
                pooler.state_dict() if pooler is not None else None
            ),
            "best_epoch": best_epoch,
            "seed": args.seed,
            "training_source": "augmented_canonical_equipment_renders",
            "training_source_scope": "fold_train_only",
            "source_fold_manifest": portable_path(args.fold_manifest),
            "source_slot_eligibility_manifest": portable_path(
                args.slot_eligibility_manifest
            ),
        },
        args.checkpoint,
    )
    write_json(
        args.output,
        {
            "model": {
                "checkpoint": portable_path(args.checkpoint),
                "model_id": args.model_id,
                "best_epoch": best_epoch,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "pooler_parameter_count": (
                    sum(parameter.numel() for parameter in pooler.parameters())
                    if pooler is not None
                    else 0
                ),
            },
            "training": {
                "seed": args.seed,
                "train_image_count": sum(map(len, train_samples.values())),
                "validation_image_count": sum(
                    map(len, validation_samples.values())
                ),
                "race_invariant_positive_policy": "all_piece_renders",
                "query_piece_scope": "fold_train_only",
                "allowed_piece_counts": {
                    slot: len(allowed_piece_ids[slot])
                    for slot in sorted(selected_slots)
                },
                "validation_piece_overlap_counts": {
                    slot: len(validation_overlap[slot])
                    for slot in sorted(selected_slots)
                },
                "slots": list(args.slots),
                "history": history,
            },
            "index_dir": portable_path(args.index_dir),
            "examples": portable_path(args.examples),
            "slot_eligibility_manifest": portable_path(
                args.slot_eligibility_manifest
            ),
            "fold_manifest": portable_path(args.fold_manifest),
            "resume_checkpoint": portable_path(args.resume_checkpoint),
        },
    )
    print(f"checkpoint={args.checkpoint} output={args.output}")
    return 0


def build_catalog_samples(
    metadata: Sequence[dict[str, Any]],
    piece_to_local: dict[str, dict[int, tuple[int, ...]]],
    *,
    allowed_piece_ids: dict[str, set[int]] | None = None,
    forbidden_piece_ids: dict[str, set[int]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    local_indices = defaultdict(int)
    output = {slot: [] for slot in SLOTS}
    for row in metadata:
        slot = str(row.get("slot", ""))
        if slot not in output:
            continue
        local_index = local_indices[slot]
        local_indices[slot] += 1
        row_piece_ids = {
            int(piece["piece_id"])
            for piece in row.get("pieces", [])
        }
        allowed_match = (
            row_piece_ids.intersection(allowed_piece_ids.get(slot, set()))
            if allowed_piece_ids is not None
            else row_piece_ids
        )
        if not allowed_match:
            continue
        forbidden_match = row_piece_ids.intersection(
            (forbidden_piece_ids or {}).get(slot, set())
        )
        if forbidden_match:
            raise ValueError(
                "Fold-train canonical row also contains validation pieces: "
                f"slot={slot} piece_ids={sorted(forbidden_match)}"
            )
        positives = sorted(
            {
                candidate_index
                for piece in row.get("pieces", [])
                for candidate_index in piece_to_local[slot].get(
                    int(piece["piece_id"]),
                    (),
                )
            }
        )
        path = Path(str(row["local_path"]))
        if not positives or not path.is_file():
            continue
        output[slot].append(
            {
                "path": str(path),
                "candidate_crop_box_normalized": row.get(
                    "candidate_crop_box_normalized"
                ),
                "local_index": local_index,
                "positives": tuple(positives),
            }
        )
    return output


def collect_piece_ids_for_role(
    examples: Sequence[dict[str, Any]],
    eligible_slots: dict[str, set[str]],
    assignments: dict[str, str],
    selected_slots: set[str],
    *,
    role: str,
) -> dict[str, set[int]]:
    output = {slot: set() for slot in selected_slots}
    for example in examples:
        if assignments.get(str(example.get("glamour_id"))) != role:
            continue
        example_id = str(example.get("example_id"))
        labels = example.get("core_slot_labels") or {}
        for slot in selected_slots.intersection(
            eligible_slots.get(example_id, set())
        ):
            output[slot].update(
                int(piece_id)
                for piece_id in (labels.get(slot) or {}).get("piece_ids", [])
            )
    return output


def split_catalog_samples(
    samples: dict[str, list[dict[str, Any]]],
    *,
    validation_percent: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    train = {slot: [] for slot in SLOTS}
    validation = {slot: [] for slot in SLOTS}
    for slot, rows in samples.items():
        for row in rows:
            digest = hashlib.sha1(row["path"].encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % 100
            target = validation if bucket < validation_percent else train
            target[slot].append(row)
    return train, validation


def limit_samples(
    samples: dict[str, list[dict[str, Any]]],
    limit: int | None,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if limit is None or sum(map(len, samples.values())) <= limit:
        return samples
    rng = random.Random(seed)
    combined = [
        (slot, row)
        for slot, rows in samples.items()
        for row in rows
    ]
    rng.shuffle(combined)
    output = {slot: [] for slot in SLOTS}
    for slot, row in combined[:limit]:
        output[slot].append(row)
    return output


def train_epoch(
    *,
    model: torch.nn.Module,
    pooler: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    samples: dict[str, list[dict[str, Any]]],
    candidates: dict[str, torch.Tensor],
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> float:
    model.train()
    if pooler is not None:
        pooler.train()
    losses = []
    for slot in SLOTS:
        order = torch.randperm(len(samples[slot])).tolist()
        for start in range(0, len(order), batch_size):
            batch = [
                samples[slot][index]
                for index in order[start : start + batch_size]
            ]
            pixels = load_synthetic_batch(
                batch,
                augment=True,
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
                logits = queries @ candidates[slot].T / temperature
                denominator = torch.logsumexp(logits, dim=1)
                numerator = torch.stack(
                    [
                        torch.logsumexp(
                            logits[index, list(row["positives"])],
                            dim=0,
                        )
                        for index, row in enumerate(batch)
                    ]
                )
                loss = (denominator - numerator).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


@torch.inference_mode()
def evaluate_augmented_catalog(
    *,
    model: torch.nn.Module,
    pooler: torch.nn.Module | None,
    samples: dict[str, list[dict[str, Any]]],
    candidates: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    if pooler is not None:
        pooler.eval()
    hits_at_1 = 0
    hits_at_10 = 0
    count = 0
    by_slot = {}
    for slot in SLOTS:
        slot_hit_1 = 0
        slot_hit_10 = 0
        rows = samples[slot]
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            pixels = load_synthetic_batch(
                batch,
                augment=True,
            ).to(device)
            tokens = model(pixel_values=pixels).last_hidden_state
            queries = (
                pooler(tokens, slot)
                if pooler is not None
                else torch.nn.functional.normalize(tokens[:, 0], dim=-1)
            )
            top = torch.topk(
                queries @ candidates[slot].T,
                k=min(10, len(candidates[slot])),
                dim=1,
            ).indices.cpu().tolist()
            for row, predicted in zip(batch, top, strict=True):
                positives = set(row["positives"])
                hit_1 = bool(predicted and predicted[0] in positives)
                hit_10 = bool(positives.intersection(predicted))
                slot_hit_1 += int(hit_1)
                slot_hit_10 += int(hit_10)
        slot_count = len(rows)
        hits_at_1 += slot_hit_1
        hits_at_10 += slot_hit_10
        count += slot_count
        by_slot[slot] = {
            "count": slot_count,
            "piece_recall_at_1": slot_hit_1 / slot_count if slot_count else 0.0,
            "piece_recall_at_10": slot_hit_10 / slot_count if slot_count else 0.0,
        }
    return {
        "count": count,
        "piece_recall_at_1": hits_at_1 / count if count else 0.0,
        "piece_recall_at_10": hits_at_10 / count if count else 0.0,
        "by_slot": by_slot,
    }


def load_synthetic_batch(
    samples: Sequence[dict[str, Any]],
    *,
    augment: bool,
) -> torch.Tensor:
    arrays = []
    for sample in samples:
        with Image.open(str(sample["path"])) as source:
            image = source.convert("RGB")
        normalized = sample.get("candidate_crop_box_normalized")
        if normalized:
            left, top, right, bottom = (float(value) for value in normalized)
            box = (
                max(0, min(image.width - 1, round(left * image.width))),
                max(0, min(image.height - 1, round(top * image.height))),
                max(1, min(image.width, round(right * image.width))),
                max(1, min(image.height, round(bottom * image.height))),
            )
            if box[2] > box[0] and box[3] > box[1]:
                cropped = image.crop(box)
                image.close()
                image = cropped
        output = augment_catalog_render(image) if augment else image
        if output is not image:
            image.close()
        resized = output.resize((224, 224), Image.Resampling.BICUBIC)
        output.close()
        array = np.asarray(resized, dtype=np.float32) / 255.0
        resized.close()
        arrays.append(
            np.transpose(
                (array - DINO_MEAN) / DINO_STD,
                (2, 0, 1),
            )
        )
    return torch.from_numpy(np.stack(arrays))


def augment_catalog_render(image: Image.Image) -> Image.Image:
    output = image.copy()
    if random.random() < 0.5:
        mirrored = ImageOps.mirror(output)
        output.close()
        output = mirrored
    for enhancer, bounds, probability in (
        (ImageEnhance.Color, (0.45, 1.35), 0.85),
        (ImageEnhance.Contrast, (0.70, 1.30), 0.80),
        (ImageEnhance.Brightness, (0.72, 1.25), 0.75),
        (ImageEnhance.Sharpness, (0.55, 1.45), 0.45),
    ):
        if random.random() < probability:
            enhanced = enhancer(output).enhance(random.uniform(*bounds))
            output.close()
            output = enhanced
    if random.random() < 0.25:
        blurred = output.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.25, 1.15))
        )
        output.close()
        output = blurred

    square = output.resize((224, 224), Image.Resampling.BICUBIC)
    output.close()
    scale = random.uniform(0.72, 1.08)
    side = max(32, int(round(224 * scale)))
    scaled = square.resize((side, side), Image.Resampling.BICUBIC)
    square.close()
    background = Image.new(
        "RGB",
        (224, 224),
        tuple(random.randint(18, 238) for _ in range(3)),
    )
    x = random.randint(min(0, 224 - side), max(0, 224 - side))
    y = random.randint(min(0, 224 - side), max(0, 224 - side))
    background.paste(scaled, (x, y))
    scaled.close()

    if random.random() < 0.30:
        draw = ImageDraw.Draw(background)
        width = random.randint(8, 40)
        height = random.randint(8, 34)
        left = random.randint(0, 224 - width)
        top = random.randint(0, 224 - height)
        draw.rectangle(
            (left, top, left + width, top + height),
            fill=tuple(random.randint(0, 255) for _ in range(3)),
        )
    return background


if __name__ == "__main__":
    raise SystemExit(main())
