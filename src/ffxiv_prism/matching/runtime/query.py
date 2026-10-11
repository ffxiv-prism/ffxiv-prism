"""Shared inference primitives; training entry points live in scripts."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageEnhance, ImageOps
from .encoders import DINO_MEAN, DINO_STD
from .session import timed
SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
HANDS_RIGHT_CROP_NAMES = frozenset({"hands_right", "hands_forearm_right"})

def normalized_model_id(model_id: str) -> str:
    # Local model directories are mapped back to their original identifiers by load_pretrained.
    path = Path(model_id)
    if path.exists():
        return str(path.resolve()).replace("\\", "/").casefold().rstrip("/")
    return model_id.replace("\\", "/").casefold().rstrip("/")


class SlotAttentionPooler(nn.Module):
    """Learn a small slot-specific residual from local DINO patch tokens."""

    def __init__(
        self,
        dimension: int,
        slots: Sequence[str] = SLOTS,
        num_register_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.num_register_tokens = num_register_tokens
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
        patches = tokens[:, 1 + self.num_register_tokens :]
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


def hands_pose_mirror_paths(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    alternate_paths: Mapping[str, str] | None = None,
) -> set[str]:
    primary = {
        str(record["path"])
        for rows in records.values()
        for record in rows
        if record.get("slot") == "HANDS"
        and record.get("name") in HANDS_RIGHT_CROP_NAMES
    }
    alternates = {
        str((alternate_paths or {})[path])
        for path in primary
        if path in (alternate_paths or {})
    }
    return primary | alternates


def load_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    pooler: nn.Module | None,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=True)
    checkpoint_model_id = payload.get("model_id")
    active_model_id = getattr(model.config, "_name_or_path", None)
    if (
        checkpoint_model_id
        and active_model_id
        and normalized_model_id(str(checkpoint_model_id))
        != normalized_model_id(str(active_model_id))
    ):
        raise ValueError(
            f"Checkpoint model_id {checkpoint_model_id!r} does not match "
            f"active model_id {active_model_id!r}"
        )
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


@timed("query_checkpoint_load")
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
        pooler = SlotAttentionPooler(
            int(model.config.hidden_size),
            num_register_tokens=int(
                getattr(model.config, "num_register_tokens", 0)
            ),
        ).to(device)
    else:
        pooler = None
    load_training_checkpoint(path, model, pooler, device)
    return pooler


@torch.inference_mode()
@timed("query_encode.adapted")
def encode_records(
    model: torch.nn.Module,
    pooler: SlotAttentionPooler | None,
    records: dict[str, list[dict[str, Any]]],
    batch_size: int,
    device: torch.device,
    *,
    pose_normalize_hands: bool = False,
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
    mirror_paths = (
        hands_pose_mirror_paths(records)
        if pose_normalize_hands
        else set()
    )
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
                mirror_paths=mirror_paths,
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
    mirror_paths: set[str] | None = None,
    random_mirror: bool = True,
    preprocess: str = "color",
) -> torch.Tensor:
    arrays = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        if str(path) in (mirror_paths or set()):
            mirrored = ImageOps.mirror(image)
            image.close()
            image = mirrored
        if augment:
            image = augment_query(image, allow_mirror=random_mirror)
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


def augment_query(
    image: Image.Image,
    *,
    allow_mirror: bool = True,
) -> Image.Image:
    output = image
    if allow_mirror and random.random() < 0.5:
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
