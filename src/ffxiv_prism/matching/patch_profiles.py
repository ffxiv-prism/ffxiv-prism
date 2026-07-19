from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps


ImageFile.LOAD_TRUNCATED_IMAGES = True
PATCH_PROFILE_VERSION = 1


@dataclass(frozen=True)
class PatchProfileIndex:
    profiles: np.ndarray
    projection: np.ndarray
    metadata_indices: np.ndarray
    piece_ids_by_profile: tuple[tuple[int, ...], ...]
    model_genders: tuple[str, ...]
    model_id: str
    preprocess: str
    candidate_crop: str
    grid_size: int


def dino_patch_grid(
    tokens: torch.Tensor,
    *,
    num_register_tokens: int = 0,
) -> torch.Tensor:
    """Remove CLS/register tokens and reshape DINO patches to a square grid."""

    if tokens.ndim != 3:
        raise ValueError("DINO tokens must have shape [batch, tokens, dim]")
    if num_register_tokens < 0:
        raise ValueError("num_register_tokens cannot be negative")
    patches = tokens[:, 1 + num_register_tokens :]
    patch_count = int(patches.shape[1])
    grid_size = int(round(patch_count**0.5))
    if grid_size * grid_size != patch_count:
        raise ValueError(
            "DINO patch count is not square after CLS/register removal: "
            f"{patch_count}"
        )
    return patches.reshape(tokens.shape[0], grid_size, grid_size, tokens.shape[2])


def pool_patch_grid(
    patch_grid: torch.Tensor,
    *,
    output_grid_size: int,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create a compact normalized 2-D set of local descriptors."""

    if patch_grid.ndim != 4:
        raise ValueError("patch_grid must have shape [batch, height, width, dim]")
    if output_grid_size < 1:
        raise ValueError("output_grid_size must be positive")
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        patch_grid.permute(0, 3, 1, 2),
        (output_grid_size, output_grid_size),
    ).permute(0, 2, 3, 1)
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    if projection is not None:
        pooled = torch.nn.functional.normalize(pooled @ projection, dim=-1)
    return pooled.flatten(1, 2)


@torch.inference_mode()
def encode_dino_patch_profiles(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    *,
    projection: np.ndarray,
    grid_size: int,
    batch_size: int,
    device: torch.device,
    preprocess: str = "color",
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    projection_tensor = torch.as_tensor(
        projection,
        dtype=torch.float32,
        device=device,
    )
    register_count = int(getattr(model.config, "num_register_tokens", 0))
    output: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        pixels = _load_batch(batch_rows, preprocess=preprocess).to(device)
        device_type = getattr(device, "type", str(device).split(":")[0])
        with torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=device_type == "cuda",
        ):
            tokens = model(pixel_values=pixels).last_hidden_state
            patches = dino_patch_grid(
                tokens,
                num_register_tokens=register_count,
            )
            profiles = pool_patch_grid(
                patches,
                output_grid_size=grid_size,
                projection=projection_tensor,
            )
        output.append(profiles.float().cpu().numpy().astype(np.float16))
    if not output:
        return np.empty(
            (0, grid_size * grid_size, projection.shape[1]),
            dtype=np.float16,
        )
    return np.concatenate(output, axis=0)


@torch.inference_mode()
def symmetric_chamfer_score_matrix(
    query_profiles: np.ndarray,
    candidate_profiles: np.ndarray,
    *,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Score unordered local descriptor sets with symmetric Chamfer cosine."""

    queries = np.asarray(query_profiles)
    candidates = np.asarray(candidate_profiles)
    if queries.ndim != 3 or candidates.ndim != 3:
        raise ValueError("profiles must have shapes [n, patches, dim]")
    if queries.shape[2] != candidates.shape[2]:
        raise ValueError("query and candidate dimensions must match")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    torch_device = torch.device(device)
    candidate_tensor = torch.as_tensor(
        candidates,
        dtype=torch.float32,
        device=torch_device,
    )
    output = []
    for start in range(0, len(queries), batch_size):
        query_tensor = torch.as_tensor(
            queries[start : start + batch_size],
            dtype=torch.float32,
            device=torch_device,
        )
        similarities = torch.einsum(
            "bqd,ncd->bnqc",
            query_tensor,
            candidate_tensor,
        )
        query_to_candidate = similarities.max(dim=3).values.mean(dim=2)
        candidate_to_query = similarities.max(dim=2).values.mean(dim=2)
        scores = (query_to_candidate + candidate_to_query) * 0.5
        output.append(scores.cpu().numpy().astype(np.float32, copy=False))
    if not output:
        return np.empty((0, len(candidates)), dtype=np.float32)
    return np.concatenate(output, axis=0)


def save_patch_profile_index(path: Path, index: PatchProfileIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray(PATCH_PROFILE_VERSION),
        profiles=index.profiles,
        projection=index.projection.astype(np.float16),
        metadata_indices=index.metadata_indices,
        piece_ids=np.asarray(
            [
                ",".join(str(value) for value in values)
                for values in index.piece_ids_by_profile
            ]
        ),
        model_genders=np.asarray(index.model_genders),
        model_id=np.asarray(index.model_id),
        preprocess=np.asarray(index.preprocess),
        candidate_crop=np.asarray(index.candidate_crop),
        grid_size=np.asarray(index.grid_size),
    )


def load_patch_profile_index(path: Path) -> PatchProfileIndex:
    with np.load(path, allow_pickle=False) as payload:
        version = int(payload["version"].tolist())
        if version != PATCH_PROFILE_VERSION:
            raise ValueError(f"Unsupported patch profile version: {version}")
        return PatchProfileIndex(
            profiles=np.asarray(payload["profiles"], dtype=np.float16),
            projection=np.asarray(payload["projection"], dtype=np.float32),
            metadata_indices=np.asarray(payload["metadata_indices"], dtype=np.int64),
            piece_ids_by_profile=tuple(
                tuple(int(value) for value in str(text).split(",") if value)
                for text in payload["piece_ids"].tolist()
            ),
            model_genders=tuple(
                str(value) for value in payload["model_genders"].tolist()
            ),
            model_id=str(payload["model_id"].tolist()),
            preprocess=str(payload["preprocess"].tolist()),
            candidate_crop=str(payload["candidate_crop"].tolist()),
            grid_size=int(payload["grid_size"].tolist()),
        )


def _load_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    preprocess: str,
) -> torch.Tensor:
    if preprocess not in {"color", "gray"}:
        raise ValueError(f"Unsupported patch preprocess: {preprocess}")
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    arrays = []
    for row in rows:
        with Image.open(Path(str(row["path"]))) as source:
            image = source.convert("RGB")
        box = row.get("crop_box_normalized")
        if box is not None:
            left, top, right, bottom = (float(value) for value in box)
            cropped = image.crop(
                (
                    int(round(left * image.width)),
                    int(round(top * image.height)),
                    int(round(right * image.width)),
                    int(round(bottom * image.height)),
                )
            )
            image.close()
            image = cropped
        if preprocess == "gray":
            gray = ImageOps.grayscale(image).convert("RGB")
            image.close()
            image = gray
        resized = image.resize((224, 224), Image.Resampling.BICUBIC)
        image.close()
        array = np.asarray(resized, dtype=np.float32) / 255.0
        resized.close()
        arrays.append(np.transpose((array - mean) / std, (2, 0, 1)))
    return torch.from_numpy(np.stack(arrays))
