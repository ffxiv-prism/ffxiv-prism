from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps


ImageFile.LOAD_TRUNCATED_IMAGES = True


SPATIAL_PROFILE_VERSION = 1
DEFAULT_PROJECTION_DIM = 256
DEFAULT_PROJECTION_SEED = 0


@dataclass(frozen=True)
class SpatialProfileIndex:
    profiles: np.ndarray
    projection: np.ndarray
    metadata_indices: np.ndarray
    image_row_ids: np.ndarray
    piece_ids_by_profile: tuple[tuple[int, ...], ...]
    model_id: str
    preprocess: str
    candidate_crop: str


def encode_dino_row_profiles(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    *,
    projection: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode a compact, vertically ordered DINO patch-token profile.

    Global CLS embeddings are deliberately avoided here: averaging each patch
    row retains where garment transitions occur (for example thigh-high versus
    calf-high socks) while horizontal averaging makes the signal tolerant of
    stance and left/right asymmetry.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    projection_tensor = torch.as_tensor(
        projection,
        dtype=torch.float32,
        device=device,
    )
    output: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        pixels = _load_grayscale_batch(batch_rows).to(device)
        with torch.inference_mode():
            tokens = model(pixel_values=pixels).last_hidden_state[:, 1:]
            patch_count = int(tokens.shape[1])
            grid_size = int(round(patch_count**0.5))
            if grid_size * grid_size != patch_count:
                raise ValueError(
                    f"DINO patch count is not square: {patch_count}"
                )
            rows_tensor = tokens.reshape(
                len(batch_rows), grid_size, grid_size, -1
            ).mean(dim=2)
            rows_tensor = torch.nn.functional.normalize(rows_tensor, dim=-1)
            rows_tensor = torch.nn.functional.normalize(
                rows_tensor @ projection_tensor,
                dim=-1,
            )
        output.append(rows_tensor.cpu().numpy().astype(np.float16))
    if not output:
        return np.empty((0, 0, projection.shape[1]), dtype=np.float16)
    return np.concatenate(output, axis=0)


def make_projection(
    input_dimension: int,
    output_dimension: int = DEFAULT_PROJECTION_DIM,
    seed: int = DEFAULT_PROJECTION_SEED,
) -> np.ndarray:
    if input_dimension < 1 or output_dimension < 1:
        raise ValueError("projection dimensions must be positive")
    generator = np.random.default_rng(seed)
    projection = generator.standard_normal(
        (input_dimension, output_dimension),
        dtype=np.float32,
    )
    projection /= np.sqrt(float(output_dimension))
    return projection.astype(np.float32, copy=False)


def spatial_profile_scores(
    query_profile: np.ndarray,
    candidate_profiles: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query_profile, dtype=np.float32)
    candidates = np.asarray(candidate_profiles, dtype=np.float32)
    if query.ndim != 2 or candidates.ndim != 3:
        raise ValueError("profiles must have shapes [rows, dim] and [n, rows, dim]")
    if candidates.shape[1:] != query.shape:
        raise ValueError(
            f"profile shape mismatch: query={query.shape} candidates={candidates.shape}"
        )
    return np.einsum("rd,nrd->n", query, candidates) / query.shape[0]


def save_spatial_profile_index(
    path: Path,
    index: SpatialProfileIndex,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray(SPATIAL_PROFILE_VERSION),
        profiles=index.profiles,
        projection=index.projection.astype(np.float16),
        metadata_indices=index.metadata_indices,
        image_row_ids=index.image_row_ids,
        piece_ids=np.asarray(
            [",".join(str(value) for value in values) for values in index.piece_ids_by_profile]
        ),
        model_id=np.asarray(index.model_id),
        preprocess=np.asarray(index.preprocess),
        candidate_crop=np.asarray(index.candidate_crop),
    )


def load_spatial_profile_index(path: Path) -> SpatialProfileIndex:
    with np.load(path, allow_pickle=False) as payload:
        version = int(payload["version"].tolist())
        if version != SPATIAL_PROFILE_VERSION:
            raise ValueError(f"Unsupported spatial profile version: {version}")
        piece_ids = tuple(
            tuple(int(value) for value in str(text).split(",") if value)
            for text in payload["piece_ids"].tolist()
        )
        return SpatialProfileIndex(
            profiles=np.asarray(payload["profiles"], dtype=np.float16),
            projection=np.asarray(payload["projection"], dtype=np.float32),
            metadata_indices=np.asarray(payload["metadata_indices"], dtype=np.int64),
            image_row_ids=np.asarray(payload["image_row_ids"], dtype=np.int64),
            piece_ids_by_profile=piece_ids,
            model_id=str(payload["model_id"].tolist()),
            preprocess=str(payload["preprocess"].tolist()),
            candidate_crop=str(payload["candidate_crop"].tolist()),
        )


def rerank_candidates_with_spatial_profile(
    candidates: Sequence[Mapping[str, Any]],
    index: SpatialProfileIndex,
    query_profile: np.ndarray,
    *,
    rerank_k: int,
    blend: float,
) -> list[dict[str, Any]]:
    """Blend retrieval and spatial percentiles within a fixed leading window."""

    if rerank_k < 1:
        raise ValueError("rerank_k must be positive")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between zero and one")
    output = [dict(row) for row in candidates]
    leading = output[: min(rerank_k, len(output))]
    if len(leading) < 2:
        return output
    target_ids = {int(row["piece_id"]) for row in leading}
    profile_indices_by_piece: dict[int, list[int]] = {
        piece_id: [] for piece_id in target_ids
    }
    for profile_index, piece_ids in enumerate(index.piece_ids_by_profile):
        for piece_id in piece_ids:
            if piece_id in profile_indices_by_piece:
                profile_indices_by_piece[piece_id].append(profile_index)
    spatial_scores = {}
    for piece_id, profile_indices in profile_indices_by_piece.items():
        if profile_indices:
            scores = spatial_profile_scores(
                query_profile,
                index.profiles[profile_indices],
            )
            spatial_scores[piece_id] = float(scores.max())
    if len(spatial_scores) < 2:
        return output
    retrieval_scores = {
        int(row["piece_id"]): float(row["score"]) for row in leading
    }
    retrieval_percentiles = _percentile_scores(retrieval_scores)
    spatial_percentiles = _percentile_scores(spatial_scores)
    for row in leading:
        piece_id = int(row["piece_id"])
        retrieval_percentile = retrieval_percentiles[piece_id]
        spatial_percentile = spatial_percentiles.get(piece_id, 0.0)
        row["retrieval_score"] = float(row["score"])
        row["spatial_score"] = spatial_scores.get(piece_id)
        row["retrieval_percentile"] = retrieval_percentile
        row["spatial_percentile"] = spatial_percentile
        row["score"] = (
            (1.0 - blend) * retrieval_percentile
            + blend * spatial_percentile
        )
    leading.sort(
        key=lambda row: (
            float(row["score"]),
            float(row["retrieval_score"]),
        ),
        reverse=True,
    )
    output[: len(leading)] = leading
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def _percentile_scores(scores: Mapping[int, float]) -> dict[int, float]:
    ordered = sorted(scores, key=scores.get)
    if len(ordered) == 1:
        return {ordered[0]: 1.0}
    denominator = len(ordered) - 1
    return {
        piece_id: rank / denominator
        for rank, piece_id in enumerate(ordered)
    }

def _load_grayscale_batch(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    # Kept local to avoid importing training code (and its CLI dependencies) in
    # the inference path.
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
        gray = ImageOps.grayscale(image).convert("RGB")
        image.close()
        resized = gray.resize((224, 224), Image.Resampling.BICUBIC)
        gray.close()
        array = np.asarray(resized, dtype=np.float32) / 255.0
        resized.close()
        arrays.append(np.transpose((array - mean) / std, (2, 0, 1)))
    return torch.from_numpy(np.stack(arrays))
