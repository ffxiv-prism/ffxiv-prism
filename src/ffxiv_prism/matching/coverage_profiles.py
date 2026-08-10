from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


COVERAGE_PROFILE_VERSION = 1


@dataclass(frozen=True)
class HeadCoverageIndex:
    exposed_fractions: np.ndarray
    person_fractions: np.ndarray
    metadata_indices: np.ndarray
    piece_ids_by_profile: tuple[tuple[int, ...], ...]
    model_id: str
    candidate_crop: str


def encode_head_coverage_profiles(
    parser: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure visible hair/face share as a proxy for headwear coverage."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    exposed_values: list[float] = []
    person_values: list[float] = []
    for start in range(0, len(rows), batch_size):
        images = [_load_image(row) for row in rows[start : start + batch_size]]
        segmentations = parser.parse(images)
        for segmentation in segmentations:
            person = segmentation != 0
            exposed = np.isin(segmentation, (2, 11))
            person_count = max(1, int(person.sum()))
            exposed_values.append(float(exposed.sum() / person_count))
            person_values.append(float(person.mean()))
        for image in images:
            image.close()
    return (
        np.asarray(exposed_values, dtype=np.float32),
        np.asarray(person_values, dtype=np.float32),
    )


def encode_head_parsing_spatial_profiles(
    parser: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    grid_size: int = 16,
) -> np.ndarray:
    """Encode scale-normalized 2D hair/face and headwear parsing layouts."""

    if batch_size < 1 or grid_size < 2:
        raise ValueError("batch_size must be positive and grid_size at least two")
    output: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        images = [_load_image(row) for row in rows[start : start + batch_size]]
        parse = getattr(parser, "parse_native", parser.parse)
        segmentations = parse(images)
        output.extend(
            _head_parsing_spatial_profile(segmentation, grid_size)
            for segmentation in segmentations
        )
        for image in images:
            image.close()
    if not output:
        return np.empty((0, 1, grid_size * grid_size * 3), dtype=np.float16)
    return np.stack(output).astype(np.float16, copy=False)[:, None, :]


def _head_parsing_spatial_profile(
    segmentation: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    person = segmentation != 0
    ys, xs = np.where(person)
    dimension = grid_size * grid_size * 3
    if not len(xs):
        return np.zeros(dimension, dtype=np.float32)
    crop = segmentation[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    channels = []
    for label_ids in ((2, 11), (1, 4, 17), (0,)):
        mask = np.isin(crop, label_ids).astype(np.uint8) * 255
        resized = Image.fromarray(mask, mode="L").resize(
            (grid_size, grid_size),
            Image.Resampling.BILINEAR,
        )
        channels.append(
            np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0
        )
        resized.close()
    profile = np.concatenate(channels)
    norm = float(np.linalg.norm(profile))
    if norm > 0.0:
        profile /= norm
    return profile.astype(np.float32, copy=False)


def save_head_coverage_index(path: Path, index: HeadCoverageIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray(COVERAGE_PROFILE_VERSION),
        exposed_fractions=index.exposed_fractions.astype(np.float16),
        person_fractions=index.person_fractions.astype(np.float16),
        metadata_indices=index.metadata_indices,
        piece_ids=np.asarray(
            [",".join(str(value) for value in values) for values in index.piece_ids_by_profile]
        ),
        model_id=np.asarray(index.model_id),
        candidate_crop=np.asarray(index.candidate_crop),
    )


def load_head_coverage_index(path: Path) -> HeadCoverageIndex:
    with np.load(path, allow_pickle=False) as payload:
        version = int(payload["version"].tolist())
        if version != COVERAGE_PROFILE_VERSION:
            raise ValueError(f"Unsupported coverage profile version: {version}")
        return HeadCoverageIndex(
            exposed_fractions=np.asarray(payload["exposed_fractions"], dtype=np.float32),
            person_fractions=np.asarray(payload["person_fractions"], dtype=np.float32),
            metadata_indices=np.asarray(payload["metadata_indices"], dtype=np.int64),
            piece_ids_by_profile=tuple(
                tuple(int(value) for value in str(text).split(",") if value)
                for text in payload["piece_ids"].tolist()
            ),
            model_id=str(payload["model_id"].tolist()),
            candidate_crop=str(payload["candidate_crop"].tolist()),
        )


def rerank_candidates_with_head_coverage(
    candidates: Sequence[Mapping[str, Any]],
    index: HeadCoverageIndex,
    query_exposed_fraction: float,
    *,
    rerank_k: int,
    blend: float,
    max_difference: float | None = None,
    minimum_query_exposed: float | None = None,
    minimum_top_difference: float | None = None,
    maximum_retrieval_margin: float | None = None,
) -> list[dict[str, Any]]:
    if rerank_k < 1:
        raise ValueError("rerank_k must be positive")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between zero and one")
    if max_difference is not None and max_difference < 0.0:
        raise ValueError("max_difference must be non-negative")
    if (
        minimum_query_exposed is not None
        and query_exposed_fraction < minimum_query_exposed
    ):
        return [dict(row) for row in candidates]
    output = [dict(row) for row in candidates]
    leading = output[: min(rerank_k, len(output))]
    if len(leading) < 2:
        return output
    target_ids = {int(row["piece_id"]) for row in leading}
    differences: dict[int, float] = {}
    candidate_fractions: dict[int, float] = {}
    for profile_index, piece_ids in enumerate(index.piece_ids_by_profile):
        candidate_fraction = float(index.exposed_fractions[profile_index])
        difference = abs(query_exposed_fraction - candidate_fraction)
        for piece_id in piece_ids:
            if (
                piece_id in target_ids
                and difference < differences.get(piece_id, float("inf"))
            ):
                differences[piece_id] = difference
                candidate_fractions[piece_id] = candidate_fraction
    if len(differences) < 2:
        return output
    retrieval_scores = {
        int(row["piece_id"]): float(row["score"]) for row in leading
    }
    retrieval_percentiles = _percentile_scores(retrieval_scores)
    top_piece_id = int(leading[0]["piece_id"])
    if (
        minimum_top_difference is not None
        and differences.get(top_piece_id, 0.0) < minimum_top_difference
    ):
        return output
    if maximum_retrieval_margin is not None:
        retrieval_margin = float(leading[0]["score"]) - float(leading[1]["score"])
        if retrieval_margin > maximum_retrieval_margin:
            return output

    if max_difference is not None:
        for row in leading:
            piece_id = int(row["piece_id"])
            retrieval_percentile = retrieval_percentiles[piece_id]
            difference = differences.get(piece_id)
            plausible = difference is not None and difference <= max_difference
            row["retrieval_score"] = float(row["score"])
            row["head_exposed_fraction"] = query_exposed_fraction
            row["candidate_head_exposed_fraction"] = candidate_fractions.get(
                piece_id
            )
            row["coverage_difference"] = difference
            row["coverage_plausible"] = plausible
            row["retrieval_percentile"] = retrieval_percentile
            row["score"] = (1.0 if plausible else 0.0) + retrieval_percentile
        leading.sort(
            key=lambda row: (float(row["score"]), float(row["retrieval_score"])),
            reverse=True,
        )
        output[: len(leading)] = leading
        for rank, row in enumerate(output, start=1):
            row["rank"] = rank
        return output

    coverage_percentiles = _percentile_scores(
        {piece_id: -difference for piece_id, difference in differences.items()}
    )
    for row in leading:
        piece_id = int(row["piece_id"])
        retrieval_percentile = retrieval_percentiles[piece_id]
        coverage_percentile = coverage_percentiles.get(piece_id, 0.0)
        row["retrieval_score"] = float(row["score"])
        row["head_exposed_fraction"] = query_exposed_fraction
        row["candidate_head_exposed_fraction"] = candidate_fractions.get(piece_id)
        row["coverage_difference"] = differences.get(piece_id)
        row["retrieval_percentile"] = retrieval_percentile
        row["coverage_percentile"] = coverage_percentile
        row["score"] = (
            (1.0 - blend) * retrieval_percentile
            + blend * coverage_percentile
        )
    leading.sort(
        key=lambda row: (float(row["score"]), float(row["retrieval_score"])),
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
    return {piece_id: rank / denominator for rank, piece_id in enumerate(ordered)}


def _load_image(row: Mapping[str, Any]) -> Image.Image:
    with Image.open(Path(str(row["path"]))) as source:
        image = source.convert("RGB")
    box = row.get("crop_box_normalized")
    if box is None:
        return image
    left, top, right, bottom = (float(value) for value in box)
    crop = image.crop(
        (
            int(round(left * image.width)),
            int(round(top * image.height)),
            int(round(right * image.width)),
            int(round(bottom * image.height)),
        )
    )
    image.close()
    return crop
