"""Shared inference primitives; training entry points live in scripts."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from ffxiv_prism.model_assets import load_pretrained
from ffxiv_prism.torch_device import resolve_torch_device
from .session import frozen_resource, timed
DINO_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def similarity_order(scores: np.ndarray) -> np.ndarray:
    """Use shared tie handling at FP32 cosine precision on every device.

    Repeated catalogue views often have equal or numerically indistinguishable
    scores. Preserve catalogue order instead of backend-specific argsort ties;
    otherwise rank fusion amplifies sub-micro cosine differences.
    """
    return np.argsort(-np.round(np.asarray(scores), decimals=6), kind="stable")


class EmbeddingEncoder:
    def __init__(self, model_id: str, kind: str, device: str, preprocess: str) -> None:
        self.model_id = model_id
        self.kind = kind
        self.device = resolve_torch_device(device)
        self.preprocess = preprocess

    @timed("query_encode.frozen")
    def encode_paths(self, paths: Sequence[Path], batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            outputs.append(self.encode_batch(batch_paths))
            done = min(start + batch_size, len(paths))
            if done == len(paths) or done % (batch_size * 20) == 0:
                print(f"embedded={done}/{len(paths)}", flush=True)
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)

    def encode_batch(self, paths: Sequence[Path]) -> np.ndarray:
        raise NotImplementedError

    def prepare_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return preprocess_image(image.convert("RGB"), self.preprocess)


class ClipLikeEncoder(EmbeddingEncoder):
    def __init__(self, model_id: str, kind: str, device: str, preprocess: str) -> None:
        super().__init__(model_id, kind, device, preprocess)
        from transformers import AutoConfig, AutoProcessor, CLIPModel, SiglipModel

        self.processor = load_pretrained(AutoProcessor, model_id)
        config = load_pretrained(AutoConfig, model_id)
        if getattr(config, "model_type", None) == "siglip":
            self.model = load_pretrained(SiglipModel, model_id).to(self.device).eval()
        else:
            self.model = load_pretrained(CLIPModel, model_id).to(self.device).eval()

    def encode_batch(self, paths: Sequence[Path]) -> np.ndarray:
        images = [self.prepare_image(path) for path in paths]
        try:
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                features = self.model.get_image_features(pixel_values=pixel_values)
                if not isinstance(features, torch.Tensor):
                    features = getattr(features, "image_embeds", None) or getattr(features, "pooler_output", None)
                if features is None:
                    raise RuntimeError("Model did not return image features")
                features = torch.nn.functional.normalize(features, dim=-1)
            return features.cpu().numpy().astype(np.float32, copy=False)
        finally:
            for image in images:
                image.close()


class DinoV2Encoder(EmbeddingEncoder):
    def __init__(
        self,
        model_id: str,
        kind: str,
        device: str,
        preprocess: str,
        pooling: str = "cls",
    ) -> None:
        super().__init__(model_id, kind, device, preprocess)
        self.pooling = pooling
        from transformers import AutoModel

        self.model = load_pretrained(AutoModel, model_id).to(self.device).eval()

    def encode_batch(self, paths: Sequence[Path]) -> np.ndarray:
        arrays = []
        for path in paths:
            image = self.prepare_image(path).resize((224, 224), Image.Resampling.BICUBIC)
            try:
                arr = np.asarray(image, dtype=np.float32) / 255.0
            finally:
                image.close()
            arrays.append(np.transpose((arr - DINO_MEAN) / DINO_STD, (2, 0, 1)))
        pixel_values = torch.from_numpy(np.stack(arrays)).to(self.device)
        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
            features = pool_dino_tokens(outputs.last_hidden_state, self.pooling)
            features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy().astype(np.float32, copy=False)


def build_encoder(
    model_id: str,
    kind: str,
    device: str,
    preprocess: str,
    pooling: str = "cls",
) -> EmbeddingEncoder:
    return frozen_resource(
        ("encoder", model_id, kind, str(resolve_torch_device(device)), preprocess, pooling),
        lambda: _build_encoder(model_id, kind, device, preprocess, pooling),
    )


def _build_encoder(model_id, kind, device, preprocess, pooling):
    if kind == "auto":
        normalized_model_id = str(model_id).lower()
        if "dinov3" in normalized_model_id:
            kind = "dinov3"
        elif "dinov2" in normalized_model_id:
            kind = "dinov2"
        else:
            kind = "clip"
    if kind in {"dinov2", "dinov3"}:
        return DinoV2Encoder(model_id, kind, device, preprocess, pooling)
    return ClipLikeEncoder(model_id, kind, device, preprocess)


def pool_dino_tokens(tokens: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return tokens[:, 0]
    patch_mean = tokens[:, 1:].mean(dim=1)
    if pooling == "mean":
        return patch_mean
    if pooling == "cls_mean":
        cls = torch.nn.functional.normalize(tokens[:, 0], dim=-1)
        mean = torch.nn.functional.normalize(patch_mean, dim=-1)
        return cls + mean
    raise ValueError(f"Unsupported DINO pooling: {pooling}")


def preprocess_image(image: Image.Image, mode: str) -> Image.Image:
    if mode == "color":
        return image.copy()
    if mode == "gray":
        return ImageOps.grayscale(image).convert("RGB")
    if mode == "edge":
        gray = ImageOps.grayscale(image)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageEnhance.Contrast(edges).enhance(1.8)
        return edges.convert("RGB")
    raise ValueError(f"Unknown preprocess mode: {mode}")


@timed("candidate_search")
def search_crops(
    metadata: list[dict[str, Any]],
    candidate_embeddings: np.ndarray,
    crop_records: dict[str, list[dict[str, Any]]],
    crop_embeddings: dict[str, np.ndarray],
    candidate_k: int,
    device: str = "cpu",
    candidate_crop_routes: Mapping[str, Sequence[str]] | None = None,
    candidate_gender_by_example: Mapping[str, str] | None = None,
    candidate_gender_fallback: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    torch_device = resolve_torch_device(device)
    use_cuda = torch_device.type == "cuda"
    index_groups: dict[tuple[str, tuple[str, ...], str], np.ndarray] = {}
    candidate_tensors: dict[
        tuple[str, tuple[str, ...], str], torch.Tensor
    ] = {}
    candidate_arrays: dict[tuple[str, tuple[str, ...], str], np.ndarray] = {}
    piece_ids_by_group: dict[
        tuple[str, tuple[str, ...], str], np.ndarray
    ] = {}
    results: dict[str, list[dict[str, Any]]] = {}
    piece_genders: dict[tuple[str, int], set[str]] = defaultdict(set)
    if candidate_gender_fallback:
        for row in metadata:
            row_slot = str(row.get("slot", ""))
            row_gender = str(row.get("model_gender", "")).lower()
            for piece in row.get("pieces", []):
                piece_genders[(row_slot, int(piece["piece_id"]))].add(row_gender)
    for records in crop_records.values():
        for record in records:
            path = str(record["path"])
            if path in results or path not in crop_embeddings:
                continue
            allowed_crops = tuple(
                (candidate_crop_routes or {}).get(str(record.get("name")), ())
            )
            candidate_gender = str(
                (candidate_gender_by_example or {}).get(
                    str(record.get("example_id")),
                    "",
                )
            ).lower()
            group_key = (
                str(record["slot"]),
                allowed_crops,
                candidate_gender,
            )
            if group_key not in index_groups:
                allowed = set(allowed_crops)
                index_groups[group_key] = np.asarray(
                    [
                        index
                        for index, row in enumerate(metadata)
                        if row.get("slot") == record["slot"]
                        and (
                            not allowed
                            or str(row.get("candidate_crop")) in allowed
                        )
                        and (
                            not candidate_gender
                            or str(row.get("model_gender", "")).lower()
                            == candidate_gender
                            or (
                                candidate_gender_fallback
                                and any(
                                    candidate_gender
                                    not in piece_genders.get(
                                        (
                                            str(row.get("slot", "")),
                                            int(piece["piece_id"]),
                                        ),
                                        set(),
                                    )
                                    for piece in row.get("pieces", [])
                                )
                            )
                        )
                    ],
                    dtype=np.int64,
                )
            indices = index_groups[group_key]
            if not len(indices):
                raise ValueError(
                    "Candidate routing removed every index row for "
                    f"slot={record['slot']} crops={allowed_crops} "
                    f"gender={candidate_gender or 'any'}"
                )
            # Advanced indexing copies the entire routed matrix. Materialize it
            # only once per route, not once per query crop (also on CUDA).
            if group_key not in candidate_arrays and group_key not in candidate_tensors:
                candidate_arrays[group_key] = np.asarray(candidate_embeddings[indices])
            if group_key not in piece_ids_by_group:
                max_pieces = max(
                    len(metadata[int(index)]["pieces"])
                    for index in indices
                )
                piece_ids = np.full(
                    (len(indices), max_pieces),
                    -1,
                    dtype=np.int64,
                )
                for local_index, metadata_index in enumerate(indices):
                    for piece_index, piece in enumerate(
                        metadata[int(metadata_index)]["pieces"]
                    ):
                        piece_ids[local_index, piece_index] = int(piece["piece_id"])
                piece_ids_by_group[group_key] = piece_ids
            if use_cuda:
                if group_key not in candidate_tensors:
                    candidate_tensors[group_key] = torch.tensor(
                        candidate_arrays.pop(group_key),
                        device=torch_device,
                    )
                query = torch.as_tensor(crop_embeddings[path], device=torch_device)
                scores_tensor = candidate_tensors[group_key] @ query
                scores = scores_tensor.cpu().numpy()
            else:
                scores = candidate_arrays[group_key] @ crop_embeddings[path]
            order = similarity_order(scores)
            results[path] = _candidate_rows_from_order(
                metadata,
                indices,
                order,
                scores,
                piece_ids_by_group[group_key],
                candidate_k,
            )
    return results


def _candidate_rows_from_order(
    metadata: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    order: np.ndarray,
    scores: np.ndarray,
    piece_ids: np.ndarray,
    candidate_k: int,
) -> list[dict[str, Any]]:
    # Flatten pieces in exact candidate-row/piece order, then retain the first
    # occurrence of each piece with NumPy.  Slot indexes contain many crop,
    # gender, race, and dye rows per piece; avoiding a Python scan through all
    # of them is the dominant strict-evaluation speedup.
    ordered_piece_ids = piece_ids[order].reshape(-1)
    valid_positions = np.flatnonzero(ordered_piece_ids >= 0)
    valid_piece_ids = ordered_piece_ids[valid_positions]
    _, first_indices = np.unique(valid_piece_ids, return_index=True)
    first_positions = np.sort(valid_positions[first_indices])[:candidate_k]
    piece_width = piece_ids.shape[1]

    rows: list[dict[str, Any]] = []
    for flat_position in first_positions:
        order_position = int(flat_position) // piece_width
        piece_position = int(flat_position) % piece_width
        offset = int(order[order_position])
        metadata_index = int(indices[offset])
        row = metadata[metadata_index]
        piece = row["pieces"][piece_position]
        rows.append(
            {
                "rank": 0,
                "score": float(scores[offset]),
                **piece,
                "image_row_id": row["image_row_id"],
                "local_path": row["local_path"],
                "source_image_url": row["source_image_url"],
                "model_gender": row.get("model_gender"),
                "race": row.get("race"),
                "dye_state": row.get("dye_state"),
            }
        )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
