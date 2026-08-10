from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFile, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.model_assets import load_pretrained
from ffxiv_prism.paths import PROCESSED_DIR, portable_path, resolve_project_path
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.evaluation_split import load_split_assignments
from ffxiv_prism.vision.retrieval import DEFAULT_INDEX_DIR, load_index

from scripts.evaluate_glamour_matching import (
    ROLE_PROFILES,
    TAG_PENALTY_PROFILES,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    collect_crop_records,
    evaluate_config,
    summarize,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
DINO_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a visual embedding model for player glamour matching.")
    parser.add_argument("--model-id", default="patrickjohncyh/fashion-clip")
    parser.add_argument(
        "--encoder-kind",
        choices=("clip", "dinov2", "dinov3", "auto"),
        default="auto",
    )
    parser.add_argument("--preprocess", choices=("color", "gray", "edge"), default="color")
    parser.add_argument(
        "--pooling",
        choices=("cls", "mean", "cls_mean"),
        default="cls",
        help="DINO token pooling; ignored by CLIP-like encoders.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/indexes/candidate-metadata/metadata.jsonl"),
    )
    parser.add_argument("--examples", type=Path, default=PROCESSED_DIR / "glamour_examples.jsonl")
    parser.add_argument(
        "--slots",
        nargs="+",
        choices=SLOTS,
        default=list(SLOTS),
        help="Restrict candidate rows, query crops, and metrics to selected slots.",
    )
    parser.add_argument(
        "--quality-manifest",
        type=Path,
        help="Optional JSONL manifest; only example_id rows with passed=true are kept.",
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--split-roles",
        help="Comma-separated manifest roles to evaluate; blind_test should remain unused.",
    )
    parser.add_argument("--regions-dir", type=Path, default=Path("data/debug/outfit-regions/overlap-final-201"))
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--crop-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--limit-candidates", type=int, default=None, help="Debug only: build a partial candidate index.")
    parser.add_argument("--limit-crops", type=int, default=None, help="Debug only: embed only the first N crop files.")
    args = parser.parse_args()
    args.device = str(resolve_torch_device(args.device))

    slug = safe_slug(
        f"{args.model_id}-{args.preprocess}"
        + (f"-{args.pooling}" if args.pooling != "cls" else "")
    )
    index_dir = args.index_dir or Path("data/indexes") / slug
    crop_cache = args.crop_cache or PROCESSED_DIR / f"glamour_crop_embeddings_{slug}.npz"
    output = args.output or PROCESSED_DIR / f"glamour_matching_eval_results_{slug}.json"

    selected_slots = set(args.slots)
    metadata = [
        row
        for row in read_jsonl(args.metadata)
        if str(row.get("slot")) in selected_slots
    ]
    if not metadata:
        raise ValueError(f"Candidate metadata is empty: {args.metadata}")
    if args.limit_candidates is not None:
        metadata = metadata[: args.limit_candidates]
    encoder = build_encoder(
        args.model_id,
        args.encoder_kind,
        args.device,
        args.preprocess,
        pooling=args.pooling,
    )

    candidate_embeddings = load_or_build_candidate_embeddings(
        encoder,
        metadata,
        index_dir,
        args.batch_size,
    )
    examples = read_jsonl(args.examples)
    selected_roles = parse_split_roles(args.split_roles)
    if selected_roles:
        if args.split_manifest is None or not args.split_manifest.is_file():
            raise ValueError("--split-roles requires an existing --split-manifest")
        assignments = load_split_assignments(args.split_manifest)
        examples = [
            example
            for example in examples
            if assignments.get(str(example["glamour_id"])) in selected_roles
        ]
    if args.quality_manifest is not None:
        quality_by_id = {
            str(row["example_id"]): bool(row.get("passed"))
            for row in read_jsonl(args.quality_manifest)
        }
        examples = [
            example
            for example in examples
            if quality_by_id.get(str(example["example_id"]), False)
        ]
    crop_records = collect_crop_records(examples, args.regions_dir)
    crop_records = {
        example_id: [
            row
            for row in rows
            if str(row.get("slot")) in selected_slots
        ]
        for example_id, rows in crop_records.items()
    }
    crop_embeddings = load_or_build_crop_embeddings(
        encoder,
        crop_records,
        crop_cache,
        args.batch_size,
        limit_crops=args.limit_crops,
    )
    search_results = search_crops(
        metadata,
        candidate_embeddings,
        crop_records,
        crop_embeddings,
        args.candidate_k,
        device=args.device,
    )
    available_piece_ids = available_pieces_by_slot(metadata)
    available_appearance_keys = available_appearance_keys_by_slot(metadata)

    configs = []
    if args.sweep:
        for role_profile in sorted(ROLE_PROFILES):
            for tag_profile in sorted(TAG_PENALTY_PROFILES):
                for aggregate in ("max", "sum"):
                    configs.append((role_profile, tag_profile, aggregate))
    else:
        configs.append(("overlap", "mild", "max"))

    evaluated = []
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
            role_profile=role_profile,
            tag_penalty_profile=tag_profile,
            aggregate=aggregate,
        )
        evaluated.append((summary, outcomes))

    best_summary, best_outcomes = max(
        evaluated,
        key=lambda item: (
            item[0]["appearance_recall_at_1"],
            item[0]["appearance_mrr"],
            item[0]["appearance_recall_at_k"],
            item[0]["exact_recall_at_1"],
        ),
    )
    payload = {
        **best_summary,
        "model_id": args.model_id,
        "encoder_kind": encoder.kind,
        "preprocess": args.preprocess,
        "pooling": args.pooling,
        "split_manifest": portable_path(args.split_manifest),
        "split_roles": sorted(selected_roles),
        "slots": sorted(selected_slots),
        "quality_manifest": portable_path(args.quality_manifest),
        "index_dir": portable_path(index_dir),
        "crop_cache": portable_path(crop_cache),
        "candidate_k": args.candidate_k,
        "sweep_results": [summary for summary, _ in evaluated],
        "cases": best_outcomes,
    }
    write_json(output, payload)
    print(
        f"model={args.model_id} preprocess={args.preprocess} eligible={best_summary['eligible_case_count']} "
        f"exact@1={best_summary['exact_recall_at_1']:.3f} exact@{args.top_k}={best_summary['exact_recall_at_k']:.3f} "
        f"appearance@1={best_summary['appearance_recall_at_1']:.3f} appearance@{args.top_k}={best_summary['appearance_recall_at_k']:.3f} "
        f"mrr={best_summary['appearance_mrr']:.3f} "
        f"profile={best_summary['role_profile']}/{best_summary['tag_penalty_profile']}/{best_summary['aggregate']}"
    )
    for slot, row in best_summary["by_slot"].items():
        print(
            f"  {slot}: app@1={row['appearance_recall_at_1']:.3f} "
            f"app@{args.top_k}={row['appearance_recall_at_k']:.3f} mrr={row['appearance_mrr']:.3f}"
        )
    return 0


class EmbeddingEncoder:
    def __init__(self, model_id: str, kind: str, device: str, preprocess: str) -> None:
        self.model_id = model_id
        self.kind = kind
        self.device = resolve_torch_device(device)
        self.preprocess = preprocess

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


def load_or_build_candidate_embeddings(
    encoder: EmbeddingEncoder,
    metadata: list[dict[str, Any]],
    index_dir: Path,
    batch_size: int,
) -> np.ndarray:
    if not metadata:
        raise ValueError("Refusing to build or overwrite an index from empty metadata")
    embeddings_path = index_dir / "embeddings.npy"
    metadata_path = index_dir / "metadata.jsonl"
    manifest_path = index_dir / "manifest.json"
    expected_paths = [row["local_path"] for row in metadata]
    existing_artifacts = (embeddings_path.is_file(), metadata_path.is_file())
    if any(existing_artifacts) and not all(existing_artifacts):
        raise ValueError(
            "Candidate index is incomplete; refusing to overwrite existing "
            f"artifacts in {index_dir}. Rebuild into a new directory."
        )
    if all(existing_artifacts):
        cached_metadata = read_jsonl(metadata_path)
        cached_paths = [row["local_path"] for row in cached_metadata]
        if cached_paths != expected_paths:
            raise ValueError(
                "Candidate metadata does not match the existing index; refusing "
                f"to overwrite {index_dir} ({len(cached_paths)} cached rows vs "
                f"{len(expected_paths)} requested rows). Pass the metadata used "
                "to build this index or choose a new index directory."
            )
        cached_embeddings = np.load(embeddings_path, mmap_mode="r")
        if cached_embeddings.ndim != 2 or cached_embeddings.shape[0] != len(
            cached_metadata
        ):
            raise ValueError(
                "Candidate index row mismatch; refusing to use or overwrite "
                f"{index_dir}: embeddings={cached_embeddings.shape}, "
                f"metadata_rows={len(cached_metadata)}"
            )
        return cached_embeddings

    valid_rows = []
    paths = []
    for row in metadata:
        path = resolve_project_path(row["local_path"])
        if path is not None and path.is_file():
            valid_rows.append(row)
            paths.append(path)
    embeddings = encoder.encode_paths(paths, batch_size)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings, allow_pickle=False)
    write_jsonl(metadata_path, valid_rows)
    write_json(
        manifest_path,
        {
            "version": 1,
            "model_id": encoder.model_id,
            "encoder_kind": encoder.kind,
            "preprocess": encoder.preprocess,
            "pooling": getattr(encoder, "pooling", "cls"),
            "embedding_count": len(valid_rows),
            "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 and len(embeddings) else 0,
            "embeddings": portable_path(embeddings_path),
            "metadata": portable_path(metadata_path),
        },
    )
    metadata[:] = valid_rows
    return embeddings


def load_or_build_crop_embeddings(
    encoder: EmbeddingEncoder,
    crop_records: dict[str, list[dict[str, Any]]],
    cache_path: Path,
    batch_size: int,
    limit_crops: int | None = None,
) -> dict[str, np.ndarray]:
    crop_paths = sorted({str(record["path"]) for records in crop_records.values() for record in records})
    if limit_crops is not None:
        crop_paths = crop_paths[:limit_crops]
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        cached_paths = [str(path) for path in cached["paths"].tolist()]
        if (
            cached_paths == crop_paths
            and str(cached["model_id"].tolist()) == encoder.model_id
            and str(cached["preprocess"].tolist()) == encoder.preprocess
            and (
                str(cached["pooling"].tolist())
                if "pooling" in cached.files
                else "cls"
            )
            == getattr(encoder, "pooling", "cls")
        ):
            embeddings = cached["embeddings"]
            return {path: embeddings[index] for index, path in enumerate(crop_paths)}

    embeddings = encoder.encode_paths([Path(path) for path in crop_paths], batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        paths=np.asarray(crop_paths),
        embeddings=embeddings,
        model_id=np.asarray(encoder.model_id),
        preprocess=np.asarray(encoder.preprocess),
        pooling=np.asarray(getattr(encoder, "pooling", "cls")),
    )
    return {path: embeddings[index] for index, path in enumerate(crop_paths)}


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
            candidate_array = np.asarray(candidate_embeddings[indices])
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
                        candidate_array,
                        device=torch_device,
                    )
                query = torch.as_tensor(crop_embeddings[path], device=torch_device)
                scores_tensor = candidate_tensors[group_key] @ query
                order = torch.argsort(scores_tensor, descending=True).cpu().numpy()
                scores = scores_tensor.cpu().numpy()
            else:
                scores = candidate_array @ crop_embeddings[path]
                order = np.argsort(scores)[::-1]
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


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")


def parse_split_roles(value: str | None) -> set[str]:
    if not value:
        return set()
    roles = {part.strip() for part in value.split(",") if part.strip()}
    valid = {"train", "validation", "development_test", "blind_test"}
    invalid = sorted(roles - valid)
    if invalid:
        raise ValueError(f"Invalid split roles: {invalid}")
    return roles


if __name__ == "__main__":
    raise SystemExit(main())
