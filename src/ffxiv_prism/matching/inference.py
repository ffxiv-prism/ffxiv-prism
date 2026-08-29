from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from ffxiv_prism.paths import (
    APPEARANCE_EQUIVALENCE_PATH,
    INDEXES_DIR,
    PROJECT_ROOT,
    portable_path,
    resolve_project_path,
)
from ffxiv_prism.model_assets import is_remote_model_id, model_weight_files
from ffxiv_prism.vision.outfit_regions import CropRegion


SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
CURRENT_BEST_SELECTION_PATH = (
    PROJECT_ROOT / "data" / "current_best_selection.json"
)
VISIBLE_WEIGHTS = {
    "visible": 1.0,
    "partially_visible": 0.72,
    "occluded": 0.45,
    "pose_uncertain": 0.0,
    "out_of_frame": 0.0,
}
ROLE_WEIGHTS = {"primary": 1.0, "auxiliary": 0.78, "context": 0.72}

HEAD_CANDIDATE_CROP_ROUTES = {
    "head_upper_accessory": ("head_upper_accessory",),
    "head_face_accessory": ("head_face_accessory",),
    "head_context": ("full",),
}
HEAD_PROMOTED_CANDIDATE_CROP_ROUTES = {
    "head_upper_accessory": ("head_face_accessory",),
}
HANDS_CANDIDATE_CROP_ROUTES = {
    "hands_left": ("left_arm_tight", "left_forearm_context"),
    "hands_forearm_left": ("left_arm_tight", "left_forearm_context"),
    "hands_right": ("right_arm_tight", "right_forearm_context"),
    "hands_forearm_right": ("right_arm_tight", "right_forearm_context"),
    "hands_arm_cover_context": ("both_hands", "hands_upper_context"),
}


@dataclass(frozen=True)
class SlotInferenceConfig:
    slot: str
    index_dir: Path
    query_checkpoint: Path | None
    reranker_checkpoint: Path | None = None
    fusion: str = "max"
    rrf_k: int = 60
    candidate_crop_routes: Mapping[str, Sequence[str]] | None = None
    query_crop_names: tuple[str, ...] | None = None
    ensemble_model_ids: tuple[str, ...] = ()
    ensemble_index_dirs: tuple[Path, ...] = ()
    ensemble_query_checkpoints: tuple[Path | None, ...] = ()
    morphology_checkpoint: Path | None = None
    ensemble_rrf_k: int = 1
    ensemble_pool_branches: int | None = None
    ensemble_pool_rrf_k: int = 1
    ensemble_final_pool_branches: int | None = None
    ensemble_fusion_order: str = "branch_then_crop"
    ensemble_branch_weights: tuple[float, ...] = ()
    ensemble_gate_baseline_branch_weights: tuple[float, ...] = ()
    ensemble_top1_gate_max_base_rank: int | None = None
    ensemble_top1_gate_max_last_branch_rank: int | None = None
    ensemble_top1_gate_max_base_last_branch_rank: int | None = None
    ensemble_top1_gate_min_last_branch_rank_ratio: float | None = None
    gender_routing: bool = False
    occupied_slot_multiplier: float | None = None
    occupied_slot_gate_max_base_rank: int | None = None
    occupied_slot_body_support_top_k: int | None = None
    spatial_index: Path | None = None
    spatial_query_crop_names: tuple[str, ...] | None = None
    spatial_rerank_k: int = 10
    spatial_blend: float = 0.35
    style_model_id: str | None = None
    style_query_crop_name: str | None = None
    style_rerank_k: int = 50
    style_blend: float = 0.35
    style_min_confidence: float = 0.5
    no_equipment_model_id: str | None = None
    no_equipment_query_crop_names: tuple[str, ...] = ()
    no_equipment_min_probability: float = 0.8
    benchmark_top1: float | None = None
    benchmark_top10: float | None = None
    benchmark_scope: str = ""


def load_current_best_selection(
    path: Path = CURRENT_BEST_SELECTION_PATH,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError("Unsupported current-best selection version")
    suffix = str(payload.get("candidate_suffix") or "")
    if not suffix or any(char not in "0123456789-" for char in suffix):
        raise ValueError("Invalid current-best candidate suffix")
    checkpoints = payload.get("checkpoints")
    benchmarks = payload.get("benchmarks")
    if not isinstance(checkpoints, dict) or not isinstance(benchmarks, dict):
        raise ValueError("Current-best selection is missing checkpoints or benchmarks")
    missing_slots = sorted(set(SLOTS) - set(benchmarks))
    if missing_slots:
        raise ValueError(
            "Current-best selection is missing benchmark slots: "
            + ", ".join(missing_slots)
        )
    return payload


def current_best_configs() -> dict[str, SlotInferenceConfig]:
    models = PROJECT_ROOT / "data" / "models" / "current-best"
    selection = load_current_best_selection()
    suffix = str(selection["candidate_suffix"])
    checkpoints = selection["checkpoints"]
    benchmarks = selection["benchmarks"]
    body_policy = str(selection.get("body_policy") or "baseline")
    if body_policy not in {"baseline", "safe-gate"}:
        raise ValueError(f"Unsupported BODY current-best policy: {body_policy}")

    def index(name: str) -> Path:
        return INDEXES_DIR / f"{name}-{suffix}"

    def checkpoint(name: str) -> Path:
        value = checkpoints.get(name)
        if not value:
            raise ValueError(f"Current-best checkpoint is missing: {name}")
        return models / str(value)

    def benchmark(slot: str, metric: str) -> float:
        return float(benchmarks[slot][metric])

    def eligible(slot: str) -> int:
        return int(benchmarks[slot]["eligible"])

    body_ensemble_model_ids = ("var/models/dinov3-vitb16-pretrain-lvd1689m",)
    body_ensemble_indexes = (
        index("facebook-dinov3-vitb16-body-multiscale-v3"),
    )
    body_ensemble_checkpoints: tuple[Path | None, ...] = (None,)
    body_weights = (1.0, 1.0)
    body_baseline_weights: tuple[float, ...] = ()
    body_gate: tuple[int | None, int | None, int | None, float | None] = (
        None,
        None,
        None,
        None,
    )
    if body_policy == "safe-gate":
        body_ensemble_model_ids += (
            "var/models/dinov3-vitb16-pretrain-lvd1689m",
        )
        body_ensemble_indexes += (
            index("facebook-dinov3-vitb16-body-multiscale-v3"),
        )
        body_ensemble_checkpoints = (None, checkpoint("body_dinov3"))
        body_weights = (1.0, 0.0, 1.25)
        body_baseline_weights = (1.0, 1.0, 0.0)
        body_gate = (10, 1, 10, 1.0)
    return {
        "BODY": SlotInferenceConfig(
            slot="BODY",
            index_dir=index(
                "facebook-dinov2-base-slot-aligned-v3-body-multiscale"
            ),
            query_checkpoint=checkpoint("body_dinov2"),
            ensemble_model_ids=body_ensemble_model_ids,
            ensemble_index_dirs=body_ensemble_indexes,
            ensemble_query_checkpoints=body_ensemble_checkpoints,
            ensemble_rrf_k=1,
            ensemble_final_pool_branches=1,
            ensemble_fusion_order="crop_then_branch",
            ensemble_branch_weights=body_weights,
            ensemble_gate_baseline_branch_weights=body_baseline_weights,
            ensemble_top1_gate_max_base_rank=body_gate[0],
            ensemble_top1_gate_max_last_branch_rank=body_gate[1],
            ensemble_top1_gate_max_base_last_branch_rank=body_gate[2],
            ensemble_top1_gate_min_last_branch_rank_ratio=body_gate[3],
            style_model_id="google/siglip2-base-patch16-224",
            style_query_crop_name="body_context",
            style_rerank_k=100,
            style_blend=0.70,
            style_min_confidence=0.45,
            benchmark_top1=benchmark("BODY", "top1"),
            benchmark_top10=benchmark("BODY", "top10"),
            benchmark_scope=(
                "strict item-disjoint three-fold pooled BODY v110 baseline "
                f"({eligible('BODY')} images): refreshed DINOv2 plus zero-shot DINOv3 with "
                "the final DINOv2 Top-1000 pool frozen. The adapted DINOv3 "
                "candidate was excluded after a fold-level domain regression; "
                "the confidence-gated style reranker is not included"
            ),
        ),
        "HEAD": SlotInferenceConfig(
            slot="HEAD",
            index_dir=index("facebook-dinov2-base-head-multiscale-v4"),
            query_checkpoint=checkpoint("head_dinov2"),
            candidate_crop_routes=HEAD_PROMOTED_CANDIDATE_CROP_ROUTES,
            query_crop_names=("head_upper_accessory",),
            ensemble_model_ids=(
                "google/siglip2-base-patch16-224",
                "var/models/dinov3-vitb16-pretrain-lvd1689m",
            ),
            ensemble_index_dirs=(
                index("google-siglip2-base-patch16-224-head-multiscale-v4"),
                index("facebook-dinov3-vitb16-head-multiscale-v4"),
            ),
            ensemble_query_checkpoints=(
                None,
                checkpoint("head_dinov3"),
            ),
            morphology_checkpoint=(
                models / "head_morphology_v73_fold01_qv65_upper_binary_ridge.npz"
            ),
            no_equipment_model_id="google/siglip2-base-patch16-224",
            no_equipment_query_crop_names=(
                "head_upper_accessory",
                "head_full_accessory",
            ),
            no_equipment_min_probability=0.80,
            ensemble_rrf_k=10,
            ensemble_pool_branches=2,
            ensemble_pool_rrf_k=1,
            benchmark_top1=benchmark("HEAD", "top1"),
            benchmark_top10=benchmark("HEAD", "top10"),
            benchmark_scope=(
                "strict three-fold pooled DINOv2+SigLIP2+DINOv3-query-adapted gated rank "
                "ensemble with upper-query to face-candidate routing "
                f"({eligible('HEAD')} images; DINOv2+SigLIP2 Top-1000 pool preserved)"
            ),
        ),
        "HANDS": SlotInferenceConfig(
            slot="HANDS",
            index_dir=index("facebook-dinov2-base-hands-multiscale-v5"),
            query_checkpoint=checkpoint("hands_dinov2"),
            fusion="rrf",
            rrf_k=1,
            candidate_crop_routes=HANDS_CANDIDATE_CROP_ROUTES,
            gender_routing=True,
            occupied_slot_multiplier=0.25,
            occupied_slot_gate_max_base_rank=3,
            occupied_slot_body_support_top_k=3,
            no_equipment_model_id="google/siglip2-base-patch16-224",
            no_equipment_query_crop_names=("hands_left", "hands_right"),
            no_equipment_min_probability=0.80,
            benchmark_top1=benchmark("HANDS", "top1"),
            benchmark_top10=benchmark("HANDS", "top10"),
            benchmark_scope=(
                f"strict item-disjoint three-fold pooled benchmark ({eligible('HANDS')} images); frozen "
                "occupied-slot Top-1 gate demotes multi-slot outfit pieces only "
                "when an exact HANDS piece is already in the base Top-3"
            ),
        ),
        "LEGS": SlotInferenceConfig(
            slot="LEGS",
            index_dir=index("facebook-dinov2-base-legs-multiscale-v6"),
            query_checkpoint=checkpoint("legs_dinov2"),
            fusion="rrf",
            rrf_k=1,
            ensemble_model_ids=(
                "var/models/dinov3-vitb16-pretrain-lvd1689m",
            ),
            ensemble_index_dirs=(
                index("facebook-dinov3-vitb16-legs-multiscale-v6"),
            ),
            ensemble_query_checkpoints=(None,),
            ensemble_rrf_k=1,
            ensemble_pool_branches=1,
            ensemble_pool_rrf_k=1,
            ensemble_final_pool_branches=1,
            style_model_id="google/siglip2-base-patch16-224",
            style_query_crop_name="legs_waist_thighs",
            style_rerank_k=100,
            style_blend=0.30,
            style_min_confidence=0.50,
            benchmark_top1=benchmark("LEGS", "top1"),
            benchmark_top10=benchmark("LEGS", "top10"),
            benchmark_scope=(
                "strict three-fold pooled DINOv2+DINOv3 branch RRF(k=1), "
                "query-crop RRF(k=1), with the final DINOv2 Top-1000 "
                f"piece pool frozen ({eligible('LEGS')} images). The confidence-gated "
                "coverage-style reranker is not included in this identity benchmark"
            ),
        ),
        "FEET": SlotInferenceConfig(
            slot="FEET",
            index_dir=index("facebook-dinov2-base-feet-multiscale-v7-clean"),
            query_checkpoint=checkpoint("feet_dinov2"),
            ensemble_model_ids=("google/siglip2-base-patch16-224",),
            ensemble_index_dirs=(
                index(
                    "google-siglip2-base-patch16-224-feet-multiscale-v7-clean"
                ),
            ),
            ensemble_query_checkpoints=(None,),
            ensemble_rrf_k=1,
            fusion="rrf",
            rrf_k=1,
            style_model_id="google/siglip2-base-patch16-224",
            style_query_crop_name="feet_height_context",
            style_rerank_k=50,
            style_blend=0.70,
            style_min_confidence=0.70,
            benchmark_top1=benchmark("FEET", "top1"),
            benchmark_top10=benchmark("FEET", "top10"),
            benchmark_scope=(
                "strict three-fold pooled DINOv2+SigLIP2 rank ensemble "
                "with query-crop RRF(k=1), after primary visual-piece "
                f"metadata cleanup ({eligible('FEET')} images); the confidence-gated "
                "footwear-height reranker is reported separately"
            ),
        ),
    }


def base_configs() -> dict[str, SlotInferenceConfig]:
    return {
        slot: SlotInferenceConfig(
            slot=slot,
            index_dir=config.index_dir,
            query_checkpoint=None,
            fusion=config.fusion,
            rrf_k=config.rrf_k,
            candidate_crop_routes=config.candidate_crop_routes,
            gender_routing=config.gender_routing,
            no_equipment_model_id=config.no_equipment_model_id,
            no_equipment_query_crop_names=config.no_equipment_query_crop_names,
            no_equipment_min_probability=config.no_equipment_min_probability,
            benchmark_scope="base DINOv2 fallback; no promoted accuracy benchmark",
        )
        for slot, config in current_best_configs().items()
    }


def validate_input_image(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Input is not a readable image: {path}") from exc
    if width < 200 or height < 200:
        raise ValueError(
            f"Input image is too small ({width}x{height}); use at least 200x200 pixels"
        )
    return {
        "path": portable_path(path),
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
    }


def inspect_assets(
    configs: Mapping[str, SlotInferenceConfig],
) -> dict[str, Any]:
    missing: list[str] = []
    invalid: list[str] = []
    downloadable_models: set[str] = set()
    cached_models: set[str] = set()
    slot_details: dict[str, Any] = {}

    if not APPEARANCE_EQUIVALENCE_PATH.is_file():
        missing.append(str(APPEARANCE_EQUIVALENCE_PATH))

    def inspect_model(model_id: str) -> None:
        if model_weight_files(model_id):
            cached_models.add(model_id)
        elif is_remote_model_id(model_id):
            downloadable_models.add(model_id)
        else:
            missing.append(f"model weights for {model_id}")

    def inspect_index(
        slot: str,
        label: str,
        index_dir: Path,
        detail: dict[str, Any],
    ) -> None:
        metadata_path = index_dir / "metadata.jsonl"
        embeddings_path = index_dir / "embeddings.npy"
        if not metadata_path.is_file():
            missing.append(str(metadata_path))
        if not embeddings_path.is_file():
            missing.append(str(embeddings_path))
        if not metadata_path.is_file() or not embeddings_path.is_file():
            return
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata_count = sum(1 for line in handle if line.strip())
            embeddings = np.load(embeddings_path, mmap_mode="r")
            detail[f"{label}_metadata_count"] = metadata_count
            detail[f"{label}_embedding_shape"] = list(embeddings.shape)
            if label == "primary":
                detail["metadata_count"] = metadata_count
                detail["embedding_shape"] = list(embeddings.shape)
            if embeddings.ndim != 2 or embeddings.shape[0] != metadata_count:
                invalid.append(
                    f"{slot}/{label}: metadata rows ({metadata_count}) and "
                    f"embeddings ({tuple(embeddings.shape)}) do not align"
                )
        except (OSError, ValueError) as exc:
            invalid.append(f"{slot}/{label}: cannot read index: {exc}")

    for slot, config in configs.items():
        if len(config.ensemble_model_ids) != len(config.ensemble_index_dirs):
            invalid.append(f"{slot}: ensemble model/index counts do not match")
        if config.ensemble_query_checkpoints and len(
            config.ensemble_query_checkpoints
        ) != len(config.ensemble_index_dirs):
            invalid.append(
                f"{slot}: ensemble checkpoint/index counts do not match"
            )
        if (
            config.ensemble_final_pool_branches is not None
            and not 1
            <= config.ensemble_final_pool_branches
            <= 1 + len(config.ensemble_index_dirs)
        ):
            invalid.append(f"{slot}: invalid final ensemble pool branch count")
        if config.ensemble_fusion_order not in {
            "branch_then_crop",
            "crop_then_branch",
        }:
            invalid.append(f"{slot}: invalid ensemble fusion order")
        if (
            config.ensemble_fusion_order == "crop_then_branch"
            and config.ensemble_pool_branches is not None
        ):
            invalid.append(
                f"{slot}: crop-then-branch fusion does not support an early pool"
            )
        branch_count = 1 + len(config.ensemble_index_dirs)
        for label, weights in (
            ("ensemble", config.ensemble_branch_weights),
            ("ensemble gate baseline", config.ensemble_gate_baseline_branch_weights),
        ):
            if weights and len(weights) != branch_count:
                invalid.append(f"{slot}: {label} weights do not match branch count")
            if weights and (any(weight < 0 for weight in weights) or not any(weights)):
                invalid.append(f"{slot}: invalid {label} weights")
        gate_values = (
            config.ensemble_top1_gate_max_base_rank,
            config.ensemble_top1_gate_max_last_branch_rank,
            config.ensemble_top1_gate_max_base_last_branch_rank,
            config.ensemble_top1_gate_min_last_branch_rank_ratio,
        )
        if any(value is not None for value in gate_values):
            if not all(value is not None for value in gate_values):
                invalid.append(f"{slot}: incomplete ensemble Top-1 gate settings")
            if not config.ensemble_gate_baseline_branch_weights:
                invalid.append(f"{slot}: ensemble Top-1 gate has no baseline weights")
            if config.ensemble_fusion_order != "crop_then_branch":
                invalid.append(f"{slot}: ensemble Top-1 gate requires crop-then-branch")
            if any(float(value or 0) < 1 for value in gate_values):
                invalid.append(f"{slot}: invalid ensemble Top-1 gate threshold")
        occupied_gate_values = (
            config.occupied_slot_multiplier,
            config.occupied_slot_gate_max_base_rank,
        )
        if (occupied_gate_values[0] is None) != (occupied_gate_values[1] is None):
            invalid.append(f"{slot}: occupied-slot gate settings must be paired")
        if (
            config.occupied_slot_multiplier is not None
            and not 0.0 <= config.occupied_slot_multiplier <= 1.0
        ):
            invalid.append(f"{slot}: invalid occupied-slot multiplier")
        if (
            config.occupied_slot_gate_max_base_rank is not None
            and config.occupied_slot_gate_max_base_rank < 1
        ):
            invalid.append(f"{slot}: invalid occupied-slot gate base rank")
        if (
            config.occupied_slot_body_support_top_k is not None
            and config.occupied_slot_body_support_top_k < 1
        ):
            invalid.append(f"{slot}: invalid occupied-slot BODY support rank")
        if (config.style_model_id is None) != (
            config.style_query_crop_name is None
        ):
            invalid.append(f"{slot}: style model and query crop must be paired")
        if config.style_model_id is not None:
            if config.style_rerank_k < 2:
                invalid.append(f"{slot}: style rerank pool must contain at least two")
            if not 0.0 <= config.style_blend <= 1.0:
                invalid.append(f"{slot}: invalid style blend")
            if not 0.0 <= config.style_min_confidence <= 1.0:
                invalid.append(f"{slot}: invalid style confidence threshold")
        if config.no_equipment_model_id is not None:
            if not config.no_equipment_query_crop_names:
                invalid.append(f"{slot}: no-equipment gate has no query crops")
            if not 0.0 <= config.no_equipment_min_probability <= 1.0:
                invalid.append(f"{slot}: invalid no-equipment threshold")
        elif config.no_equipment_query_crop_names:
            invalid.append(f"{slot}: no-equipment crops require a model")
        detail: dict[str, Any] = {
            "index_dir": portable_path(config.index_dir),
            "query_checkpoint": portable_path(config.query_checkpoint),
            "reranker_checkpoint": portable_path(config.reranker_checkpoint),
            "query_crop_names": config.query_crop_names,
            "ensemble_model_ids": list(config.ensemble_model_ids),
            "ensemble_index_dirs": [
                portable_path(index_dir)
                for index_dir in config.ensemble_index_dirs
            ],
            "ensemble_query_checkpoints": [
                portable_path(checkpoint)
                for checkpoint in config.ensemble_query_checkpoints
            ],
            "morphology_checkpoint": portable_path(config.morphology_checkpoint),
            "ensemble_pool_branches": config.ensemble_pool_branches,
            "ensemble_final_pool_branches": config.ensemble_final_pool_branches,
            "ensemble_fusion_order": config.ensemble_fusion_order,
            "ensemble_branch_weights": list(config.ensemble_branch_weights),
            "ensemble_gate_baseline_branch_weights": list(
                config.ensemble_gate_baseline_branch_weights
            ),
            "ensemble_top1_gate": {
                "max_base_rank": config.ensemble_top1_gate_max_base_rank,
                "max_last_branch_rank": (
                    config.ensemble_top1_gate_max_last_branch_rank
                ),
                "max_base_last_branch_rank": (
                    config.ensemble_top1_gate_max_base_last_branch_rank
                ),
                "min_last_branch_rank_ratio": (
                    config.ensemble_top1_gate_min_last_branch_rank_ratio
                ),
            },
            "occupied_slot_multiplier": config.occupied_slot_multiplier,
            "occupied_slot_gate_max_base_rank": (
                config.occupied_slot_gate_max_base_rank
            ),
            "occupied_slot_body_support_top_k": (
                config.occupied_slot_body_support_top_k
            ),
            "spatial_index": portable_path(config.spatial_index),
            "spatial_query_crop_names": config.spatial_query_crop_names,
            "spatial_rerank_k": (
                config.spatial_rerank_k if config.spatial_index is not None else None
            ),
            "spatial_blend": (
                config.spatial_blend if config.spatial_index is not None else None
            ),
            "style_model_id": config.style_model_id,
            "style_query_crop_name": config.style_query_crop_name,
            "style_rerank_k": (
                config.style_rerank_k
                if config.style_model_id is not None
                else None
            ),
            "style_blend": (
                config.style_blend if config.style_model_id is not None else None
            ),
            "style_min_confidence": (
                config.style_min_confidence
                if config.style_model_id is not None
                else None
            ),
            "no_equipment_model_id": config.no_equipment_model_id,
            "no_equipment_query_crop_names": list(
                config.no_equipment_query_crop_names
            ),
            "no_equipment_min_probability": (
                config.no_equipment_min_probability
                if config.no_equipment_model_id is not None
                else None
            ),
        }
        inspect_index(slot, "primary", config.index_dir, detail)
        for offset, index_dir in enumerate(config.ensemble_index_dirs, start=1):
            inspect_index(slot, f"ensemble_{offset}", index_dir, detail)
        inspect_model("facebook/dinov2-base")
        for model_id in config.ensemble_model_ids:
            inspect_model(model_id)
        if config.style_model_id is not None:
            inspect_model(config.style_model_id)
        if config.no_equipment_model_id is not None:
            inspect_model(config.no_equipment_model_id)
        for checkpoint in (config.query_checkpoint, config.reranker_checkpoint):
            if checkpoint is not None and not checkpoint.is_file():
                missing.append(str(checkpoint))
        for checkpoint in config.ensemble_query_checkpoints:
            if checkpoint is not None and not checkpoint.is_file():
                missing.append(str(checkpoint))
        if (
            config.morphology_checkpoint is not None
            and not config.morphology_checkpoint.is_file()
        ):
            missing.append(str(config.morphology_checkpoint))
        if config.spatial_index is not None and not config.spatial_index.is_file():
            missing.append(str(config.spatial_index))
        slot_details[slot] = detail
    return {
        "ok": not missing and not invalid,
        "missing": sorted(set(missing)),
        "invalid": sorted(set(invalid)),
        "cached_models": sorted(cached_models),
        "downloadable_models": sorted(downloadable_models),
        "slots": slot_details,
    }


def crop_records(
    regions: Sequence[CropRegion],
    paths: Sequence[Path],
    *,
    example_id: str = "query",
) -> dict[str, list[dict[str, Any]]]:
    if len(regions) != len(paths):
        raise ValueError("Every crop region must have exactly one saved crop path")
    return {
        example_id: [
            {
                "example_id": example_id,
                "slot": region.slot,
                "name": region.name,
                "path": path,
                "visibility": region.visibility,
                "confidence": float(region.confidence),
                "role": region.role,
                "tags": tuple(region.tags),
            }
            for region, path in zip(regions, paths, strict=True)
        ]
    }


def filter_records_for_slot(
    records: Mapping[str, Sequence[dict[str, Any]]],
    slot: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        example_id: [row for row in rows if row.get("slot") == slot]
        for example_id, rows in records.items()
    }


def crop_weight(record: Mapping[str, Any]) -> float:
    weight = VISIBLE_WEIGHTS.get(str(record.get("visibility")), 0.0)
    weight *= ROLE_WEIGHTS.get(str(record.get("role", "primary")), 0.0)
    weight *= max(0.35, min(1.0, float(record.get("confidence", 0.0))))
    weight *= max(0.0, float(record.get("style_weight_multiplier", 1.0)))
    return weight


def fuse_branch_search_results(
    searches: Sequence[Mapping[str, Sequence[dict[str, Any]]]],
    *,
    top_k: int,
    rrf_k: int = 1,
    candidate_pool: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    branch_weights: Sequence[float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not searches:
        raise ValueError("At least one retrieval branch is required")
    if top_k < 1:
        raise ValueError("top-k must be positive")
    if rrf_k < 1:
        raise ValueError("rrf-k must be positive")
    paths = set(searches[0])
    if any(set(search) != paths for search in searches[1:]):
        raise ValueError("Retrieval branches contain different crop paths")
    if candidate_pool is not None and set(candidate_pool) != paths:
        raise ValueError("Candidate pool contains different crop paths")
    weights = (
        tuple(float(weight) for weight in branch_weights)
        if branch_weights is not None
        else (1.0,) * len(searches)
    )
    if len(weights) != len(searches):
        raise ValueError("Branch weights do not match retrieval branches")
    if any(weight < 0.0 for weight in weights) or not any(weights):
        raise ValueError("Branch weights must be non-negative with a positive sum")

    fused: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(paths):
        scores: dict[int, float] = {}
        exemplar: dict[int, dict[str, Any]] = {}
        allowed_ids = (
            {int(row["piece_id"]) for row in candidate_pool[path]}
            if candidate_pool is not None
            else None
        )
        for search, branch_weight in zip(searches, weights, strict=True):
            if branch_weight == 0.0:
                continue
            for offset, result in enumerate(search[path], start=1):
                piece_id = int(result["piece_id"])
                if allowed_ids is not None and piece_id not in allowed_ids:
                    continue
                result_rank = int(result.get("rank", offset))
                scores[piece_id] = scores.get(piece_id, 0.0) + branch_weight / (
                    rrf_k + result_rank
                )
                exemplar.setdefault(piece_id, dict(result))
        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        fused[path] = [
            {
                **exemplar[piece_id],
                "rank": rank,
                "score": scores[piece_id],
            }
            for rank, piece_id in enumerate(ranked_ids, start=1)
        ]
    return fused


def constrain_candidates_to_pool(
    pool_candidates: Sequence[Mapping[str, Any]],
    reranked_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pool_by_id = {
        int(row["piece_id"]): dict(row) for row in pool_candidates
    }
    output = [
        dict(row)
        for row in reranked_candidates
        if int(row["piece_id"]) in pool_by_id
    ]
    present = {int(row["piece_id"]) for row in output}
    output.extend(
        row for piece_id, row in pool_by_id.items() if piece_id not in present
    )
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def attach_appearance_equivalent_candidates(
    candidates: Sequence[Mapping[str, Any]],
    direct_equivalents: Mapping[int, Sequence[int]],
    *,
    piece_catalog: Mapping[int, Mapping[str, Any]] | None = None,
    max_equivalents_per_anchor: int | None = 1,
) -> list[dict[str, Any]]:
    """Attach same-model alternatives without changing candidate order."""

    if (
        max_equivalents_per_anchor is not None
        and max_equivalents_per_anchor < 0
    ):
        raise ValueError("max_equivalents_per_anchor must be non-negative")
    source = [dict(row) for row in candidates]
    by_piece = {int(row["piece_id"]): row for row in source}
    catalog = piece_catalog or {}
    for anchor in source:
        anchor_id = int(anchor["piece_id"])
        if max_equivalents_per_anchor == 0:
            continue
        alternatives = []
        attached_ids: set[int] = set()
        for equivalent_id in direct_equivalents.get(anchor_id, ()):
            parsed_id = int(equivalent_id)
            if parsed_id == anchor_id or parsed_id in attached_ids:
                continue
            ranked_equivalent = by_piece.get(parsed_id)
            equivalent = ranked_equivalent or catalog.get(parsed_id)
            if equivalent is None:
                continue
            name_en = equivalent.get("name_en") or equivalent.get("name")
            name_zh = equivalent.get("name_zh")
            if not name_en and not name_zh:
                continue
            alternative = {
                "piece_id": parsed_id,
                "name_en": name_en,
                "name_zh": name_zh,
                "relation": "same_model",
            }
            if ranked_equivalent is not None:
                alternative.update(
                    {
                        "original_rank": int(ranked_equivalent.get("rank", 0)),
                        "score": float(ranked_equivalent.get("score", 0.0)),
                        "source_image_url": ranked_equivalent.get("source_image_url"),
                        "local_preview": ranked_equivalent.get("local_preview"),
                    }
                )
            alternatives.append(alternative)
            attached_ids.add(parsed_id)
            if (
                max_equivalents_per_anchor is not None
                and len(alternatives) >= max_equivalents_per_anchor
            ):
                break
        if alternatives:
            anchor["appearance_alternatives"] = alternatives
            anchor["appearance_equivalence"] = {
                "piece_id": anchor_id,
                "name_en": anchor.get("name_en"),
                "relation": "same_model",
            }
    return source


def collapse_appearance_equivalent_candidates(
    candidates: Sequence[Mapping[str, Any]],
    direct_equivalents: Mapping[int, Sequence[int]],
) -> list[dict[str, Any]]:
    """Keep only the highest-ranked candidate from each same-model group.

    Equivalence data is treated as an undirected, transitive relationship so
    incomplete one-way links cannot leave duplicate appearances in the final
    Top-K.  Candidate order is otherwise stable, and ranks are compacted after
    duplicate rows are removed.
    """

    parent: dict[int, int] = {}

    def find(piece_id: int) -> int:
        parent.setdefault(piece_id, piece_id)
        root = piece_id
        while parent[root] != root:
            root = parent[root]
        while parent[piece_id] != piece_id:
            next_piece = parent[piece_id]
            parent[piece_id] = root
            piece_id = next_piece
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for raw_piece_id, raw_equivalents in direct_equivalents.items():
        piece_id = int(raw_piece_id)
        find(piece_id)
        for raw_equivalent_id in raw_equivalents:
            union(piece_id, int(raw_equivalent_id))

    output: list[dict[str, Any]] = []
    seen_groups: set[int] = set()
    for candidate in candidates:
        copied = dict(candidate)
        group = find(int(copied["piece_id"]))
        if group in seen_groups:
            continue
        seen_groups.add(group)
        output.append(copied)

    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def select_slot_candidate_index_rows(
    metadata: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    slot: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Select one slot without breaking metadata/embedding row alignment."""

    if embeddings.ndim != 2 or embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Candidate index row mismatch: "
            f"embeddings={embeddings.shape}, metadata_rows={len(metadata)}"
        )
    indices = [
        index
        for index, row in enumerate(metadata)
        if str(row.get("slot")) == slot
    ]
    if not indices:
        raise ValueError(f"Candidate index contains no {slot} rows")
    selected_metadata = [metadata[index] for index in indices]
    if len(indices) == len(metadata):
        return selected_metadata, embeddings
    return selected_metadata, np.asarray(embeddings[indices], dtype=np.float32)
def fuse_crops_then_branch_results(
    records: Sequence[dict[str, Any]],
    branch_searches: Sequence[Mapping[str, Sequence[dict[str, Any]]]],
    *,
    top_k: int,
    candidate_k: int,
    crop_fusion: str,
    crop_rrf_k: int,
    branch_rrf_k: int,
    final_pool_branches: int | None = None,
    branch_weights: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Preserve raw crop-score fusion inside each model before branch RRF."""

    if not branch_searches:
        raise ValueError("At least one branch search is required")
    if final_pool_branches is not None and not (
        1 <= final_pool_branches <= len(branch_searches)
    ):
        raise ValueError("Invalid final_pool_branches")
    if branch_weights is not None and len(branch_weights) != len(branch_searches):
        raise ValueError("Branch weights do not match retrieval branches")
    branch_candidates = [
        fuse_crop_results(
            records,
            search,
            top_k=candidate_k,
            fusion=crop_fusion,
            rrf_k=crop_rrf_k,
        )
        for search in branch_searches
    ]

    def fuse(
        candidate_lists: Sequence[Sequence[dict[str, Any]]],
        weights: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        wrapped = [
            {"query": list(candidates)} for candidates in candidate_lists
        ]
        return fuse_branch_search_results(
            wrapped,
            top_k=candidate_k,
            rrf_k=branch_rrf_k,
            branch_weights=weights,
        )["query"]

    candidates = fuse(branch_candidates, branch_weights)
    if final_pool_branches is not None:
        pool = fuse(branch_candidates[:final_pool_branches])
        candidates = constrain_candidates_to_pool(pool, candidates)
    return candidates[:top_k]


def apply_ensemble_top1_gate(
    base_candidates: Sequence[Mapping[str, Any]],
    candidate_candidates: Sequence[Mapping[str, Any]],
    last_branch_candidates: Sequence[Mapping[str, Any]],
    *,
    max_base_rank: int,
    max_last_branch_rank: int,
    max_base_last_branch_rank: int,
    min_last_branch_rank_ratio: float,
) -> list[dict[str, Any]]:
    """Promote a branch-supported winner while preserving the baseline tail."""

    if min(
        max_base_rank,
        max_last_branch_rank,
        max_base_last_branch_rank,
    ) < 1:
        raise ValueError("Ensemble Top-1 gate rank thresholds must be positive")
    if min_last_branch_rank_ratio < 1.0:
        raise ValueError("Ensemble Top-1 gate rank ratio must be at least one")

    baseline = [dict(row) for row in base_candidates]
    if not baseline or not candidate_candidates:
        return baseline
    base_top1_id = int(baseline[0]["piece_id"])
    candidate_top1_id = int(candidate_candidates[0]["piece_id"])
    if base_top1_id == candidate_top1_id:
        return baseline

    def piece_rank(rows: Sequence[Mapping[str, Any]], piece_id: int) -> int | None:
        return next(
            (
                offset
                for offset, row in enumerate(rows, start=1)
                if int(row["piece_id"]) == piece_id
            ),
            None,
        )

    candidate_base_rank = piece_rank(baseline, candidate_top1_id)
    candidate_last_rank = piece_rank(last_branch_candidates, candidate_top1_id)
    base_last_rank = piece_rank(last_branch_candidates, base_top1_id)
    ratio_ready = bool(
        candidate_last_rank is not None
        and (
            base_last_rank is None
            or base_last_rank
            >= candidate_last_rank * min_last_branch_rank_ratio
        )
    )
    accepted = bool(
        candidate_base_rank is not None
        and candidate_base_rank <= max_base_rank
        and candidate_last_rank is not None
        and candidate_last_rank <= max_last_branch_rank
        and base_last_rank is not None
        and base_last_rank <= max_base_last_branch_rank
        and ratio_ready
    )
    if not accepted:
        return baseline

    promoted = dict(candidate_candidates[0])
    promoted["ensemble_top1_gate"] = {
        "promoted": True,
        "base_rank": candidate_base_rank,
        "candidate_last_branch_rank": candidate_last_rank,
        "base_last_branch_rank": base_last_rank,
        "demoted_piece_id": base_top1_id,
    }
    output = [promoted]
    output.extend(
        row for row in baseline if int(row["piece_id"]) != candidate_top1_id
    )
    output = output[: len(baseline)]
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def fuse_crop_results(
    records: Sequence[dict[str, Any]],
    search_results_by_crop: Mapping[str, Sequence[dict[str, Any]]],
    *,
    top_k: int,
    fusion: str = "max",
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top-k must be positive")
    if fusion not in {"max", "rrf"}:
        raise ValueError(f"Unsupported fusion: {fusion}")
    if rrf_k < 1:
        raise ValueError("rrf-k must be positive")

    candidates: dict[int, dict[str, Any]] = {}
    for record in records:
        weight = crop_weight(record)
        if weight <= 0.0:
            continue
        for offset, result in enumerate(
            search_results_by_crop.get(str(record["path"]), []),
            start=1,
        ):
            piece_id = int(result["piece_id"])
            result_rank = int(result.get("rank", offset))
            raw_score = float(result["score"])
            contribution = (
                weight / (rrf_k + result_rank)
                if fusion == "rrf"
                else weight * raw_score
            )
            row = candidates.setdefault(
                piece_id,
                {
                    "piece_id": piece_id,
                    "score": 0.0 if fusion == "rrf" else -float("inf"),
                    "best_result": result,
                    "best_contribution": -float("inf"),
                    "evidence": [],
                },
            )
            if fusion == "rrf":
                row["score"] += contribution
            else:
                row["score"] = max(float(row["score"]), contribution)
            if contribution > float(row["best_contribution"]):
                row["best_contribution"] = contribution
                row["best_result"] = result
            row["evidence"].append(
                {
                    "crop": record.get("name"),
                    "visibility": record.get("visibility"),
                    "weight": weight,
                    "candidate_rank": result_rank,
                    "raw_score": raw_score,
                    "contribution": contribution,
                }
            )

    ranked = sorted(
        candidates.values(),
        key=lambda row: float(row["score"]),
        reverse=True,
    )[:top_k]
    output = []
    for rank, row in enumerate(ranked, start=1):
        best = row["best_result"]
        evidence = sorted(
            row["evidence"],
            key=lambda item: float(item["contribution"]),
            reverse=True,
        )[:5]
        output.append(
            {
                "rank": rank,
                "piece_id": row["piece_id"],
                "name_en": best.get("name_en"),
                "name_zh": best.get("name_zh"),
                "gearset_slug": best.get("gearset_slug"),
                "score": float(row["score"]),
                "source_image_url": best.get("source_image_url"),
                "local_preview": portable_path(best.get("local_path")),
                "model_gender": best.get("model_gender"),
                "race": best.get("race"),
                "dye_state": best.get("dye_state"),
                "link_type": best.get("link_type"),
                "evidence": evidence,
            }
        )
    return output


def rerank_occupied_slot_search_results(
    search_results_by_crop: Mapping[str, Sequence[dict[str, Any]]],
    *,
    occupied_multiplier: float,
    rrf_k: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    if not 0.0 <= occupied_multiplier <= 1.0:
        raise ValueError("occupied_multiplier must be between zero and one")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    output = {}
    for path, rows in search_results_by_crop.items():
        rescored = []
        for offset, row in enumerate(rows, start=1):
            rank = int(row.get("rank", offset))
            multiplier = (
                occupied_multiplier
                if str(row.get("link_type")) == "occupied_slot"
                else 1.0
            )
            rescored.append(
                (
                    multiplier / (rrf_k + rank),
                    offset,
                    row,
                )
            )
        ranked = sorted(rescored, key=lambda item: (-item[0], item[1]))
        output[path] = [
            {
                **dict(row),
                "rank": rank,
                "occupied_slot_adjusted_score": float(score),
            }
            for rank, (score, _, row) in enumerate(ranked, start=1)
        ]
    return output


def apply_occupied_slot_top1_gate(
    base_candidates: Sequence[Mapping[str, Any]],
    adjusted_candidates: Sequence[Mapping[str, Any]],
    *,
    max_base_rank: int,
) -> list[dict[str, Any]]:
    if max_base_rank < 1:
        raise ValueError("max_base_rank must be positive")
    output = [dict(row) for row in base_candidates]
    if not output or not adjusted_candidates:
        return output
    base_top1 = output[0]
    candidate_top1 = adjusted_candidates[0]
    if (
        str(base_top1.get("link_type")) != "occupied_slot"
        or str(candidate_top1.get("link_type")) != "exact_slot"
        or int(base_top1["piece_id"]) == int(candidate_top1["piece_id"])
    ):
        return output
    candidate_id = int(candidate_top1["piece_id"])
    candidate_index = next(
        (
            index
            for index, row in enumerate(output)
            if int(row["piece_id"]) == candidate_id
        ),
        None,
    )
    if candidate_index is None or candidate_index + 1 > max_base_rank:
        return output
    promoted = output.pop(candidate_index)
    promoted["occupied_slot_gate"] = {
        "promoted": True,
        "base_rank": candidate_index + 1,
        "demoted_piece_id": int(base_top1["piece_id"]),
    }
    output.insert(0, promoted)
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def restore_body_supported_occupied_hands_top1(
    slot_results: Sequence[Mapping[str, Any]],
    *,
    body_support_top_k: int,
) -> list[dict[str, Any]]:
    if body_support_top_k < 1:
        raise ValueError("body_support_top_k must be positive")
    output = [
        {
            **dict(result),
            "candidates": [dict(row) for row in result.get("candidates", [])],
        }
        for result in slot_results
    ]
    by_slot = {str(result.get("slot")): result for result in output}
    body = by_slot.get("BODY")
    hands = by_slot.get("HANDS")
    if body is None or hands is None or not hands["candidates"]:
        return output
    supported_names = {
        str(row.get("name_en") or "").strip().casefold()
        for row in body.get("candidates", [])[:body_support_top_k]
        if row.get("name_en")
    }
    promoted = hands["candidates"][0]
    gate = promoted.get("occupied_slot_gate")
    if not isinstance(gate, Mapping) or not gate.get("promoted"):
        return output
    demoted_piece_id = int(gate["demoted_piece_id"])
    demoted_index = next(
        (
            index
            for index, row in enumerate(hands["candidates"])
            if int(row["piece_id"]) == demoted_piece_id
        ),
        None,
    )
    if demoted_index is None:
        return output
    demoted = hands["candidates"][demoted_index]
    demoted_name = str(demoted.get("name_en") or "").strip().casefold()
    if not demoted_name or demoted_name not in supported_names:
        return output

    restored = hands["candidates"].pop(demoted_index)
    restored["cross_slot_consistency"] = {
        "restored": True,
        "body_support_top_k": body_support_top_k,
        "matched_name_en": restored.get("name_en"),
    }
    gate_copy = dict(gate)
    gate_copy["cross_slot_restored"] = True
    promoted["occupied_slot_gate"] = gate_copy
    hands["candidates"].insert(0, restored)
    for rank, row in enumerate(hands["candidates"], start=1):
        row["rank"] = rank
    scores = [float(row["score"]) for row in hands["candidates"]]
    hands["top_score_margin"] = (
        scores[0] - scores[1] if len(scores) >= 2 else None
    )
    reference = hands.get("reference")
    if isinstance(reference, dict):
        reference_rank = next(
            (
                int(row["rank"])
                for row in hands["candidates"]
                if row.get("reference_match")
            ),
            None,
        )
        reference["rank"] = reference_rank
        reference["top1_hit"] = reference_rank == 1
        reference["displayed_top_k_hit"] = reference_rank is not None
    return output


def no_visible_equipment_decision(
    probabilities_by_crop: Sequence[Sequence[float]],
    *,
    minimum_probability: float,
) -> dict[str, Any]:
    """Conservatively decide whether every inspected crop looks unequipped.

    Each row is ``(no visible equipment, visible equipment)``. Requiring the
    least-confident crop to clear the threshold prevents one bare or occluded
    hand from hiding equipment visible on the other side.
    """

    if not 0.0 <= minimum_probability <= 1.0:
        raise ValueError("minimum_probability must be between zero and one")
    parsed: list[tuple[float, float]] = []
    for raw_probabilities in probabilities_by_crop:
        probabilities = tuple(float(value) for value in raw_probabilities)
        if len(probabilities) != 2:
            raise ValueError("no-equipment probabilities must contain two classes")
        if any(
            not np.isfinite(value) or value < 0.0 or value > 1.0
            for value in probabilities
        ):
            raise ValueError(
                "no-equipment probabilities must be finite and between zero and one"
            )
        total = sum(probabilities)
        if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-5):
            raise ValueError("no-equipment probabilities must sum to one")
        parsed.append((probabilities[0] / total, probabilities[1] / total))

    absence_probabilities = [row[0] for row in parsed]
    minimum_absence_probability = (
        min(absence_probabilities) if absence_probabilities else None
    )
    return {
        "active": bool(parsed),
        "no_visible_equipment": (
            minimum_absence_probability is not None
            and minimum_absence_probability >= minimum_probability
        ),
        "minimum_probability": minimum_probability,
        "minimum_no_equipment_probability": minimum_absence_probability,
        "probabilities": [
            {
                "no_visible_equipment": no_equipment,
                "visible_equipment": visible_equipment,
            }
            for no_equipment, visible_equipment in parsed
        ],
    }


def slot_region_presence_decision(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Decide whether crop generation found any usable view of a slot.

    This is deliberately separate from the no-equipment classifier. A bare,
    visible hand is present and can support an invisible-equipment candidate;
    a hand inferred outside a half-body source image is not present and should
    not produce any candidate at all.
    """

    evidence = [
        {
            "name": str(row.get("name") or ""),
            "visibility": str(row.get("visibility") or ""),
            "confidence": float(row.get("confidence", 0.0)),
            "role": str(row.get("role") or "primary"),
        }
        for row in records
        if str(row.get("role") or "primary") != "diagnostic"
    ]
    usable_names = [
        str(row.get("name") or "")
        for row in records
        if crop_weight(row) > 0.0
    ]
    contains_slot = bool(usable_names)
    if contains_slot:
        reason = "usable_slot_crop_available"
    elif evidence:
        reason = "slot_crops_out_of_frame_or_pose_uncertain"
    else:
        reason = "no_slot_crops"
    return {
        "active": True,
        "contains_slot": contains_slot,
        "reason": reason,
        "usable_crops": usable_names,
        "evidence": evidence,
    }


def prepend_no_equipment_candidate(
    candidates: Sequence[Mapping[str, Any]],
    catalog_item: Mapping[str, Any] | None,
    diagnostic: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Represent a confident absence decision with an Emperor-series item."""

    output = [dict(row) for row in candidates]
    if not diagnostic or not diagnostic.get("no_visible_equipment"):
        return output
    if catalog_item is None:
        return output

    game_item_id = int(catalog_item["game_item_id"])
    if game_item_id <= 0:
        raise ValueError("no-equipment game item id must be positive")
    no_equipment_score = float(
        diagnostic.get("minimum_no_equipment_probability") or 0.0
    )
    no_equipment = {
        "piece_id": -game_item_id,
        "game_item_id": game_item_id,
        "synthetic_piece_id": True,
        "candidate_type": "no_equipment",
        "is_no_equipment": True,
        "slot": str(catalog_item.get("slot") or ""),
        "name_en": str(catalog_item.get("name_en") or ""),
        "name_zh": str(catalog_item.get("name_zh") or ""),
        "score": no_equipment_score,
        "no_equipment_score": no_equipment_score,
        "score_source": "no_equipment_prompt_classifier",
        "source_item_url": catalog_item.get("source_item_url")
        or catalog_item.get("source"),
        "local_preview": None,
        "source_image_url": None,
        "occupied_slots": [str(catalog_item.get("slot") or "")],
        "recommendation_reliable": True,
        "reference_match": False,
    }
    ranked = [no_equipment, *output]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def apply_multislot_consistency_gate(
    slot_results: Sequence[Mapping[str, Any]],
    occupied_slots_by_piece: Mapping[int, Sequence[str]],
    *,
    support_top_k: int,
) -> list[dict[str, Any]]:
    """Demote multi-slot candidates unsupported by their other visible slots."""

    if support_top_k < 1:
        raise ValueError("support_top_k must be positive")
    output = [
        {
            **dict(result),
            "candidates": [dict(row) for row in result.get("candidates", [])],
        }
        for result in slot_results
    ]
    by_slot = {str(result.get("slot")): result for result in output}
    original_candidates = {
        slot: [dict(row) for row in result.get("candidates", [])]
        for slot, result in by_slot.items()
    }

    def normalized_name(row: Mapping[str, Any]) -> str:
        return " ".join(str(row.get("name_en") or "").casefold().split())

    for slot, result in by_slot.items():
        unsupported_count = 0
        for original_index, candidate in enumerate(result["candidates"]):
            piece_id = int(candidate["piece_id"])
            occupied_slots = tuple(
                occupied_slot
                for occupied_slot in (
                    str(value).upper()
                    for value in occupied_slots_by_piece.get(piece_id, ())
                )
                if occupied_slot in SLOTS
            )
            if len(set(occupied_slots)) <= 1:
                continue

            candidate_name = normalized_name(candidate)
            required_slots = [
                occupied_slot
                for occupied_slot in dict.fromkeys(occupied_slots)
                if occupied_slot != slot and occupied_slot in by_slot
            ]
            support: dict[str, dict[str, Any]] = {}
            unsupported_slots: list[str] = []
            for occupied_slot in required_slots:
                matched = next(
                    (
                        row
                        for row in original_candidates[occupied_slot]
                        if int(row.get("piece_id", -1)) == piece_id
                        or (
                            candidate_name
                            and normalized_name(row) == candidate_name
                        )
                    ),
                    None,
                )
                matched_rank = (
                    int(matched.get("rank", 0)) if matched is not None else None
                )
                supported = (
                    matched_rank is not None
                    and 1 <= matched_rank <= support_top_k
                )
                support[occupied_slot] = {
                    "rank": matched_rank,
                    "supported": supported,
                }
                if not supported:
                    unsupported_slots.append(occupied_slot)

            if not required_slots:
                state = "unchecked"
                reliable = True
            elif unsupported_slots:
                state = "unsupported"
                reliable = False
                unsupported_count += 1
            else:
                state = "supported"
                reliable = True
            candidate["recommendation_reliable"] = reliable
            candidate["cross_slot_consistency"] = {
                "state": state,
                "occupied_slots": list(dict.fromkeys(occupied_slots)),
                "checked_slots": required_slots,
                "support_top_k": support_top_k,
                "support": support,
                "unsupported_slots": unsupported_slots,
            }
            candidate["_cross_slot_original_index"] = original_index

        result["candidates"].sort(
            key=lambda row: (
                not bool(row.get("recommendation_reliable", True)),
                int(row.get("_cross_slot_original_index", row.get("rank", 0))),
            )
        )
        reordered = any(
            int(row.get("rank", index)) != index
            for index, row in enumerate(result["candidates"], start=1)
        )
        for rank, row in enumerate(result["candidates"], start=1):
            row["rank"] = rank
            row.pop("_cross_slot_original_index", None)
        result["cross_slot_consistency"] = {
            "support_top_k": support_top_k,
            "unsupported_candidate_count": unsupported_count,
            "ranking_changed": reordered,
        }
        if reordered:
            result["top_score_margin"] = None
        reference = result.get("reference")
        if isinstance(reference, dict):
            reference_rank = next(
                (
                    int(row["rank"])
                    for row in result["candidates"]
                    if row.get("reference_match")
                ),
                None,
            )
            reference["rank"] = reference_rank
            reference["top1_hit"] = reference_rank == 1
            reference["displayed_top_k_hit"] = reference_rank is not None
    return output


def slot_result(
    config: SlotInferenceConfig,
    records: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    usable = [row for row in records if crop_weight(row) > 0.0]
    scores = [float(row["score"]) for row in candidates]
    score_margin = scores[0] - scores[1] if len(scores) >= 2 else None
    if candidates and (
        candidates[0].get("occupied_slot_gate")
        or candidates[0].get("ensemble_top1_gate")
        or candidates[0].get("is_no_equipment")
    ):
        score_margin = None
    quality_reasons = []
    if (
        config.slot == "HEAD"
        and score_margin is not None
        and score_margin < 0.01
    ):
        quality_reasons.append(
            "Top head candidates are nearly tied; hair, ears, or occlusion may dominate the crop."
        )
    if any(
        "source_edge_truncation_risk" in tuple(row.get("tags", ()))
        for row in records
    ):
        if config.slot == "FEET":
            quality_reasons.append(
                "The source image probably cuts through the full shoe area; "
                "footwear ranking may be incomplete."
            )
        elif config.slot == "HEAD":
            quality_reasons.append(
                "The source image probably cuts through the full headwear area; "
                "headwear ranking may be incomplete."
            )
        else:
            quality_reasons.append(
                "The source image probably truncates part of this equipment area; "
                "ranking may be incomplete."
            )
    return {
        "slot": config.slot,
        "status": "ok" if candidates else "no_usable_candidates",
        "method": {
            "index_dir": portable_path(config.index_dir),
            "query_checkpoint": portable_path(config.query_checkpoint),
            "reranker_checkpoint": portable_path(config.reranker_checkpoint),
            "query_crop_names": config.query_crop_names,
            "ensemble_model_ids": list(config.ensemble_model_ids),
            "ensemble_index_dirs": [
                portable_path(index_dir)
                for index_dir in config.ensemble_index_dirs
            ],
            "ensemble_query_checkpoints": [
                portable_path(checkpoint)
                for checkpoint in config.ensemble_query_checkpoints
            ],
            "morphology_checkpoint": portable_path(config.morphology_checkpoint),
            "ensemble_rrf_k": (
                config.ensemble_rrf_k if config.ensemble_index_dirs else None
            ),
            "ensemble_pool_branches": config.ensemble_pool_branches,
            "ensemble_final_pool_branches": config.ensemble_final_pool_branches,
            "ensemble_fusion_order": config.ensemble_fusion_order,
            "ensemble_branch_weights": list(config.ensemble_branch_weights),
            "ensemble_gate_baseline_branch_weights": list(
                config.ensemble_gate_baseline_branch_weights
            ),
            "ensemble_top1_gate": {
                "max_base_rank": config.ensemble_top1_gate_max_base_rank,
                "max_last_branch_rank": (
                    config.ensemble_top1_gate_max_last_branch_rank
                ),
                "max_base_last_branch_rank": (
                    config.ensemble_top1_gate_max_base_last_branch_rank
                ),
                "min_last_branch_rank_ratio": (
                    config.ensemble_top1_gate_min_last_branch_rank_ratio
                ),
            },
            "occupied_slot_multiplier": config.occupied_slot_multiplier,
            "occupied_slot_gate_max_base_rank": (
                config.occupied_slot_gate_max_base_rank
            ),
            "occupied_slot_body_support_top_k": (
                config.occupied_slot_body_support_top_k
            ),
            "ensemble_pool_rrf_k": (
                config.ensemble_pool_rrf_k
                if config.ensemble_pool_branches is not None
                else None
            ),
            "spatial_index": portable_path(config.spatial_index),
            "spatial_query_crop_names": config.spatial_query_crop_names,
            "spatial_rerank_k": (
                config.spatial_rerank_k if config.spatial_index is not None else None
            ),
            "spatial_blend": (
                config.spatial_blend if config.spatial_index is not None else None
            ),
            "style_model_id": config.style_model_id,
            "style_query_crop_name": config.style_query_crop_name,
            "style_rerank_k": (
                config.style_rerank_k
                if config.style_model_id is not None
                else None
            ),
            "style_blend": (
                config.style_blend if config.style_model_id is not None else None
            ),
            "style_min_confidence": (
                config.style_min_confidence
                if config.style_model_id is not None
                else None
            ),
            "no_equipment_model_id": config.no_equipment_model_id,
            "no_equipment_query_crop_names": list(
                config.no_equipment_query_crop_names
            ),
            "no_equipment_min_probability": (
                config.no_equipment_min_probability
                if config.no_equipment_model_id is not None
                else None
            ),
            "fusion": config.fusion,
            "rrf_k": config.rrf_k if config.fusion == "rrf" else None,
        },
        "benchmark": {
            "top1": config.benchmark_top1,
            "top10": config.benchmark_top10,
            "scope": config.benchmark_scope,
            "applies_to_this_image": False,
        },
        "usable_crop_count": len(usable),
        "used_crops": [
            {
                "name": row.get("name"),
                "visibility": row.get("visibility"),
                "confidence": row.get("confidence"),
                "role": row.get("role"),
                "weight": crop_weight(row),
                "style_weight_multiplier": float(
                    row.get("style_weight_multiplier", 1.0)
                ),
            }
            for row in usable
        ],
        "top_score_margin": score_margin,
        "ranking_quality": {
            "level": "low" if quality_reasons else "normal",
            "reasons": quality_reasons,
        },
        "score_is_calibrated_probability": False,
        "candidates": list(candidates),
    }

def write_html_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_path = resolve_project_path(payload.get("input", {}).get("path"))
    source_uri = _file_uri(source_path)
    artifacts = payload.get("artifacts", {})
    overlay_uri = _file_uri(resolve_project_path(artifacts.get("overlay")))
    contact_uri = _file_uri(resolve_project_path(artifacts.get("contact_sheet")))
    slot_sections = "".join(_slot_html(row) for row in payload.get("slots", []))
    warnings = "".join(
        f"<li>{html.escape(str(value))}</li>" for value in payload.get("warnings", [])
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>FFXIV Prism 识别结果</title>
  <style>
    :root {{ color-scheme: dark; --bg:#11151c; --panel:#1a202b; --muted:#9ba7b7; --accent:#76b7ff; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font:15px/1.5 system-ui,sans-serif; background:var(--bg); color:#eef3f8; }}
    main {{ max-width:1280px; margin:auto; padding:28px; }} h1,h2,h3 {{ margin:.2em 0 .6em; }}
    .notice {{ background:#2b2517; border:1px solid #6e5b25; padding:12px 16px; border-radius:10px; }}
    .visuals,.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; }}
    .panel,.card {{ background:var(--panel); border:1px solid #30394a; border-radius:12px; padding:14px; }}
    .card.reference {{ border-color:#48c78e; box-shadow:0 0 0 1px #48c78e inset; }}
    .reference-note {{ background:#153327; border:1px solid #2f8f64; padding:12px 16px; border-radius:10px; }}
    .visuals img {{ width:100%; max-height:480px; object-fit:contain; background:#0b0e13; border-radius:8px; }}
    .card img {{ width:100%; height:220px; object-fit:contain; background:#0b0e13; border-radius:8px; }}
    .same-models {{ margin:-.35em 0 .8em; color:#c8d2de; }}
    .rank {{ color:var(--accent); font-weight:700; }} .muted {{ color:var(--muted); }}
    a {{ color:#8dc6ff; }} code {{ overflow-wrap:anywhere; }}
  </style>
</head>
<body><main>
  <h1>FFXIV Prism 识别结果</h1>
  <p class="notice">候选分数是相对排序信号，不是准确率或概率。请结合裁剪质检图与候选预览人工确认，尤其是 HEAD/HANDS/LEGS/FEET。</p>
  {f'<ul class="notice">{warnings}</ul>' if warnings else ''}
  <section class="visuals">
    {_visual_panel('输入截图', source_uri)}
    {_visual_panel('人体与区域质检', overlay_uri)}
    {_visual_panel('槽位裁剪', contact_uri)}
  </section>
  {slot_sections}
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def _slot_html(result: Mapping[str, Any]) -> str:
    slot = html.escape(str(result.get("slot", "")))
    benchmark = result.get("benchmark", {})
    top1 = benchmark.get("top1")
    top10 = benchmark.get("top10")
    benchmark_text = html.escape(str(benchmark.get("scope", "")))
    metrics = (
        f"Top-1 {float(top1):.1%} / Top-10 {float(top10):.1%}"
        if top1 is not None and top10 is not None
        else "No registered benchmark"
    )
    ranking_quality = result.get("ranking_quality", {})
    quality_reasons = ranking_quality.get("reasons", [])
    quality_note = (
        '<p class="notice">Low-confidence ranking: '
        + html.escape(" ".join(str(value) for value in quality_reasons))
        + " Visual review is recommended.</p>"
        if quality_reasons
        else ""
    )
    reference = result.get("reference")
    reference_note = ""
    if reference:
        reference_name = html.escape(str(reference.get("item_name") or "Unknown"))
        reference_rank = reference.get("rank")
        rank_text = (
            f"current rank #{int(reference_rank)}"
            if reference_rank is not None
            else f"not found in Top-{int(reference.get('candidate_k', 0))} candidate pool"
        )
        hit_text = "Top-1 hit" if reference.get("top1_hit") else "Top-1 miss"
        reference_note = (
            f'<p class="reference-note"><strong>Known ground truth:</strong> '
            f'{reference_name} · {html.escape(rank_text)} · {hit_text}</p>'
        )
    cards = "".join(_candidate_html(row) for row in result.get("candidates", []))
    if not cards:
        cards = '<div class="panel">No usable candidates; check crop visibility.</div>'
    return f"""<section>
  <h2>{slot}</h2>
  <p class="muted">Reference benchmark: {metrics}; {benchmark_text}. This is not per-image confidence.</p>
  {reference_note}
  {quality_note}
  <div class="cards">{cards}</div>
</section>"""


def _candidate_html(candidate: Mapping[str, Any]) -> str:
    preview = _file_uri(resolve_project_path(candidate.get("local_preview")))
    source = str(
        candidate.get("source_image_url")
        or candidate.get("source_item_url")
        or ""
    )
    name_en = html.escape(str(candidate.get("name_en") or "未知装备"))
    name_zh = html.escape(str(candidate.get("name_zh") or ""))
    name = f"{name_zh}<br><span class=\"muted\">{name_en}</span>" if name_zh else name_en
    source_link = (
        f'<a href="{html.escape(source, quote=True)}">来源图片</a>' if source else "无来源链接"
    )
    alternative_names = []
    for row in candidate.get("appearance_alternatives", []):
        alternative_name_zh = html.escape(str(row.get("name_zh") or ""))
        alternative_name_en = html.escape(str(row.get("name_en") or ""))
        if alternative_name_zh and alternative_name_en:
            alternative_names.append(
                f'{alternative_name_zh} <span class="muted">'
                f'({alternative_name_en})</span>'
            )
        elif alternative_name_zh or alternative_name_en:
            alternative_names.append(alternative_name_zh or alternative_name_en)
    alternatives = "、".join(alternative_names)
    alternatives_note = (
        f'<p class="same-models"><strong>同模装备：</strong>{alternatives}</p>'
        if alternatives
        else ""
    )
    reference_class = " reference" if candidate.get("reference_match") else ""
    reference_badge = " · Known ground truth" if candidate.get("reference_match") else ""
    no_equipment = bool(candidate.get("is_no_equipment"))
    no_equipment_badge = (
        '<p class="notice"><strong>未检测到可见装备</strong>：'
        "以皇帝系列的隐形外观表示。</p>"
        if no_equipment
        else ""
    )
    identifier = (
        f"game_item_id={int(candidate.get('game_item_id', 0))}"
        if no_equipment
        else f"piece_id={int(candidate.get('piece_id', 0))}"
    )
    score_label = (
        f"no-equipment score={float(candidate.get('no_equipment_score', 0.0)):.5f}"
        if no_equipment
        else f"score={float(candidate.get('score', 0.0)):.5f}"
    )
    return f"""<article class="card{reference_class}">
  {f'<img src="{html.escape(preview, quote=True)}" alt="{name_en}">' if preview else ''}
  <h3><span class="rank">#{int(candidate.get('rank', 0))}</span> {name}{reference_badge}</h3>
  {alternatives_note}
  {no_equipment_badge}
  <p>{identifier} · {score_label}</p>
  <p>{source_link}</p>
</article>"""


def _visual_panel(title: str, uri: str | None) -> str:
    if not uri:
        return ""
    return (
        f'<figure class="panel"><figcaption>{html.escape(title)}</figcaption>'
        f'<img src="{html.escape(uri, quote=True)}" alt="{html.escape(title)}"></figure>'
    )


def _file_uri(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return path.resolve().as_uri()


def write_json_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
