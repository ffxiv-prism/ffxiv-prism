from __future__ import annotations

import argparse
import gc
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.item_names import enrich_report_item_names
from ffxiv_prism.matching.equipment_slots import filter_candidates_for_slot, load_official_slots
from ffxiv_prism.matching.inference import (
    apply_ensemble_top1_gate,
    apply_multislot_consistency_gate,
    apply_occupied_slot_top1_gate,
    SLOTS,
    base_configs,
    collapse_appearance_equivalent_candidates,
    constrain_candidates_to_pool,
    crop_records,
    crop_weight,
    current_best_configs,
    filter_records_for_slot,
    fuse_branch_search_results,
    fuse_crops_then_branch_results,
    fuse_crop_results,
    inspect_assets,
    no_visible_equipment_decision,
    attach_appearance_equivalent_candidates,
    prepend_no_equipment_candidate,
    rerank_occupied_slot_search_results,
    select_slot_candidate_index_rows,
    slot_region_presence_decision,
    slot_result,
    validate_input_image,
    write_html_report,
    write_json_report,
)
from ffxiv_prism.matching.head_morphology import (
    head_morphology_class,
    predict_ridge_probabilities,
)
from ffxiv_prism.model_assets import load_pretrained
from ffxiv_prism.matching.garment_morphology import (
    LONG_BODY_COVERED_LEGWEAR_CANDIDATE_PROMPTS,
    LONG_BODY_COVERED_LEGWEAR_QUERY_PROMPTS,
    LONG_BODY_COVERED_LEGWEAR_RERANK_BLEND,
    LONG_BODY_LEGWEAR_COVERAGE_CLASSES,
    LONG_BODY_LEGWEAR_COVERAGE_PROMPTS,
    LONG_BODY_LEGWEAR_MIN_LONG_TOP_PROBABILITY,
    LONG_BODY_LEGWEAR_MIN_STYLE_CONFIDENCE,
    LONG_BODY_LEGWEAR_MIN_VISIBILITY,
    LONG_BODY_LEGWEAR_QUERY_CROPS,
    LONG_BODY_LEGWEAR_VISIBILITY_CLASSES,
    LONG_BODY_LEGWEAR_VISIBILITY_PROMPTS,
    STYLE_CANDIDATE_CROP_BOXES_NORMALIZED,
    STYLE_PROMPTS_BY_SLOT,
    TARGET_STYLE_CLASSES,
    StylePrediction,
    apply_style_crop_multipliers,
    long_body_legwear_decision,
    long_top_probability,
    rerank_candidates_with_legwear_profile,
    rerank_candidates_with_style,
    style_policy,
)
from ffxiv_prism.paths import (
    APPEARANCE_EQUIVALENCE_PATH,
    PROCESSED_DIR,
    PROJECT_ROOT,
    portable_path,
    resolve_project_path,
)
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.outfit_regions import (
    DEFAULT_POSE_MODEL,
    SinglePersonPoseEstimator,
    build_outfit_regions,
    pose_quality,
    region_manifest,
    render_crop_contact_sheet,
    render_debug_overlay,
    save_region_crops,
)


NO_EQUIPMENT_PROMPTS_BY_SLOT = {
    "HEAD": (
        "a character with no headgear, only natural hair is visible",
        "a character wearing visible headgear or a head accessory",
    ),
    "HANDS": (
        "natural bare hands with uncovered fingers and no hand equipment",
        (
            "hands wearing visible gloves, fingerless gloves, gauntlets, "
            "armguards, or wristbands"
        ),
    ),
}
MULTISLOT_SUPPORT_TOP_K = 5
EMPEROR_NO_EQUIPMENT_GAME_ITEM_IDS_BY_SLOT = {
    "HEAD": 10032,
    "BODY": 10033,
    "HANDS": 10034,
    "LEGS": 10035,
    "FEET": 10036,
}
_APPEARANCE_EQUIVALENCE: dict[int, tuple[int, ...]] | None = None
_APPEARANCE_EQUIVALENCE_CATALOG: dict[int, dict[str, Any]] | None = None
_EMPEROR_NO_EQUIPMENT_CATALOG: dict[str, dict[str, Any]] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match one FFXIV-style character screenshot to ranked gear candidates "
            "for HEAD, BODY, HANDS, LEGS, and FEET."
        )
    )
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--slots", nargs="+", choices=SLOTS, default=list(SLOTS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--gender", choices=("female", "male"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("current-best", "base"),
        default="current-best",
        help="Use promoted/query-trained local models or the base DINOv2 fallback.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to data/outputs/glamour-match/<image-stem>.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate selected local indexes/checkpoints without running inference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.candidate_k < args.top_k:
        parser.error("--candidate-k must be at least --top-k")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    all_configs = current_best_configs() if args.profile == "current-best" else base_configs()
    configs = {slot: all_configs[slot] for slot in args.slots}
    assets = inspect_assets(configs)
    if args.check:
        print(json.dumps(assets, ensure_ascii=False, indent=2))
        return 0 if assets["ok"] else 2
    if args.image is None:
        parser.error("Provide an input image, or use --check")
    if not assets["ok"]:
        problems = [*assets["missing"], *assets["invalid"]]
        parser.error("Required local assets are not ready:\n  " + "\n  ".join(problems))

    image_info = validate_input_image(args.image)
    known_reference = _find_known_reference(args.image)
    output_dir = args.output_dir or (
        PROJECT_ROOT / "data" / "outputs" / "glamour-match" / args.image.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)

    print(f"stage=pose device={device} image={args.image}", flush=True)
    estimator = SinglePersonPoseEstimator(args.pose_model, str(device))
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    keypoints, person_box = estimator.estimate(image, refine=not args.no_refine)
    regions = build_outfit_regions(keypoints, image.size)
    crop_dir = output_dir / "crops"
    paths = save_region_crops(image, regions, crop_dir, args.image.stem)
    overlay_path = output_dir / "overlay.jpg"
    contact_path = output_dir / "crops.jpg"
    render_debug_overlay(image, keypoints, person_box, regions).save(overlay_path, quality=94)
    render_crop_contact_sheet(image, regions).save(contact_path, quality=92)
    image.close()
    records = crop_records(regions, paths)
    del estimator
    _release_memory()

    warnings = [
        "Candidate scores are ranking signals, not calibrated probabilities.",
        "The system assumes one main person and works best on clear front/full-body FFXIV screenshots.",
    ]
    quality = pose_quality(keypoints, person_box)
    if quality < 0.55:
        warnings.append(
            f"Pose quality is low ({quality:.3f}); inspect overlay.jpg before trusting candidates."
        )

    results = []
    if args.profile == "base":
        results.extend(
            _run_base_profile(
                configs,
                records,
                device,
                args.batch_size,
                args.candidate_k,
                args.top_k,
                args.gender,
                known_reference.get("core_slot_labels", {}) if known_reference else {},
            )
        )
    else:
        execution_slots = list(configs)
        if (
            "BODY" in execution_slots
            and "LEGS" in execution_slots
            and execution_slots.index("BODY") > execution_slots.index("LEGS")
        ):
            execution_slots.remove("BODY")
            execution_slots.insert(execution_slots.index("LEGS"), "BODY")
        results_by_slot: dict[str, dict[str, Any]] = {}
        for slot in execution_slots:
            config = configs[slot]
            print(f"stage=retrieve slot={slot}", flush=True)
            result = _run_slot(
                config,
                records,
                device,
                args.batch_size,
                args.candidate_k,
                args.top_k,
                args.gender,
                (
                    (known_reference.get("core_slot_labels") or {}).get(slot)
                    if known_reference
                    else None
                ),
                body_style_diagnostic=(
                    results_by_slot.get("BODY", {}).get("style_diagnostic")
                ),
            )
            results_by_slot[slot] = result
        results.extend(results_by_slot[slot] for slot in configs)

    occupied_slots_by_piece = _load_piece_occupied_slots(
        {
            int(item["piece_id"])
            for result in results
            for candidate in result.get("candidates", [])
            for item in [candidate, *candidate.get("appearance_alternatives", [])]
        }
    )
    results = apply_multislot_consistency_gate(
        results,
        occupied_slots_by_piece,
        support_top_k=MULTISLOT_SUPPORT_TOP_K,
        display_top_k=args.top_k,
    )

    manifest_path = output_dir / "regions.json"
    manifest = {
        "source_image": portable_path(args.image),
        "model_id": args.pose_model,
        "pose_quality": quality,
        "known_reference": (
            {
                "example_id": known_reference.get("example_id"),
                "glamour_id": known_reference.get("glamour_id"),
                "title": known_reference.get("title"),
                "source": "frozen labelled evaluation dataset",
            }
            if known_reference
            else None
        ),
        **region_manifest(keypoints, person_box, regions),
        "crop_paths": [portable_path(path) for path in paths],
        "overlay_path": portable_path(overlay_path),
        "contact_sheet_path": portable_path(contact_path),
    }
    write_json_report(manifest_path, manifest)

    payload: dict[str, Any] = {
        "version": 1,
        "input": image_info,
        "profile": args.profile,
        "device": str(device),
        "gender_filter": args.gender,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "pose_quality": quality,
        "known_reference": manifest.get("known_reference"),
        "warnings": warnings,
        "artifacts": {
            "regions": portable_path(manifest_path),
            "overlay": portable_path(overlay_path),
            "contact_sheet": portable_path(contact_path),
        },
        "slots": results,
    }
    json_path = output_dir / "results.json"
    html_path = output_dir / "report.html"
    payload = enrich_report_item_names(payload)
    write_json_report(json_path, payload)
    write_html_report(html_path, payload)
    print(
        json.dumps(
            {
                "status": "ok",
                "json": str(json_path.resolve()),
                "html": str(html_path.resolve()),
                "overlay": str(overlay_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _load_piece_occupied_slots(
    piece_ids: set[int],
) -> dict[int, tuple[str, ...]]:
    if not piece_ids:
        return {}
    placeholders = ",".join("?" for _ in piece_ids)
    connection = sqlite3.connect(PROCESSED_DIR / "eorzea.sqlite")
    try:
        rows = connection.execute(
            (
                "SELECT piece_id, occupied_slots_json FROM gear_pieces "
                f"WHERE piece_id IN ({placeholders})"
            ),
            sorted(piece_ids),
        ).fetchall()
    finally:
        connection.close()
    return {
        int(piece_id): tuple(str(value).upper() for value in json.loads(raw_slots))
        for piece_id, raw_slots in rows
    }


def _load_emperor_no_equipment_catalog() -> dict[str, dict[str, Any]]:
    global _EMPEROR_NO_EQUIPMENT_CATALOG
    if _EMPEROR_NO_EQUIPMENT_CATALOG is not None:
        return _EMPEROR_NO_EQUIPMENT_CATALOG

    path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "game_data"
        / "cafemaker_item_names_raw.jsonl"
    )
    wanted_ids = set(EMPEROR_NO_EQUIPMENT_GAME_ITEM_IDS_BY_SLOT.values())
    rows_by_id = {
        int(row["game_item_id"]): row
        for row in read_jsonl(path)
        if int(row.get("game_item_id", -1)) in wanted_ids
    } if path.is_file() else {}
    catalog: dict[str, dict[str, Any]] = {}
    for slot, game_item_id in EMPEROR_NO_EQUIPMENT_GAME_ITEM_IDS_BY_SLOT.items():
        row = rows_by_id.get(game_item_id)
        if row is None:
            continue
        name_en = str(row.get("name_en") or "")
        if not name_en.startswith("The Emperor's New "):
            continue
        catalog[slot] = {
            **dict(row),
            "slot": slot,
            "source_item_url": row.get("source"),
        }
    _EMPEROR_NO_EQUIPMENT_CATALOG = catalog
    return catalog


def _load_appearance_equivalence() -> dict[int, tuple[int, ...]]:
    global _APPEARANCE_EQUIVALENCE
    if _APPEARANCE_EQUIVALENCE is None:
        if not APPEARANCE_EQUIVALENCE_PATH.is_file():
            raise FileNotFoundError(
                "Required same-model equivalence asset is missing: "
                f"{APPEARANCE_EQUIVALENCE_PATH}. Restore tracked files with "
                "`git pull` before generating a report."
            )
        payload = json.loads(
            APPEARANCE_EQUIVALENCE_PATH.read_text(encoding="utf-8")
        )
        equivalents_by_piece = payload.get("same_model_equivalents")
        if not isinstance(equivalents_by_piece, dict) or not equivalents_by_piece:
            raise ValueError(
                "Same-model equivalence asset has no usable "
                "same_model_equivalents mapping: "
                f"{APPEARANCE_EQUIVALENCE_PATH}"
            )
        _APPEARANCE_EQUIVALENCE = {
            int(piece_id): tuple(int(value) for value in equivalents)
            for piece_id, equivalents in equivalents_by_piece.items()
        }
    return _APPEARANCE_EQUIVALENCE


def _load_appearance_equivalence_catalog() -> dict[int, dict[str, Any]]:
    global _APPEARANCE_EQUIVALENCE_CATALOG
    if _APPEARANCE_EQUIVALENCE_CATALOG is None:
        direct_equivalents = _load_appearance_equivalence()
        wanted_ids = set(direct_equivalents)
        wanted_ids.update(
            equivalent_id
            for equivalents in direct_equivalents.values()
            for equivalent_id in equivalents
        )
        sorted_wanted_ids = sorted(wanted_ids)
        database_path = PROCESSED_DIR / "eorzea.sqlite"
        catalog: dict[int, dict[str, Any]] = {}
        if sorted_wanted_ids and database_path.is_file():
            connection = sqlite3.connect(database_path)
            connection.row_factory = sqlite3.Row
            try:
                for offset in range(0, len(sorted_wanted_ids), 900):
                    piece_ids = sorted_wanted_ids[offset : offset + 900]
                    placeholders = ",".join("?" for _ in piece_ids)
                    rows = connection.execute(
                        (
                            "SELECT piece_id, name, name_en, name_zh, "
                            "source_item_url FROM gear_pieces "
                            f"WHERE piece_id IN ({placeholders})"
                        ),
                        piece_ids,
                    ).fetchall()
                    catalog.update(
                        {
                            int(row["piece_id"]): dict(row)
                            for row in rows
                        }
                    )
            finally:
                connection.close()
        _APPEARANCE_EQUIVALENCE_CATALOG = catalog
    return _APPEARANCE_EQUIVALENCE_CATALOG


def _run_slot(
    config: Any,
    records: dict[str, list[dict[str, Any]]],
    device: torch.device,
    batch_size: int,
    candidate_k: int,
    top_k: int,
    gender: str | None,
    reference_label: dict[str, Any] | None = None,
    body_style_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from transformers import AutoModel

    from ffxiv_prism.jsonl import read_jsonl
    from ffxiv_prism.matching.spatial_profiles import (
        encode_dino_row_profiles,
        load_spatial_profile_index,
        rerank_candidates_with_spatial_profile,
    )
    from scripts.evaluate_glamour_embedding_model import build_encoder, search_crops
    from scripts.train_glamour_pair_reranker import PairReranker, rerank_crops
    from scripts.train_glamour_query_encoder import (
        encode_records,
        load_query_head_from_checkpoint,
    )

    slot_records = filter_records_for_slot(records, config.slot)
    slot_presence_diagnostic = slot_region_presence_decision(
        slot_records["query"]
    )
    if not slot_presence_diagnostic["contains_slot"]:
        result = slot_result(config, slot_records["query"], [])
        result["status"] = "slot_not_in_frame"
        result["slot_presence_diagnostic"] = slot_presence_diagnostic
        return result

    no_equipment_diagnostic = None
    if config.no_equipment_model_id is not None:
        allowed_no_equipment_crops = set(
            config.no_equipment_query_crop_names
        )
        no_equipment_records = [
            row
            for row in slot_records["query"]
            if str(row.get("name")) in allowed_no_equipment_crops
            and str(row.get("visibility"))
            not in {"pose_uncertain", "out_of_frame"}
        ]
        no_equipment_paths = [
            resolve_project_path(row.get("path"))
            for row in no_equipment_records
        ]
        if no_equipment_records and all(
            path is not None and path.is_file()
            for path in no_equipment_paths
        ):
            prompts = NO_EQUIPMENT_PROMPTS_BY_SLOT.get(config.slot)
            if prompts is None:
                raise ValueError(
                    f"No no-equipment prompts are configured for {config.slot}"
                )
            probabilities = _classify_style_paths(
                str(config.no_equipment_model_id),
                prompts,
                [path for path in no_equipment_paths if path is not None],
                device=device,
                batch_size=batch_size,
            )
            no_equipment_diagnostic = no_visible_equipment_decision(
                probabilities,
                minimum_probability=config.no_equipment_min_probability,
            )
            no_equipment_diagnostic["query_crops"] = [
                str(row.get("name")) for row in no_equipment_records
            ]
        else:
            no_equipment_diagnostic = {
                "active": False,
                "no_visible_equipment": False,
                "reason": "query_crops_unavailable",
                "query_crops": [
                    str(row.get("name")) for row in no_equipment_records
                ],
            }
    active_candidate_crop_routes = config.candidate_crop_routes
    body_long_probability = long_top_probability(body_style_diagnostic)
    long_body_legwear_routing = (
        config.slot == "LEGS"
        and body_long_probability
        >= LONG_BODY_LEGWEAR_MIN_LONG_TOP_PROBABILITY
    )
    style_prediction = None
    style_policy_result = None
    style_diagnostic = None
    long_body_legwear_diagnostic = None
    long_body_legwear_profile = None
    long_body_legwear_mode = None
    long_body_legwear_blend = None
    if long_body_legwear_routing and config.style_model_id is not None:
        legwear_records_by_name = {
            str(row.get("name")): row
            for row in slot_records["query"]
            if str(row.get("name")) in LONG_BODY_LEGWEAR_QUERY_CROPS
            and str(row.get("visibility"))
            not in {"pose_uncertain", "out_of_frame"}
        }
        legwear_records = [
            legwear_records_by_name[name]
            for name in LONG_BODY_LEGWEAR_QUERY_CROPS
            if name in legwear_records_by_name
        ]
        legwear_paths = [
            resolve_project_path(row.get("path")) for row in legwear_records
        ]
        if legwear_records and all(
            path is not None and path.is_file() for path in legwear_paths
        ):
            query_profiles = _classify_style_path_prompt_groups(
                str(config.style_model_id),
                {
                    "visibility": LONG_BODY_LEGWEAR_VISIBILITY_PROMPTS,
                    "coverage": LONG_BODY_LEGWEAR_COVERAGE_PROMPTS,
                    "covered_style": LONG_BODY_COVERED_LEGWEAR_QUERY_PROMPTS,
                },
                [path for path in legwear_paths if path is not None],
                device=device,
                batch_size=batch_size,
            )
            legwear_decision = long_body_legwear_decision(
                query_profiles["visibility"],
                query_profiles["coverage"],
                query_profiles["covered_style"],
            )
            selected_crop = (
                str(legwear_records[legwear_decision.crop_index].get("name"))
                if legwear_decision.crop_index is not None
                else None
            )
            rerank_active = (
                legwear_decision.observable
                and legwear_decision.confidence
                >= LONG_BODY_LEGWEAR_MIN_STYLE_CONFIDENCE
            )
            if rerank_active:
                long_body_legwear_profile = legwear_decision.probabilities
                long_body_legwear_mode = legwear_decision.mode
                long_body_legwear_blend = (
                    config.style_blend
                    if legwear_decision.mode == "exposed"
                    else LONG_BODY_COVERED_LEGWEAR_RERANK_BLEND
                )
            reason = legwear_decision.reason
            if legwear_decision.observable and not rerank_active:
                reason = "legwear_style_confidence_below_threshold"
            style_diagnostic = {
                "query_crop": selected_crop,
                "query_crops": [
                    str(row.get("name")) for row in legwear_records
                ],
                "classes": list(legwear_decision.classes),
                "probabilities": dict(
                    zip(
                        legwear_decision.classes,
                        legwear_decision.probabilities,
                        strict=True,
                    )
                ),
                "predicted_class": legwear_decision.predicted_class,
                "confidence": legwear_decision.confidence,
                "minimum_confidence": LONG_BODY_LEGWEAR_MIN_STYLE_CONFIDENCE,
                "active": rerank_active,
                "reason": reason,
                "rerank_k": config.style_rerank_k,
                "blend": long_body_legwear_blend,
                "ranking_changed": False,
            }
            crop_profiles = []
            for index, row in enumerate(legwear_records):
                crop_profiles.append(
                    {
                        "crop": str(row.get("name")),
                        "visibility": dict(
                            zip(
                                LONG_BODY_LEGWEAR_VISIBILITY_CLASSES,
                                (
                                    float(value)
                                    for value in query_profiles["visibility"][index]
                                ),
                                strict=True,
                            )
                        ),
                        "coverage": dict(
                            zip(
                                LONG_BODY_LEGWEAR_COVERAGE_CLASSES,
                                (
                                    float(value)
                                    for value in query_profiles["coverage"][index]
                                ),
                                strict=True,
                            )
                        ),
                    }
                )
            long_body_legwear_diagnostic = {
                "active": rerank_active,
                "reason": reason,
                "body_long_top_probability": body_long_probability,
                "minimum_body_long_top_probability": (
                    LONG_BODY_LEGWEAR_MIN_LONG_TOP_PROBABILITY
                ),
                "observable": legwear_decision.observable,
                "visible_probability": legwear_decision.visible_probability,
                "minimum_visibility": LONG_BODY_LEGWEAR_MIN_VISIBILITY,
                "mode": legwear_decision.mode,
                "mode_probability": legwear_decision.mode_probability,
                "selected_crop": selected_crop,
                "classes": list(legwear_decision.classes),
                "probabilities": dict(
                    zip(
                        legwear_decision.classes,
                        legwear_decision.probabilities,
                        strict=True,
                    )
                ),
                "predicted_class": legwear_decision.predicted_class,
                "confidence": legwear_decision.confidence,
                "minimum_style_confidence": (
                    LONG_BODY_LEGWEAR_MIN_STYLE_CONFIDENCE
                ),
                "crop_profiles": crop_profiles,
                "rerank_k": config.style_rerank_k,
                "blend": long_body_legwear_blend,
                "ranking_changed": False,
            }
        else:
            style_diagnostic = {
                "query_crop": config.style_query_crop_name,
                "query_crops": [
                    str(row.get("name")) for row in legwear_records
                ],
                "active": False,
                "reason": "query_crops_unavailable",
                "ranking_changed": False,
            }
            long_body_legwear_diagnostic = {
                "active": False,
                "reason": "query_crops_unavailable",
                "body_long_top_probability": body_long_probability,
                "observable": False,
                "ranking_changed": False,
            }
    elif (
        config.style_model_id is not None
        and config.style_query_crop_name is not None
    ):
        style_record = next(
            (
                row
                for row in slot_records["query"]
                if str(row.get("name")) == config.style_query_crop_name
                and str(row.get("visibility"))
                not in {"pose_uncertain", "out_of_frame"}
            ),
            None,
        )
        if style_record is None:
            style_diagnostic = {
                "query_crop": config.style_query_crop_name,
                "active": False,
                "reason": "query_crop_unavailable",
                "ranking_changed": False,
            }
        else:
            style_path = resolve_project_path(style_record.get("path"))
            if style_path is None or not style_path.is_file():
                raise FileNotFoundError(
                    f"Style query crop does not exist: {style_record.get('path')}"
                )
            style_probabilities = _classify_style_paths(
                config.style_model_id,
                STYLE_PROMPTS_BY_SLOT[config.slot],
                [style_path],
                device=device,
                batch_size=1,
            )[0]
            style_prediction = StylePrediction.from_probabilities(
                config.slot,
                style_probabilities,
            )
            style_policy_result = style_policy(
                style_prediction,
                minimum_confidence=config.style_min_confidence,
            )
            if style_policy_result.active:
                slot_records = {
                    example_id: apply_style_crop_multipliers(
                        rows,
                        style_policy_result,
                    )
                    for example_id, rows in slot_records.items()
                }
                active_candidate_crop_routes = {
                    **dict(config.candidate_crop_routes or {}),
                    **dict(style_policy_result.candidate_crop_routes),
                }
            style_diagnostic = {
                "query_crop": config.style_query_crop_name,
                "classes": list(style_prediction.classes),
                "probabilities": dict(
                    zip(
                        style_prediction.classes,
                        style_prediction.probabilities,
                        strict=True,
                    )
                ),
                "predicted_class": style_prediction.predicted_class,
                "confidence": style_prediction.confidence,
                "target_probability": sum(
                    probability
                    for class_name, probability in zip(
                        style_prediction.classes,
                        style_prediction.probabilities,
                        strict=True,
                    )
                    if class_name in TARGET_STYLE_CLASSES[config.slot]
                ),
                "minimum_confidence": config.style_min_confidence,
                "active": style_policy_result.active,
                "reason": style_policy_result.reason,
                "crop_multipliers": dict(
                    style_policy_result.crop_multipliers
                ),
                "candidate_crop_routes": {
                    name: list(routes)
                    for name, routes in (
                        style_policy_result.candidate_crop_routes.items()
                    )
                },
                "retrieval_role_overrides": dict(
                    style_policy_result.retrieval_role_overrides
                ),
                "rerank_k": config.style_rerank_k,
                "blend": config.style_blend,
                "ranking_changed": False,
            }
    if config.query_crop_names is not None:
        allowed_names = set(config.query_crop_names)
        slot_records = {
            example_id: [
                row
                for row in rows
                if str(row.get("name")) in allowed_names
            ]
            for example_id, rows in slot_records.items()
        }
    # QA-only crops (for example the expanded full-shoe diagnostics) are kept
    # in regions.json/contact sheets but must not change a promoted retriever's
    # input distribution or consume encoder memory.
    slot_records = {
        example_id: [row for row in rows if crop_weight(row) > 0.0]
        for example_id, rows in slot_records.items()
    }
    usable = [
        row
        for row in slot_records["query"]
        if row["visibility"] not in {"pose_uncertain", "out_of_frame"}
    ]
    if not usable:
        result = slot_result(config, slot_records["query"], [])
        result["slot_presence_diagnostic"] = slot_presence_diagnostic
        return result

    metadata = read_jsonl(config.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(
        config.index_dir / "embeddings.npy",
        mmap_mode="r",
    )
    metadata, candidate_embeddings = select_slot_candidate_index_rows(
        metadata,
        candidate_embeddings,
        config.slot,
    )
    model = load_pretrained(AutoModel, "facebook/dinov2-base").to(device)
    spatial_index = None
    spatial_query_profile = None
    if config.spatial_index is not None:
        spatial_index = load_spatial_profile_index(config.spatial_index)
        allowed_spatial_names = set(config.spatial_query_crop_names or ())
        spatial_rows = [
            row
            for row in slot_records["query"]
            if not allowed_spatial_names
            or str(row.get("name")) in allowed_spatial_names
        ]
        if spatial_rows:
            priority = {
                name: rank
                for rank, name in enumerate(config.spatial_query_crop_names or ())
            }
            spatial_row = min(
                spatial_rows,
                key=lambda row: priority.get(
                    str(row.get("name")), len(priority)
                ),
            )
            spatial_query_profile = encode_dino_row_profiles(
                model,
                [{"path": spatial_row["path"]}],
                projection=spatial_index.projection,
                batch_size=1,
                device=device,
            )[0]
    query_head = (
        load_query_head_from_checkpoint(config.query_checkpoint, model, device)
        if config.query_checkpoint is not None
        else None
    )
    embeddings = encode_records(model, query_head, slot_records, batch_size, device)
    candidate_gender = (
        {"query": gender}
        if gender and config.gender_routing
        else None
    )
    if config.reranker_checkpoint is not None:
        payload = torch.load(
            config.reranker_checkpoint,
            map_location=device,
            weights_only=True,
        )
        reranker = PairReranker(
            int(payload["dimension"]),
            per_slot_heads=bool(payload.get("per_slot_heads", False)),
            dropout=float(payload.get("dropout", 0.1)),
        ).to(device)
        reranker.load_state_dict(payload["state_dict"])
        search = rerank_crops(
            reranker,
            metadata,
            candidate_embeddings,
            slot_records,
            embeddings,
            candidate_k,
            candidate_k,
            device,
            candidate_view_aggregate="max",
        )
        del reranker, payload
    else:
        search = search_crops(
            metadata,
            candidate_embeddings,
            slot_records,
            embeddings,
            candidate_k,
            device=str(device),
            candidate_crop_routes=active_candidate_crop_routes,
            candidate_gender_by_example=candidate_gender,
        )

    final_pool_search = None
    crop_then_branch_candidates = None
    if config.ensemble_index_dirs:
        if config.reranker_checkpoint is not None:
            raise ValueError("Rank ensemble cannot follow a pair reranker")
        if len(config.ensemble_model_ids) != len(config.ensemble_index_dirs):
            raise ValueError("Ensemble model/index counts do not align")
        ensemble_query_checkpoints = config.ensemble_query_checkpoints or (
            (None,) * len(config.ensemble_index_dirs)
        )
        if len(ensemble_query_checkpoints) != len(config.ensemble_index_dirs):
            raise ValueError("Ensemble checkpoint/index counts do not align")
        branch_searches = [search]
        for ensemble_model_id, ensemble_index_dir, ensemble_query_checkpoint in zip(
            config.ensemble_model_ids,
            config.ensemble_index_dirs,
            ensemble_query_checkpoints,
            strict=True,
        ):
            ensemble_metadata = read_jsonl(ensemble_index_dir / "metadata.jsonl")
            ensemble_candidate_embeddings = np.load(
                ensemble_index_dir / "embeddings.npy",
                mmap_mode="r",
            )
            ensemble_metadata, ensemble_candidate_embeddings = (
                select_slot_candidate_index_rows(
                    ensemble_metadata,
                    ensemble_candidate_embeddings,
                    config.slot,
                )
            )
            if [row["local_path"] for row in ensemble_metadata] != [
                row["local_path"] for row in metadata
            ]:
                raise ValueError(
                    "Primary and ensemble candidate metadata do not align"
                )
            query_paths = [Path(str(row["path"])) for row in slot_records["query"]]
            if ensemble_query_checkpoint is None:
                ensemble_encoder = build_encoder(
                    ensemble_model_id,
                    "auto",
                    str(device),
                    "color",
                )
                query_vectors = ensemble_encoder.encode_paths(query_paths, batch_size)
                ensemble_queries = dict(
                    zip(
                        [str(path) for path in query_paths],
                        query_vectors,
                        strict=True,
                    )
                )
                del ensemble_encoder, query_vectors
            else:
                ensemble_model = load_pretrained(
                    AutoModel,
                    ensemble_model_id,
                ).to(device)
                ensemble_query_head = load_query_head_from_checkpoint(
                    ensemble_query_checkpoint,
                    ensemble_model,
                    device,
                )
                ensemble_queries = encode_records(
                    ensemble_model,
                    ensemble_query_head,
                    slot_records,
                    batch_size,
                    device,
                )
                del ensemble_model, ensemble_query_head
            branch_searches.append(
                search_crops(
                    ensemble_metadata,
                    ensemble_candidate_embeddings,
                    slot_records,
                    ensemble_queries,
                    candidate_k,
                    device=str(device),
                    candidate_crop_routes=active_candidate_crop_routes,
                    candidate_gender_by_example=candidate_gender,
                )
            )
            del (
                ensemble_queries,
                ensemble_metadata,
                ensemble_candidate_embeddings,
            )
            _release_memory()
        if config.ensemble_fusion_order == "crop_then_branch":
            if config.ensemble_pool_branches is not None:
                raise ValueError(
                    "crop_then_branch does not support ensemble_pool_branches"
                )
            crop_then_branch_candidates = fuse_crops_then_branch_results(
                slot_records["query"],
                branch_searches,
                top_k=candidate_k,
                candidate_k=candidate_k,
                crop_fusion=config.fusion,
                crop_rrf_k=config.rrf_k,
                branch_rrf_k=config.ensemble_rrf_k,
                final_pool_branches=config.ensemble_final_pool_branches,
                branch_weights=config.ensemble_branch_weights or None,
            )
            if config.ensemble_gate_baseline_branch_weights:
                baseline_candidates = fuse_crops_then_branch_results(
                    slot_records["query"],
                    branch_searches,
                    top_k=candidate_k,
                    candidate_k=candidate_k,
                    crop_fusion=config.fusion,
                    crop_rrf_k=config.rrf_k,
                    branch_rrf_k=config.ensemble_rrf_k,
                    final_pool_branches=config.ensemble_final_pool_branches,
                    branch_weights=config.ensemble_gate_baseline_branch_weights,
                )
                last_branch_candidates = fuse_crop_results(
                    slot_records["query"],
                    branch_searches[-1],
                    top_k=candidate_k,
                    fusion=config.fusion,
                    rrf_k=config.rrf_k,
                )
                crop_then_branch_candidates = apply_ensemble_top1_gate(
                    baseline_candidates,
                    crop_then_branch_candidates,
                    last_branch_candidates,
                    max_base_rank=int(
                        config.ensemble_top1_gate_max_base_rank or 0
                    ),
                    max_last_branch_rank=int(
                        config.ensemble_top1_gate_max_last_branch_rank or 0
                    ),
                    max_base_last_branch_rank=int(
                        config.ensemble_top1_gate_max_base_last_branch_rank or 0
                    ),
                    min_last_branch_rank_ratio=float(
                        config.ensemble_top1_gate_min_last_branch_rank_ratio or 0
                    ),
                )
        else:
            candidate_pool = None
            if config.ensemble_pool_branches is not None:
                if not 1 <= config.ensemble_pool_branches <= len(branch_searches):
                    raise ValueError("Invalid ensemble_pool_branches")
                candidate_pool = fuse_branch_search_results(
                    branch_searches[: config.ensemble_pool_branches],
                    top_k=candidate_k,
                    rrf_k=config.ensemble_pool_rrf_k,
                )
            search = fuse_branch_search_results(
                branch_searches,
                top_k=candidate_k,
                rrf_k=config.ensemble_rrf_k,
                candidate_pool=candidate_pool,
                branch_weights=config.ensemble_branch_weights or None,
            )
            if config.ensemble_final_pool_branches is not None:
                if not 1 <= config.ensemble_final_pool_branches <= len(branch_searches):
                    raise ValueError("Invalid ensemble_final_pool_branches")
                final_pool_search = fuse_branch_search_results(
                    branch_searches[: config.ensemble_final_pool_branches],
                    top_k=candidate_k,
                    rrf_k=config.ensemble_pool_rrf_k,
                )
            del candidate_pool
        del branch_searches

    # Keep the full retrieval pool until same-model deduplication and the
    # cross-slot gate finish. Truncating here can leave the report underfilled.
    retrieval_top_k = candidate_k
    if spatial_index is not None and spatial_query_profile is not None:
        retrieval_top_k = max(retrieval_top_k, config.spatial_rerank_k)
    if style_policy_result is not None and style_policy_result.active:
        retrieval_top_k = max(retrieval_top_k, config.style_rerank_k)
    if long_body_legwear_profile is not None:
        retrieval_top_k = max(retrieval_top_k, config.style_rerank_k)
    candidates = (
        crop_then_branch_candidates[:retrieval_top_k]
        if crop_then_branch_candidates is not None
        else fuse_crop_results(
            slot_records["query"],
            search,
            top_k=retrieval_top_k,
            fusion=config.fusion,
            rrf_k=config.rrf_k,
        )
    )
    if (
        config.occupied_slot_multiplier is not None
        and config.occupied_slot_gate_max_base_rank is not None
    ):
        occupied_adjusted_search = rerank_occupied_slot_search_results(
            search,
            occupied_multiplier=config.occupied_slot_multiplier,
            rrf_k=1,
        )
        occupied_adjusted_candidates = fuse_crop_results(
            slot_records["query"],
            occupied_adjusted_search,
            top_k=retrieval_top_k,
            fusion=config.fusion,
            rrf_k=config.rrf_k,
        )
        candidates = apply_occupied_slot_top1_gate(
            candidates,
            occupied_adjusted_candidates,
            max_base_rank=config.occupied_slot_gate_max_base_rank,
        )
    if final_pool_search is not None:
        pool_candidates = fuse_crop_results(
            slot_records["query"],
            final_pool_search,
            top_k=candidate_k,
            fusion=config.fusion,
            rrf_k=config.rrf_k,
        )
        candidates = constrain_candidates_to_pool(pool_candidates, candidates)
    if spatial_index is not None and spatial_query_profile is not None:
        candidates = rerank_candidates_with_spatial_profile(
            candidates,
            spatial_index,
            spatial_query_profile,
            rerank_k=config.spatial_rerank_k,
            blend=config.spatial_blend,
        )
    if (
        long_body_legwear_profile is not None
        and long_body_legwear_blend is not None
        and config.style_model_id is not None
        and candidates
    ):
        style_pool = candidates[: min(config.style_rerank_k, len(candidates))]
        candidate_style_paths = [
            resolve_project_path(row.get("local_preview"))
            for row in style_pool
        ]
        if all(
            path is not None and path.is_file()
            for path in candidate_style_paths
        ):
            candidate_prompts = (
                LONG_BODY_LEGWEAR_COVERAGE_PROMPTS
                if long_body_legwear_mode == "exposed"
                else LONG_BODY_COVERED_LEGWEAR_CANDIDATE_PROMPTS
            )
            candidate_crop_box = (
                STYLE_CANDIDATE_CROP_BOXES_NORMALIZED["LEGS"]["legs_thighs"]
                if long_body_legwear_mode == "exposed"
                else None
            )
            candidate_style_probabilities = _classify_style_paths(
                str(config.style_model_id),
                candidate_prompts,
                [
                    path
                    for path in candidate_style_paths
                    if path is not None
                ],
                device=device,
                batch_size=batch_size,
                crop_box_normalized=candidate_crop_box,
            )
            probabilities_by_piece = {
                int(row["piece_id"]): probabilities
                for row, probabilities in zip(
                    style_pool,
                    candidate_style_probabilities,
                    strict=True,
                )
            }
            previous_top1 = int(candidates[0]["piece_id"])
            candidates = rerank_candidates_with_legwear_profile(
                candidates,
                long_body_legwear_profile,
                probabilities_by_piece,
                rerank_k=config.style_rerank_k,
                blend=long_body_legwear_blend,
                minimum_confidence=LONG_BODY_LEGWEAR_MIN_STYLE_CONFIDENCE,
            )
            current_top1 = int(candidates[0]["piece_id"])
            if long_body_legwear_diagnostic is not None:
                long_body_legwear_diagnostic["candidate_profile_count"] = len(
                    probabilities_by_piece
                )
                long_body_legwear_diagnostic["ranking_changed"] = (
                    current_top1 != previous_top1
                )
                long_body_legwear_diagnostic[
                    "previous_top1_piece_id"
                ] = previous_top1
                long_body_legwear_diagnostic[
                    "current_top1_piece_id"
                ] = current_top1
                if style_diagnostic is not None:
                    style_diagnostic["candidate_profile_count"] = len(
                        probabilities_by_piece
                    )
                    style_diagnostic["ranking_changed"] = (
                        current_top1 != previous_top1
                    )
                    style_diagnostic["previous_top1_piece_id"] = previous_top1
                    style_diagnostic["current_top1_piece_id"] = current_top1
        elif long_body_legwear_diagnostic is not None:
            long_body_legwear_diagnostic["rerank_reason"] = (
                "candidate_preview_unavailable"
            )
    elif (
        style_prediction is not None
        and style_policy_result is not None
        and style_policy_result.active
        and candidates
    ):
        style_pool = candidates[: min(config.style_rerank_k, len(candidates))]
        candidate_style_paths = [
            resolve_project_path(row.get("local_preview"))
            for row in style_pool
        ]
        if all(
            path is not None and path.is_file()
            for path in candidate_style_paths
        ):
            candidate_crop_box = (
                STYLE_CANDIDATE_CROP_BOXES_NORMALIZED["LEGS"]["legs_thighs"]
                if config.slot == "LEGS"
                else None
            )
            candidate_style_probabilities = _classify_style_paths(
                str(config.style_model_id),
                STYLE_PROMPTS_BY_SLOT[config.slot],
                [
                    path
                    for path in candidate_style_paths
                    if path is not None
                ],
                device=device,
                batch_size=batch_size,
                crop_box_normalized=candidate_crop_box,
            )
            probabilities_by_piece = {
                int(row["piece_id"]): probabilities
                for row, probabilities in zip(
                    style_pool,
                    candidate_style_probabilities,
                    strict=True,
                )
            }
            previous_top1 = int(candidates[0]["piece_id"])
            candidates = rerank_candidates_with_style(
                candidates,
                style_prediction,
                probabilities_by_piece,
                rerank_k=config.style_rerank_k,
                blend=config.style_blend,
                minimum_confidence=config.style_min_confidence,
            )
            current_top1 = int(candidates[0]["piece_id"])
            if style_diagnostic is not None:
                style_diagnostic["candidate_profile_count"] = len(
                    probabilities_by_piece
                )
                style_diagnostic["ranking_changed"] = (
                    current_top1 != previous_top1
                )
                style_diagnostic["previous_top1_piece_id"] = previous_top1
                style_diagnostic["current_top1_piece_id"] = current_top1
        elif style_diagnostic is not None:
            style_diagnostic["rerank_reason"] = "candidate_preview_unavailable"
    appearance_equivalence = _load_appearance_equivalence()
    candidates = attach_appearance_equivalent_candidates(
        candidates,
        appearance_equivalence,
        piece_catalog=_load_appearance_equivalence_catalog(),
        max_equivalents_per_anchor=None,
    )
    candidates = filter_candidates_for_slot(candidates, config.slot, load_official_slots())
    candidates = collapse_appearance_equivalent_candidates(
        candidates,
        appearance_equivalence,
    )
    hand_shape_diagnostic = None
    if config.slot == "HANDS" and config.no_equipment_model_id is not None:
        candidates, hand_shape_diagnostic = _rerank_hands_shape(
            candidates, filter_records_for_slot(records, "HANDS")["query"],
            config.no_equipment_model_id, device=device, batch_size=batch_size,
        )
    morphology_diagnostic = None
    if config.morphology_checkpoint is not None and candidates:
        with np.load(config.morphology_checkpoint, allow_pickle=False) as payload:
            classifier_weights = np.asarray(payload["weights"], dtype=np.float32)
            classifier_classes = [
                str(value) for value in payload["classes"].tolist()
            ]
            classifier_threshold = float(payload["minimum_confidence"].tolist())
            classifier_crop = str(payload["crop_name"].tolist())
        morphology_record = next(
            (
                row
                for row in slot_records["query"]
                if str(row.get("name")) == classifier_crop
                and str(row["path"]) in embeddings
            ),
            None,
        )
        if morphology_record is not None:
            morphology_probabilities = predict_ridge_probabilities(
                np.asarray(
                    [embeddings[str(morphology_record["path"])]],
                    dtype=np.float32,
                ),
                classifier_weights,
            )[0]
            predicted_index = int(morphology_probabilities.argmax())
            predicted_class = classifier_classes[predicted_index]
            confidence = float(morphology_probabilities[predicted_index])
            top_morphology = head_morphology_class(candidates[0].get("name_en"))
            top_class = (
                "enclosing"
                if top_morphology == "enclosing"
                else "non_enclosing"
            )
            morphology_diagnostic = {
                "predicted_class": predicted_class,
                "probabilities": dict(
                    zip(
                        classifier_classes,
                        (float(value) for value in morphology_probabilities),
                        strict=True,
                    )
                ),
                "minimum_confidence": classifier_threshold,
                "high_confidence": confidence >= classifier_threshold,
                "top1_class": top_class,
                "top1_matches_prediction": top_class == predicted_class,
                "ranking_changed": False,
            }
    if no_equipment_diagnostic is not None:
        emperor_item = _load_emperor_no_equipment_catalog().get(config.slot)
        candidates = prepend_no_equipment_candidate(
            candidates,
            emperor_item,
            no_equipment_diagnostic,
        )
        no_equipment_diagnostic["candidate_added"] = bool(
            candidates and candidates[0].get("is_no_equipment")
        )
        no_equipment_diagnostic["catalog_game_item_id"] = (
            int(emperor_item["game_item_id"])
            if emperor_item is not None
            else None
        )
        if (
            no_equipment_diagnostic["no_visible_equipment"]
            and emperor_item is None
        ):
            no_equipment_diagnostic["reason"] = (
                "emperor_no_equipment_catalog_item_unavailable"
            )
    reference_piece_ids = {
        int(value) for value in (reference_label or {}).get("piece_ids", [])
    }
    reference_rank = next(
        (
            int(candidate["rank"])
            for candidate in candidates
            if int(candidate.get("piece_id", -1)) in reference_piece_ids
        ),
        None,
    )
    pooled_candidates = []
    for candidate in candidates:
        row = dict(candidate)
        row["reference_match"] = int(row.get("piece_id", -1)) in reference_piece_ids
        pooled_candidates.append(row)
    result = slot_result(config, slot_records["query"], pooled_candidates)
    result["slot_presence_diagnostic"] = slot_presence_diagnostic
    if hand_shape_diagnostic is not None:
        result["hand_shape_diagnostic"] = hand_shape_diagnostic
    if no_equipment_diagnostic is not None:
        result["no_equipment_diagnostic"] = no_equipment_diagnostic
    if style_diagnostic is not None:
        result["style_diagnostic"] = style_diagnostic
    if long_body_legwear_diagnostic is not None:
        result["long_body_legwear_diagnostic"] = long_body_legwear_diagnostic
        if not long_body_legwear_diagnostic.get("active"):
            result["ranking_quality"]["level"] = "low"
            if long_body_legwear_diagnostic.get("observable"):
                warning = (
                    "长上衣场景中的下装样式置信度不足；未加入固定下装"
                    "先验，请人工复核腿部候选。"
                )
            else:
                warning = (
                    "长上衣场景中没有足够可见的下装区域；系统保留原始"
                    "检索排序，不猜测短裤、裙子或长裤。"
                )
            result["ranking_quality"]["reasons"].append(warning)
    if morphology_diagnostic is not None:
        result["morphology_diagnostic"] = morphology_diagnostic
        if (
            morphology_diagnostic["high_confidence"]
            and not morphology_diagnostic["top1_matches_prediction"]
        ):
            result["ranking_quality"]["level"] = "low"
            predicted_description = (
                "包覆式头具"
                if morphology_diagnostic["predicted_class"] == "enclosing"
                else "开放式或小型头饰"
            )
            top_description = (
                "包覆式头具"
                if morphology_diagnostic["top1_class"] == "enclosing"
                else "开放式或小型头饰"
            )
            result["ranking_quality"]["reasons"].append(
                f"头饰形态分类器判断查询更接近{predicted_description}，"
                f"但 Top-1 是{top_description}；建议重点复核头部候选。"
            )
    if reference_label is not None:
        result["reference"] = {
            "item_name": reference_label.get("item_name"),
            "piece_ids": sorted(reference_piece_ids),
            "matched_piece_names": reference_label.get("matched_piece_names", []),
            "dyes": reference_label.get("dyes", []),
            "rank": reference_rank,
            "top1_hit": reference_rank == 1,
            "displayed_top_k_hit": (
                reference_rank is not None and reference_rank <= top_k
            ),
            "candidate_k": candidate_k,
        }
    del model, query_head, embeddings, search, metadata, candidate_embeddings
    _release_memory()
    return result


def _run_base_profile(
    configs: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    device: torch.device,
    batch_size: int,
    candidate_k: int,
    top_k: int,
    gender: str | None,
    reference_labels: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output = []
    for slot, config in configs.items():
        print(f"stage=retrieve slot={slot} profile=base", flush=True)
        output.append(
            _run_slot(
                config,
                records,
                device,
                batch_size,
                candidate_k,
                top_k,
                gender,
                (reference_labels or {}).get(slot),
            )
        )
    return output


def _rerank_hands_shape(
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    model_id: str,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ffxiv_prism.matching.hands_morphology import (
        HANDS_SHAPE_PROMPTS, HANDS_SHAPE_QUERY_CROPS,
        hands_shape_decision, rerank_candidates_with_hand_shape,
    )

    by_name = {str(row["name"]): row for row in records}
    rows = [by_name[name] for name in HANDS_SHAPE_QUERY_CROPS if name in by_name]
    paths = [resolve_project_path(row.get("path")) for row in rows]
    diagnostic = {
        "active": False, "ranking_changed": False,
        "reason": "full_hand_crops_unavailable", "model_id": model_id,
        "query_crops": list(HANDS_SHAPE_QUERY_CROPS),
    }
    if (len(rows) != 2 or any(row.get("visibility") != "visible" for row in rows)
            or any(path is None or not path.is_file() for path in paths)):
        return candidates, diagnostic
    probabilities = _classify_style_paths(
        model_id, HANDS_SHAPE_PROMPTS, paths, device=device, batch_size=batch_size,
    )
    diagnostic.update(hands_shape_decision(probabilities))
    if not diagnostic["active"] or not candidates:
        return candidates, diagnostic
    candidate_paths = [resolve_project_path(row.get("local_preview")) for row in candidates]
    if any(path is None or not path.is_file() for path in candidate_paths):
        diagnostic.update(active=False, reason="candidate_preview_unavailable")
        return candidates, diagnostic
    # A candidate can have multiple item identities sharing the same preview.
    # Classify each actual image only once, without shrinking the retrieval pool.
    unique_paths = list(dict.fromkeys(candidate_paths))
    profiles = _classify_style_paths(
        model_id, HANDS_SHAPE_PROMPTS, unique_paths, device=device, batch_size=batch_size,
    )
    by_path = dict(zip(unique_paths, profiles, strict=True))
    output, decision = rerank_candidates_with_hand_shape(
        candidates, probabilities,
        {int(row["piece_id"]): by_path[path]
         for row, path in zip(candidates, candidate_paths, strict=True)},
    )
    diagnostic.update(decision)
    return output, diagnostic


def _classify_style_paths(
    model_id: str,
    prompts: tuple[str, ...],
    paths: list[Path],
    *,
    device: torch.device,
    batch_size: int,
    crop_box_normalized: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Classify garment topology with a local SigLIP-style image/text model."""

    return _classify_style_path_prompt_groups(
        model_id,
        {"default": prompts},
        paths,
        device=device,
        batch_size=batch_size,
        crop_box_normalized=crop_box_normalized,
    )["default"]


def _classify_style_path_prompt_groups(
    model_id: str,
    prompt_groups: dict[str, tuple[str, ...]],
    paths: list[Path],
    *,
    device: torch.device,
    batch_size: int,
    crop_box_normalized: tuple[float, float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Classify several independent prompt groups with one model load."""

    if not prompt_groups or any(not prompts for prompts in prompt_groups.values()):
        raise ValueError("prompt groups must be non-empty")
    if not paths:
        return {
            name: np.empty((0, len(prompts)), dtype=np.float32)
            for name, prompts in prompt_groups.items()
        }
    from transformers import AutoModel, AutoProcessor

    flattened_prompts = []
    prompt_offsets = {}
    for name, prompts in prompt_groups.items():
        start = len(flattened_prompts)
        flattened_prompts.extend(prompts)
        prompt_offsets[name] = (start, len(flattened_prompts))
    processor = load_pretrained(AutoProcessor, model_id)
    model = load_pretrained(AutoModel, model_id).to(device).eval()
    probabilities: dict[str, list[np.ndarray]] = {
        name: [] for name in prompt_groups
    }
    try:
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                if crop_box_normalized is not None:
                    left, top, right, bottom = crop_box_normalized
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
                images.append(image)
            inputs = processor(
                text=flattened_prompts,
                images=images,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(**inputs).logits_per_image
            for name, (prompt_start, prompt_end) in prompt_offsets.items():
                probabilities[name].extend(
                    torch.softmax(
                        logits[:, prompt_start:prompt_end],
                        dim=-1,
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
            for image in images:
                image.close()
    finally:
        del model, processor
        _release_memory()
    return {
        name: np.asarray(values, dtype=np.float32)
        for name, values in probabilities.items()
    }


def _find_known_reference(image_path: Path) -> dict[str, Any] | None:
    examples_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "glamour_examples_huiji_models_closed.jsonl"
    )
    if not examples_path.is_file():
        return None
    target = image_path.resolve()
    for example in read_jsonl(examples_path):
        local_path = example.get("local_path")
        if not local_path:
            continue
        candidate = Path(str(local_path))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.resolve() == target:
            return example
    return None


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
