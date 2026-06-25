from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import (
    EORZEA_GLAMOUR_METADATA_DIR,
    PROCESSED_DIR,
    portable_path,
    resolve_project_path,
)

CORE_SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
DEFAULT_INPUT = EORZEA_GLAMOUR_METADATA_DIR / "glamour_examples_raw.jsonl"
DEFAULT_OUTPUT = PROCESSED_DIR / "glamour_examples.jsonl"
DEFAULT_SUMMARY = PROCESSED_DIR / "glamour_examples_summary.json"


def build_glamour_examples_dataset(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    source_records = read_jsonl(input_path)
    rows = []
    missing_images = []
    slot_status = Counter()
    piece_match_status = Counter()
    for record in source_records:
        for item in record.get("equipment", []):
            if item.get("is_core_slot") and not item.get("is_empty"):
                piece_match_status[item.get("piece_match_type") or "legacy_unknown"] += 1
        resolved_slots = resolve_core_slots(record.get("equipment", []))
        for label in resolved_slots.values():
            slot_status[label["label_type"]] += 1
        for image in record.get("images", []):
            local_path = resolve_project_path(image.get("local_path"))
            if local_path is None or not local_path.is_file():
                missing_images.append(image.get("local_path"))
                continue
            with Image.open(local_path) as opened:
                width, height = opened.size
                image_format = (opened.format or "").lower()
                opened.verify()
            rows.append(
                {
                    "example_id": f"{record['glamour_id']}:{image['position']}",
                    "glamour_id": record["glamour_id"],
                    "title": record.get("title"),
                    "page_url": record.get("canonical_url") or record.get("page_url"),
                    "image_position": image["position"],
                    "source_image_url": image["source_image_url"],
                    "local_path": portable_path(local_path),
                    "width": width,
                    "height": height,
                    "image_format": image_format,
                    "model_race": record.get("model_race"),
                    "model_gender": record.get("model_gender"),
                    "intended_for": record.get("intended_for"),
                    "level_to_equip": record.get("level_to_equip"),
                    "tags": record.get("tags", []),
                    "description": record.get("description"),
                    "creator": record.get("creator"),
                    "core_slot_labels": resolved_slots,
                }
            )

    write_jsonl(output_path, rows)
    summary = {
        "source": portable_path(input_path),
        "output": portable_path(output_path),
        "glamour_records": len(source_records),
        "image_examples": len(rows),
        "missing_images": missing_images,
        "slot_label_status": dict(sorted(slot_status.items())),
        "piece_match_status": dict(sorted(piece_match_status.items())),
    }
    write_json(summary_path, summary)
    return summary


def resolve_core_slots(equipment: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    direct = {
        item["slot"]: item
        for item in equipment
        if item.get("is_core_slot") and not item.get("is_empty")
    }
    result = {}
    for slot in CORE_SLOTS:
        if slot in direct:
            result[slot] = label_from_item(direct[slot], "direct", slot)
            continue
        occupying = next(
            (
                item
                for item in direct.values()
                if slot in item.get("occupied_slots", []) and item.get("slot") != slot
            ),
            None,
        )
        if occupying:
            result[slot] = label_from_item(occupying, "occupied_by_multislot", occupying["slot"])
        else:
            result[slot] = {
                "label_type": "empty_or_unlabeled",
                "source_slot": slot,
                "item_name": None,
                "piece_ids": [],
                "piece_match_type": "empty",
                "matched_piece_names": [],
                "appearance_equivalence_rule": None,
                "dyes": [],
            }
    return result


def label_from_item(item: dict[str, Any], label_type: str, source_slot: str) -> dict[str, Any]:
    return {
        "label_type": label_type,
        "source_slot": source_slot,
        "item_name": item.get("item_name"),
        "piece_ids": item.get("matched_piece_ids", []),
        "piece_match_type": item.get("piece_match_type"),
        "matched_piece_names": item.get("matched_piece_names", []),
        "appearance_equivalence_rule": item.get("appearance_equivalence_rule"),
        "site_item_id": item.get("site_item_id"),
        "lodestone_url": item.get("lodestone_url"),
        "dyes": item.get("dyes", []),
    }
