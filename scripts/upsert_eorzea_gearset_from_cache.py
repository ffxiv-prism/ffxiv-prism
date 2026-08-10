from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.parser import (  # noqa: E402
    parse_gearset_detail,
)
from ffxiv_prism.jsonl import append_jsonl, read_jsonl  # noqa: E402
from ffxiv_prism.paths import EORZEA_HTML_DIR, EORZEA_METADATA_DIR  # noqa: E402


DEFAULT_XIV_ITEM_CSV = (
    PROJECT_ROOT / "data" / "raw" / "game_data" / "xiv_datamining_item_en.csv"
)
DEFAULT_XIV_ITEM_CN_DELTA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "game_data"
    / "xiv_datamining_item_cn_delta.jsonl"
)
DEFAULT_EQUIP_SLOT_CATEGORIES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "game_data"
    / "cafemaker_equip_slot_categories_raw.jsonl"
)
GAME_SLOT_TO_DATASET_SLOT = {
    "Head": "HEAD",
    "Body": "BODY",
    "Gloves": "HANDS",
    "Legs": "LEGS",
    "Feet": "FEET",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse one cached Eorzea gearset page and atomically replace that "
            "gearset's detail and piece rows without clearing the full dataset."
        )
    )
    parser.add_argument("slug", help="Eorzea Collection gearset slug.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=EORZEA_HTML_DIR / "gearset",
        help="Directory containing rendered or source gearset HTML.",
    )
    parser.add_argument(
        "--details",
        type=Path,
        default=EORZEA_METADATA_DIR / "gearset_details_raw.jsonl",
    )
    parser.add_argument(
        "--pieces",
        type=Path,
        default=EORZEA_METADATA_DIR / "gear_pieces_raw.jsonl",
    )
    parser.add_argument(
        "--xiv-item-csv",
        type=Path,
        default=DEFAULT_XIV_ITEM_CSV,
        help="Current XIV Item.csv used for exact English-name game item IDs.",
    )
    parser.add_argument(
        "--xiv-item-cn-delta",
        type=Path,
        default=DEFAULT_XIV_ITEM_CN_DELTA,
        help="Audited Chinese-name delta JSONL keyed by game_item_id.",
    )
    parser.add_argument(
        "--equip-slot-categories",
        type=Path,
        default=DEFAULT_EQUIP_SLOT_CATEGORIES,
        help="EquipSlotCategory JSONL used to derive official occupied slots.",
    )
    args = parser.parse_args()

    cache_path = args.cache_dir / f"{args.slug}.html"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Gearset cache does not exist: {cache_path}")
    page_url = f"https://ffxiv.eorzeacollection.com/gearset/{args.slug}"
    record = parse_gearset_detail(
        cache_path.read_text(encoding="utf-8", errors="replace"),
        page_url,
    )
    if record.slug != args.slug:
        raise ValueError(
            f"Parsed slug {record.slug!r} does not match requested {args.slug!r}"
        )
    if not record.pieces:
        raise ValueError(
            f"Refusing to replace {args.slug}: parsed equipment list is empty"
        )

    detail_rows = read_jsonl(args.details)
    old_detail_count = sum(
        1 for row in detail_rows if str(row.get("slug")) == args.slug
    )
    detail_payload = record.to_dict()
    # Canonical image records are maintained by the dedicated URL/download
    # manifests; detail collection intentionally keeps this field lightweight.
    detail_payload["images"] = []
    updated_details = replace_matching_rows(
        detail_rows,
        lambda row: str(row.get("slug")) == args.slug,
        [detail_payload],
    )

    piece_rows = read_jsonl(args.pieces)
    old_piece_count = sum(
        1 for row in piece_rows if str(row.get("gearset_slug")) == args.slug
    )
    xiv_items = load_xiv_items(args.xiv_item_csv)
    cn_names = load_cn_names(args.xiv_item_cn_delta)
    equip_slot_categories = load_equip_slot_categories(args.equip_slot_categories)
    new_piece_rows = []
    for piece in record.pieces:
        row = {
            "gearset_slug": record.slug,
            "source_page_url": record.source_page_url,
            "title": record.title,
            "present_slots": record.present_slots,
            "missing_slots": record.missing_slots,
            **piece.to_dict(),
        }
        enrich_piece_with_xiv_item(
            row,
            xiv_items,
            cn_names=cn_names,
            equip_slot_categories=equip_slot_categories,
        )
        new_piece_rows.append(row)
    updated_pieces = replace_matching_rows(
        piece_rows,
        lambda row: str(row.get("gearset_slug")) == args.slug,
        new_piece_rows,
    )

    atomic_write_jsonl(args.details, updated_details)
    atomic_write_jsonl(args.pieces, updated_pieces)
    print(
        json.dumps(
            {
                "slug": args.slug,
                "cache": str(cache_path),
                "detail_rows_before": old_detail_count,
                "detail_rows_after": 1,
                "piece_rows_before": old_piece_count,
                "piece_rows_after": len(new_piece_rows),
                "piece_names": [row["name"] for row in new_piece_rows],
                "game_item_ids": {
                    str(row["name"]): row.get("game_item_ids") or []
                    for row in new_piece_rows
                },
                "details_total": len(updated_details),
                "pieces_total": len(updated_pieces),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def replace_matching_rows(
    rows: list[dict[str, object]],
    predicate: object,
    replacements: list[dict[str, object]],
) -> list[dict[str, object]]:
    matches = [
        index
        for index, row in enumerate(rows)
        if callable(predicate) and predicate(row)
    ]
    insert_at = matches[0] if matches else len(rows)
    output = [
        row
        for row in rows
        if not (callable(predicate) and predicate(row))
    ]
    output[insert_at:insert_at] = replacements
    return output


def load_xiv_items(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.is_file():
        return {}
    output: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("Name") or "").strip()
            if not name:
                continue
            try:
                item_id = int(str(row.get("#") or ""))
            except ValueError:
                continue
            output[normalize_name(name)].append(
                {
                    "game_item_id": item_id,
                    "equip_slot_category_id": normalize_int(
                        row.get("EquipSlotCategory")
                    ),
                    "item_ui_category_id": normalize_int(
                        row.get("ItemUICategory")
                    ),
                    "description": str(row.get("Description") or "").strip()
                    or None,
                }
            )
    return dict(output)


def load_cn_names(path: Path) -> dict[int, dict[str, object]]:
    if not path.is_file():
        return {}
    output: dict[int, dict[str, object]] = {}
    for row in read_jsonl(path):
        item_id = normalize_int(row.get("game_item_id"))
        if item_id is not None:
            output[item_id] = row
    return output


def load_equip_slot_categories(path: Path) -> dict[int, dict[str, int]]:
    if not path.is_file():
        return {}
    output: dict[int, dict[str, int]] = {}
    for row in read_jsonl(path):
        category_id = normalize_int(row.get("equip_slot_category_id"))
        if category_id is None:
            continue
        output[category_id] = {
            key: normalize_int(row.get(key)) or 0
            for key in GAME_SLOT_TO_DATASET_SLOT
        }
    return output


def enrich_piece_with_xiv_item(
    row: dict[str, object],
    xiv_items: dict[str, list[dict[str, object]]],
    *,
    cn_names: dict[int, dict[str, object]],
    equip_slot_categories: dict[int, dict[str, int]],
) -> None:
    name = str(row.get("name") or "").strip()
    matches = xiv_items.get(normalize_name(name), [])
    item_ids = sorted(
        {
            int(match["game_item_id"])
            for match in matches
            if match.get("game_item_id") is not None
        }
    )
    row["name_en"] = name or None
    row["game_item_id"] = item_ids[0] if len(item_ids) == 1 else None
    row["game_item_ids"] = item_ids
    cn_match = cn_names.get(item_ids[0]) if len(item_ids) == 1 else None
    name_zh = str(cn_match.get("name_zh") or "").strip() if cn_match else ""
    row["name_zh"] = name_zh or None
    row["name_zh_source"] = cn_match.get("source") if name_zh else None
    row["name_zh_match_status"] = "matched" if name_zh else "unmapped"
    row["name_zh_candidates"] = [name_zh] if name_zh else []

    category_ids = sorted(
        {
            int(match["equip_slot_category_id"])
            for match in matches
            if match.get("equip_slot_category_id") is not None
        }
    )
    category_id = category_ids[0] if len(category_ids) == 1 else None
    category = equip_slot_categories.get(category_id) if category_id else None
    primary_slots = (
        [
            dataset_slot
            for game_slot, dataset_slot in GAME_SLOT_TO_DATASET_SLOT.items()
            if category.get(game_slot) == 1
        ]
        if category
        else []
    )
    blocked_slots = (
        [
            dataset_slot
            for game_slot, dataset_slot in GAME_SLOT_TO_DATASET_SLOT.items()
            if category.get(game_slot) == -1
        ]
        if category
        else []
    )
    descriptions = sorted(
        {
            str(match["description"])
            for match in matches
            if match.get("description")
        }
    )
    item_ui_category_ids = sorted(
        {
            int(match["item_ui_category_id"])
            for match in matches
            if match.get("item_ui_category_id") is not None
        }
    )

    row["visual_inferred_occupied_slots"] = list(row.get("occupied_slots") or [])
    row["visual_inferred_blocked_slots"] = list(row.get("blocked_slots") or [])
    row["visual_inferred_conflict_note"] = row.get("equip_conflict_note")
    row["visual_inferred_conflict_source"] = row.get("equip_conflict_source")
    row["official_slot_category_id"] = category_id
    row["official_item_ui_category"] = (
        str(item_ui_category_ids[0]) if len(item_ui_category_ids) == 1 else None
    )
    row["official_primary_slots"] = primary_slots
    row["official_blocked_slots"] = blocked_slots
    row["official_occupied_slots"] = unique([*primary_slots, *blocked_slots])
    row["official_equip_slot_source"] = (
        "xivapi/ffxiv-datamining Item.csv + CafeMaker EquipSlotCategory"
        if category
        else None
    )
    if category:
        row["occupied_slots"] = row["official_occupied_slots"]
        row["blocked_slots"] = blocked_slots
        row["equip_conflict_note"] = descriptions[0] if descriptions else None
        row["equip_conflict_source"] = (
            "https://github.com/xivapi/ffxiv-datamining/blob/master/csv/Item.csv"
            if descriptions or blocked_slots
            else None
        )


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def normalize_name(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def normalize_int(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("", encoding="utf-8")
    append_jsonl(temporary, rows)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
