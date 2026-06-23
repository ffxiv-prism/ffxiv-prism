from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import read_jsonl, write_json  # noqa: E402
from ffxiv_prism.paths import EORZEA_METADATA_DIR, GAME_DATA_RAW_DIR  # noqa: E402


DEFAULT_PIECES = EORZEA_METADATA_DIR / "gear_pieces_raw.jsonl"
DEFAULT_MAPPING = GAME_DATA_RAW_DIR / "cafemaker_item_names_raw.jsonl"
DEFAULT_OUTPUT = GAME_DATA_RAW_DIR / "cafemaker_item_names_for_gear_pieces_raw.jsonl"
DEFAULT_SUMMARY = GAME_DATA_RAW_DIR / "cafemaker_item_names_for_gear_pieces_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a slim CafeMaker item-name mapping containing only gear piece item names."
    )
    parser.add_argument("--pieces", type=Path, default=DEFAULT_PIECES, help="Gear piece JSONL.")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="Full CafeMaker Item mapping JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Filtered mapping output JSONL.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Summary JSON path.")
    args = parser.parse_args()

    needed_names = needed_english_names(read_jsonl(args.pieces))
    mapping_rows = read_jsonl(args.mapping)
    output_rows, matched_names = filter_mapping(mapping_rows, needed_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    unmatched_names = sorted(needed_names - matched_names)
    summary = {
        "pieces": str(args.pieces),
        "full_mapping": str(args.mapping),
        "output": str(args.output),
        "needed_unique_english_names": len(needed_names),
        "matched_unique_english_names": len(matched_names),
        "unmatched_unique_english_names": len(unmatched_names),
        "full_mapping_rows": len(mapping_rows),
        "filtered_mapping_rows": len(output_rows),
        "unmatched_name_examples": unmatched_names[:50],
        "finished_at": timestamp(),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def needed_english_names(rows: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for row in rows:
        name = str(row.get("name") or row.get("name_en") or "").strip()
        if name:
            names.add(normalize_key(name))
    return names


def filter_mapping(
    rows: list[dict[str, Any]],
    needed_names: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    output_rows: list[dict[str, Any]] = []
    matched_names: set[str] = set()
    seen_rows: set[tuple[int | None, str, str, int | None, str | None]] = set()
    for row in rows:
        name_en = str(row.get("name_en") or "").strip()
        name_zh = str(row.get("name_zh") or "").strip()
        if not name_en or not name_zh:
            continue
        normalized = normalize_key(name_en)
        if normalized not in needed_names:
            continue
        item_id = normalize_item_id(row.get("game_item_id"))
        equip_slot_category_id = normalize_item_id(row.get("equip_slot_category_id"))
        item_ui_category_en = str(row.get("item_ui_category_en") or "").strip() or None
        dedupe_key = (item_id, name_en, name_zh, equip_slot_category_id, item_ui_category_en)
        if dedupe_key in seen_rows:
            continue
        seen_rows.add(dedupe_key)
        matched_names.add(normalized)
        output_rows.append(
            {
                "equip_slot_category_id": equip_slot_category_id,
                "game_item_id": item_id,
                "item_ui_category_en": item_ui_category_en,
                "name_en": name_en,
                "name_zh": name_zh,
                "source": row.get("source") or "https://cafemaker.wakingsands.com/Item",
            }
        )
    output_rows.sort(key=lambda row: (str(row["name_en"]).casefold(), row["game_item_id"] or 0))
    return output_rows, matched_names


def normalize_item_id(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalize_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
