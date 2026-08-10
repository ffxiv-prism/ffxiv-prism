from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import append_jsonl, write_json


DEFAULT_INPUT = Path("data/raw/game_data/xiv_datamining_item_en.csv")
DEFAULT_OUTPUT = Path(
    "data/raw/game_data/xiv_datamining_item_models_raw.jsonl"
)
DEFAULT_SUMMARY = Path(
    "data/raw/game_data/xiv_datamining_item_models_summary.json"
)
SOURCE = (
    "https://github.com/xivapi/ffxiv-datamining/"
    "blob/master/csv/en/Item.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize XIV datamining Item.csv model fields for the "
            "HuijiWiki same-model rule."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = read_item_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")
    append_jsonl(args.output, rows)
    summary = {
        "source": SOURCE,
        "input": args.input.as_posix(),
        "output": args.output.as_posix(),
        "rows_written": len(rows),
        "rows_with_names": sum(bool(row["name_en"]) for row in rows),
        "rows_with_nonzero_model_main": sum(
            bool(row["model_main"] and row["model_main"][0])
            for row in rows
        ),
        "finished_at": timestamp(),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def read_item_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "#",
            "Name",
            "ModelMain",
            "ModelSub",
            "ItemUICategory",
            "EquipSlotCategory",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Item.csv missing columns: {sorted(missing)}")
        for raw in reader:
            item_id = parse_int(raw.get("#"))
            if item_id is None:
                continue
            rows.append(
                {
                    "game_item_id": item_id,
                    "name_en": normalize_text(raw.get("Name")),
                    "name_zh": None,
                    "model_main": unpack_model(raw.get("ModelMain")),
                    "model_sub": unpack_model(raw.get("ModelSub")),
                    "equip_slot_category_id": parse_int(
                        raw.get("EquipSlotCategory")
                    ),
                    "item_ui_category_id": parse_int(
                        raw.get("ItemUICategory")
                    ),
                    "source": SOURCE,
                }
            )
    return rows


def unpack_model(value: Any) -> list[int]:
    packed = parse_int(value)
    if packed is None:
        return []
    return [
        (packed >> (16 * index)) & 0xFFFF
        for index in range(4)
    ]


def parse_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
