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

from ffxiv_prism.boilmaster import (  # noqa: E402
    DEFAULT_API_ROOT,
    fetch_sheet_page,
    next_after,
    row_fields,
    row_id,
)
from ffxiv_prism.jsonl import write_json_atomic, write_jsonl_atomic  # noqa: E402
from ffxiv_prism.paths import GAME_DATA_RAW_DIR  # noqa: E402


DEFAULT_BASE_URL = f"{DEFAULT_API_ROOT}/EquipSlotCategory"
DEFAULT_OUTPUT = GAME_DATA_RAW_DIR / "cafemaker_equip_slot_categories_raw.jsonl"
DEFAULT_SUMMARY = GAME_DATA_RAW_DIR / "cafemaker_equip_slot_categories_summary.json"
EQUIP_SLOT_FIELDS = (
    "MainHand",
    "OffHand",
    "Head",
    "Body",
    "Gloves",
    "Legs",
    "Feet",
    "Ears",
    "Neck",
    "Wrists",
    "FingerL",
    "FingerR",
    "SoulCrystal",
    "Waist",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache Boilmaster EquipSlotCategory rows.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Boilmaster EquipSlotCategory sheet endpoint.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Output summary JSON path.")
    parser.add_argument("--page-size", type=int, default=1000, help="Rows requested per page.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Network timeout per request.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per page.")
    args = parser.parse_args()
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive")

    rows: list[dict[str, Any]] = []
    after: int | None = None
    pages_fetched = 0
    schema: str | None = None
    game_version: str | None = None
    while True:
        payload = fetch_sheet_page(
            args.base_url,
            fields=EQUIP_SLOT_FIELDS,
            after=after,
            limit=args.page_size,
            timeout=args.timeout,
            retries=args.retries,
        )
        page_rows = payload["rows"]
        if not page_rows:
            break
        pages_fetched += 1
        schema = str(payload.get("schema") or "") or schema
        game_version = str(payload.get("version") or "") or game_version
        rows.extend(normalize_row(row, source=args.base_url) for row in page_rows)
        after = next_after(page_rows, after)

    if not rows:
        raise RuntimeError(
            "Boilmaster returned no EquipSlotCategory rows; "
            f"refusing to replace existing output: {args.output}"
        )
    write_jsonl_atomic(args.output, rows)

    summary = {
        "source": args.base_url,
        "fields": list(EQUIP_SLOT_FIELDS),
        "output": str(args.output),
        "rows_written": len(rows),
        "pages_fetched": pages_fetched,
        "last_row_id": after,
        "schema": schema,
        "game_version": game_version,
        "finished_at": timestamp(),
    }
    write_json_atomic(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def normalize_row(row: object, source: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    fields = row_fields(row)
    normalized = {"equip_slot_category_id": row_id(row), "source": source}
    for field in EQUIP_SLOT_FIELDS:
        normalized[field] = normalize_int(fields.get(field)) or 0
    return normalized


def normalize_int(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
