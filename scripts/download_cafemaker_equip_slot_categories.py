from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import append_jsonl, write_json  # noqa: E402
from ffxiv_prism.paths import GAME_DATA_RAW_DIR  # noqa: E402


DEFAULT_BASE_URL = "https://cafemaker.wakingsands.com/EquipSlotCategory"
DEFAULT_OUTPUT = GAME_DATA_RAW_DIR / "cafemaker_equip_slot_categories_raw.jsonl"
DEFAULT_SUMMARY = GAME_DATA_RAW_DIR / "cafemaker_equip_slot_categories_summary.json"
EQUIP_SLOT_COLUMNS = [
    "ID",
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
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache CafeMaker EquipSlotCategory rows.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="CafeMaker EquipSlotCategory endpoint.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Output summary JSON path.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Network timeout.")
    args = parser.parse_args()

    payload = fetch_rows(args.base_url, args.timeout)
    rows = [normalize_row(row) for row in payload.get("Results") or [] if isinstance(row, dict)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")
    append_jsonl(args.output, rows)

    summary = {
        "source": args.base_url,
        "columns": EQUIP_SLOT_COLUMNS,
        "output": str(args.output),
        "rows_written": len(rows),
        "finished_at": timestamp(),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def fetch_rows(base_url: str, timeout: float) -> dict[str, Any]:
    query = urlencode({"columns": ",".join(EQUIP_SLOT_COLUMNS), "limit": "1000"})
    request = Request(f"{base_url}?{query}", headers={"User-Agent": "ffxiv-prism/0.1 equip-slot-cache"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("CafeMaker response must be a JSON object")
    return payload


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {"equip_slot_category_id": normalize_int(row.get("ID")), "source": DEFAULT_BASE_URL}
    for column in EQUIP_SLOT_COLUMNS:
        if column == "ID":
            continue
        normalized[column] = normalize_int(row.get(column)) or 0
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
