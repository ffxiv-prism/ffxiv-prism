from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ffxiv_prism.paths import GAME_DATA_RAW_DIR


def load_item_dye_counts(path: Path | None = None) -> tuple[dict[int, int], dict[str, int]]:
    """Read per-item dye channels; only resolve unambiguous wearable names."""
    if path is None:
        current = GAME_DATA_RAW_DIR / "cafemaker_item_names_raw.jsonl"
        if current.is_file():
            with current.open(encoding="utf-8-sig") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            # Refreshed data is authoritative, including explicit unknowns.
            # Only pre-dye snapshots fall back to the legacy local CSV.
            if any("dye_count" in row for row in rows):
                return _index_dye_rows(rows)
        path = GAME_DATA_RAW_DIR / "xiv_datamining_item_en.csv"
    if not path.is_file():
        return {}, {}
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            return _index_dye_rows(json.loads(line) for line in handle if line.strip())
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return _index_dye_rows({
            "game_item_id": row.get("#"),
            "equip_slot_category_id": row.get("EquipSlotCategory"),
            "dye_count": row.get("DyeCount"),
            "name_en": row.get("Name"),
        } for row in csv.DictReader(handle))


def _index_dye_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[int, int], dict[str, int]]:
    by_id: dict[int, int] = {}
    names: dict[str, set[int | None]] = defaultdict(set)
    for row in rows:
        try:
            item_id = int(row["game_item_id"])
            slot = int(row["equip_slot_category_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if item_id <= 0 or slot <= 0:
            continue
        try:
            value = row["dye_count"]
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError("Invalid dye channel count")
            count = int(value)
            if count < 0:
                count = None
        except (KeyError, TypeError, ValueError):
            count = None
        if count is not None:
            by_id[item_id] = count
        if name := str(row.get("name_en") or "").strip():
            names[name].add(count)
    by_name = {
        name: next(iter(counts))
        for name, counts in names.items()
        if len(counts) == 1 and None not in counts
    }
    return by_id, by_name


def enrich_report_dye_counts(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add display metadata without modifying saved candidates or rankings."""
    by_id, by_name = load_item_dye_counts()

    def enrich(candidate: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(candidate)
        item_id = row.get("game_item_id")
        if item_id is not None:
            try:
                count = by_id.get(int(item_id))
            except (TypeError, ValueError):
                count = None
        else:
            count = by_name.get(str(row.get("name_en") or row.get("name") or "").strip())
        row["dye_count"] = count
        if "appearance_alternatives" in row:
            row["appearance_alternatives"] = [enrich(item) for item in row["appearance_alternatives"]]
        return row

    return {
        **payload,
        "slots": [
            {**slot, "candidates": [enrich(row) for row in slot.get("candidates", [])]}
            for slot in payload.get("slots", [])
        ],
    }
