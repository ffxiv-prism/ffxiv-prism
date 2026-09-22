from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from ffxiv_prism.paths import GAME_DATA_RAW_DIR


def load_item_dye_counts(path: Path | None = None) -> tuple[dict[int, int], dict[str, int]]:
    """Read per-item dye channels; only resolve unambiguous wearable names."""
    path = path if path is not None else GAME_DATA_RAW_DIR / "xiv_datamining_item_en.csv"
    by_id: dict[int, int] = {}
    names: dict[str, set[int | None]] = defaultdict(set)
    if not path.is_file():
        return {}, {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                item_id = int(row["#"])
                slot = int(row["EquipSlotCategory"])
            except (KeyError, TypeError, ValueError):
                continue
            if item_id <= 0 or slot <= 0:
                continue
            try:
                count = int(row["DyeCount"])
                if count < 0:
                    count = None
            except (KeyError, TypeError, ValueError):
                count = None
            if count is not None:
                by_id[item_id] = count
            if name := str(row.get("Name") or "").strip():
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
