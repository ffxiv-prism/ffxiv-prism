from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.paths import EORZEA_METADATA_DIR, GAME_DATA_RAW_DIR


def build_chinese_item_names(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Resolve exact English names only when the Chinese mapping is unambiguous."""
    names: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        # Outfit bundles can share the equipment's English name but have a
        # different Chinese name. They are not wearable candidate items.
        if row.get("equip_slot_category_id") == 0:
            continue
        english = str(row.get("name_en") or row.get("name") or "").strip()
        chinese = str(row.get("name_zh") or "").strip()
        if english and chinese:
            names[english].add(chinese)
    return {
        english: next(iter(chinese))
        for english, chinese in names.items()
        if len(chinese) == 1
    }


def load_chinese_item_names() -> dict[str, str]:
    # Candidate indexes are frozen model assets. Their embedded names can lag
    # behind the current catalogue, so resolve display names at report time.
    # Read afresh so a dataset update also takes effect in a long-running app.
    return build_chinese_item_names(
        row
        for path in (
            EORZEA_METADATA_DIR / "gear_pieces_raw.jsonl",
            GAME_DATA_RAW_DIR / "cafemaker_item_names_for_gear_pieces_raw.jsonl",
        )
        for row in read_jsonl(path)
    )


def enrich_report_item_names(
    payload: Mapping[str, Any],
    names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fill missing candidate and same-model names without changing rankings."""
    if names is None:
        names = load_chinese_item_names()

    def enrich(candidate: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(candidate)
        if not str(row.get("name_zh") or "").strip():
            english = str(row.get("name_en") or row.get("name") or "").strip()
            if chinese := names.get(english):
                row["name_zh"] = chinese
        if "appearance_alternatives" in row:
            row["appearance_alternatives"] = [
                enrich(alternative) for alternative in row["appearance_alternatives"]
            ]
        return row

    return {
        **payload,
        "slots": [
            {
                **slot,
                "candidates": [enrich(row) for row in slot.get("candidates", [])],
            }
            for slot in payload.get("slots", [])
        ],
    }
