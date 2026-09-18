from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.paths import EORZEA_METADATA_DIR


def official_slots_by_name(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, frozenset[str]]:
    """An empty official slot set means non-armour, not missing evidence."""
    slots: dict[str, set[str]] = {}
    for row in rows:
        name = str(row.get("name_en") or row.get("name") or "").strip()
        if name and row.get("official_equip_slot_source"):
            slots.setdefault(name, set()).update(row.get("official_occupied_slots") or [])
    return {name: frozenset(values) for name, values in slots.items()}


def load_official_slots() -> dict[str, frozenset[str]]:
    return official_slots_by_name(
        read_jsonl(EORZEA_METADATA_DIR / "gear_pieces_raw.jsonl")
    )


def candidate_matches_slot(
    candidate: Mapping[str, Any], slot: str, official: Mapping[str, frozenset[str]],
) -> bool:
    name = str(candidate.get("name_en") or candidate.get("name") or "").strip()
    allowed = official.get(name)
    return allowed is None or slot in allowed


def filter_candidates_for_slot(
    candidates: Sequence[Mapping[str, Any]],
    slot: str,
    official: Mapping[str, frozenset[str]],
) -> list[dict[str, Any]]:
    output = []
    for candidate in candidates:
        if not candidate_matches_slot(candidate, slot, official):
            continue
        row = dict(candidate)
        if "appearance_alternatives" in row:
            row["appearance_alternatives"] = [
                dict(item) for item in row["appearance_alternatives"]
                if candidate_matches_slot(item, slot, official)
            ]
        row["rank"] = len(output) + 1
        output.append(row)
    return output
