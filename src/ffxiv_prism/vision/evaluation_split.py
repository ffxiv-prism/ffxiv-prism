from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

DEFAULT_SPLIT_SEED = 20260627


def stable_new_group_role(
    glamour_id: str | int,
    seed: int = DEFAULT_SPLIT_SEED,
) -> str:
    digest = hashlib.sha256(f"expansion:{seed}:{glamour_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < 0.67:
        return "train"
    if value < 0.84:
        return "validation"
    return "blind_test"


def load_split_assignments(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = {
        str(glamour_id): str(role)
        for glamour_id, role in payload.get("assignments", {}).items()
    }
    valid = {
        "train",
        "validation",
        "development_test",
        "blind_test",
        "confirmation_test",
    }
    invalid = sorted(set(assignments.values()) - valid)
    if invalid:
        raise ValueError(f"Invalid split roles: {invalid}")
    return assignments


def development_split_map(
    glamour_ids: Iterable[str | int],
    assignments: dict[str, str],
) -> dict[str, str]:
    mapped = {}
    for value in glamour_ids:
        glamour_id = str(value)
        role = assignments.get(glamour_id)
        if role == "development_test":
            mapped[glamour_id] = "test"
        elif role in {"train", "validation"}:
            mapped[glamour_id] = role
    return mapped
