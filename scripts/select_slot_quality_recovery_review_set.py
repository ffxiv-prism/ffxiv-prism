from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402


DEFAULT_TARGETS = {"HEAD": 32, "BODY": 28, "HANDS": 12}
SLOT_WEIGHTS = {"HEAD": 3.0, "BODY": 2.0, "HANDS": 1.0}


def conversion_keys(
    row: Mapping[str, Any],
    targets: Mapping[str, int],
) -> set[tuple[str, int]]:
    return {
        (slot, int(row["slot_details"][slot]["appearance_identity"]))
        for slot in row.get("single_to_multi_conversion_slots") or []
        if slot in targets
    }


def select_review_set(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Mapping[str, int],
    require_roegadyn_head: bool = True,
) -> list[dict[str, Any]]:
    if any(value < 1 for value in targets.values()):
        raise ValueError("all targets must be positive")
    candidates = [dict(row) for row in rows if conversion_keys(row, targets)]
    selected: list[dict[str, Any]] = []
    selected_glamours: set[str] = set()
    covered: set[tuple[str, int]] = set()
    has_roegadyn_head = False
    while True:
        counts = {
            slot: sum(key[0] == slot for key in covered) for slot in targets
        }
        targets_met = all(counts[slot] >= targets[slot] for slot in targets)
        race_met = not require_roegadyn_head or has_roegadyn_head
        if targets_met and race_met:
            break

        best_row = None
        best_key = None
        for row in candidates:
            glamour_id = str(row["glamour_id"])
            if glamour_id in selected_glamours:
                continue
            keys = conversion_keys(row, targets)
            new_keys = {
                key
                for key in keys - covered
                if counts[key[0]] < targets[key[0]]
            }
            race_bonus = (
                20.0
                if require_roegadyn_head
                and not has_roegadyn_head
                and row.get("model_race") == "Roegadyn"
                and any(key[0] == "HEAD" for key in keys)
                else 0.0
            )
            gain = sum(SLOT_WEIGHTS.get(slot, 1.0) for slot, _ in new_keys)
            if gain <= 0.0 and race_bonus <= 0.0:
                continue
            score = gain + race_bonus
            key = (
                score,
                len(new_keys),
                float(row.get("quality_score") or 0.0),
                -len(str(row["example_id"])),
                str(row["example_id"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
        if best_row is None:
            raise ValueError(
                f"cannot satisfy conversion targets; covered={counts}, "
                f"roegadyn_head={has_roegadyn_head}"
            )
        selected.append(best_row)
        selected_glamours.add(str(best_row["glamour_id"]))
        keys = conversion_keys(best_row, targets)
        covered.update(keys)
        has_roegadyn_head = has_roegadyn_head or (
            best_row.get("model_race") == "Roegadyn"
            and any(key[0] == "HEAD" for key in keys)
        )
    return selected


def parse_target(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("target must be SLOT=COUNT")
    slot, count_text = value.split("=", 1)
    try:
        count = int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target count must be an integer") from exc
    if count < 1:
        raise argparse.ArgumentTypeError("target count must be positive")
    return slot.upper(), count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Greedily select one image per glamour for a compact manual "
            "single-to-multi appearance recovery review set."
        )
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--target", type=parse_target, action="append")
    parser.add_argument(
        "--require-roegadyn-head",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    targets = dict(args.target or DEFAULT_TARGETS.items())
    unknown = sorted(set(targets) - set(SLOT_WEIGHTS))
    if unknown:
        parser.error(f"unsupported target slots: {unknown}")

    selected = select_review_set(
        read_jsonl(args.candidates),
        targets=targets,
        require_roegadyn_head=args.require_roegadyn_head,
    )
    write_jsonl(args.output, selected)
    covered = {
        key for row in selected for key in conversion_keys(row, targets)
    }
    summary = {
        "version": 1,
        "candidates": portable_path(args.candidates),
        "selected_images": len(selected),
        "selected_glamours": len({row["glamour_id"] for row in selected}),
        "targets": targets,
        "covered_conversion_identities": {
            slot: sum(key[0] == slot for key in covered) for slot in targets
        },
        "roegadyn_head_selected": any(
            row.get("model_race") == "Roegadyn"
            and "HEAD" in row.get("single_to_multi_conversion_slots", [])
            for row in selected
        ),
        "output": portable_path(args.output),
        "policy": (
            "Queue compaction only. Every selected image and target slot "
            "still requires a manual decision before recovery."
        ),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
