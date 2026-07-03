from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl


def body_item_key(example: dict) -> str | None:
    body = (example.get("core_slot_labels") or {}).get("BODY") or {}
    piece_ids = sorted(int(value) for value in body.get("piece_ids", []))
    if piece_ids:
        return "pieces:" + ",".join(str(value) for value in piece_ids)
    item_name = str(body.get("item_name") or "").strip().casefold()
    return f"name:{item_name}" if item_name else None


def item_fold(item_key: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(
        f"inner-item-cv:{seed}:{item_key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % folds


def build_balanced_item_fold_map(
    item_weights: dict[str, int],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    """Greedily balance eligible image counts without splitting item identities."""
    loads = [0] * folds
    output: dict[str, int] = {}
    ordered_items = sorted(
        item_weights,
        key=lambda key: (
            -item_weights[key],
            hashlib.sha256(
                f"inner-item-cv-order:{seed}:{key}".encode("utf-8")
            ).digest(),
        ),
    )
    for item_key in ordered_items:
        minimum = min(loads)
        candidates = [
            fold_index
            for fold_index, load in enumerate(loads)
            if load == minimum
        ]
        selected = candidates[item_fold(item_key, len(candidates), seed)]
        output[item_key] = selected
        loads[selected] += item_weights[item_key]
    return output


def build_item_disjoint_assignments(
    base_assignments: dict[str, str],
    item_keys: dict[str, str],
    *,
    fold: int,
    folds: int,
    seed: int,
    item_fold_map: dict[str, int] | None = None,
) -> dict[str, str]:
    output = {}
    for glamour_id, role in base_assignments.items():
        if role != "train" or glamour_id not in item_keys:
            output[glamour_id] = "blind_test"
            continue
        selected_fold = (
            item_fold_map[item_keys[glamour_id]]
            if item_fold_map is not None
            else item_fold(item_keys[glamour_id], folds, seed)
        )
        output[glamour_id] = (
            "validation"
            if selected_fold == fold
            else "train"
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-only CV manifests that keep every BODY item identity "
            "entirely in one fold."
        )
    )
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument(
        "--quality-manifest",
        type=Path,
        help=(
            "Optional JSONL quality manifest. When supplied, BODY identities "
            "are greedily balanced by passed train image count."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    base = json.loads(args.base_split.read_text(encoding="utf-8"))
    base_assignments = {
        str(key): str(value)
        for key, value in base.get("assignments", {}).items()
    }
    accepted_example_ids: set[str] | None = None
    if args.quality_manifest:
        accepted_example_ids = {
            str(row["example_id"])
            for row in read_jsonl(args.quality_manifest)
            if row.get("passed")
        }

    item_keys: dict[str, str] = {}
    item_weights: dict[str, int] = {}
    for example in read_jsonl(args.examples):
        glamour_id = str(example["glamour_id"])
        key = body_item_key(example)
        if key is None:
            continue
        previous = item_keys.setdefault(glamour_id, key)
        if previous != key:
            raise ValueError(
                f"glamour {glamour_id} has inconsistent BODY identities: "
                f"{previous!r} vs {key!r}"
            )
        if (
            accepted_example_ids is not None
            and str(example.get("example_id")) in accepted_example_ids
            and base_assignments.get(glamour_id) == "train"
        ):
            item_weights[key] = item_weights.get(key, 0) + 1

    item_fold_map = None
    if accepted_example_ids is not None:
        all_item_keys = set(item_keys.values())
        item_weights = {
            key: item_weights.get(key, 0)
            for key in all_item_keys
        }
        item_fold_map = build_balanced_item_fold_map(
            item_weights,
            folds=args.folds,
            seed=args.seed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        assignments = build_item_disjoint_assignments(
            base_assignments,
            item_keys,
            fold=fold,
            folds=args.folds,
            seed=args.seed,
            item_fold_map=item_fold_map,
        )
        counts = {
            role: sum(value == role for value in assignments.values())
            for role in ("train", "validation", "blind_test")
        }
        validation_item_keys = {
            item_key
            for item_key, item_fold_index in (item_fold_map or {}).items()
            if item_fold_index == fold
        }
        payload = {
            "version": 2 if item_fold_map is not None else 1,
            "policy": (
                "Inner CV over official train only; all glamour groups sharing "
                "a BODY piece identity remain in one fold."
            ),
            "source_split": args.base_split.as_posix(),
            "source_examples": args.examples.as_posix(),
            "source_quality_manifest": (
                args.quality_manifest.as_posix()
                if args.quality_manifest
                else None
            ),
            "seed": args.seed,
            "fold": fold,
            "folds": args.folds,
            "counts": counts,
            "eligible_validation_image_count": (
                sum(item_weights[key] for key in validation_item_keys)
                if item_fold_map is not None
                else None
            ),
            "validation_item_identity_count": (
                len(validation_item_keys)
                if item_fold_map is not None
                else None
            ),
            "assignments": assignments,
        }
        output = args.output_dir / f"fold-{fold + 1:02d}.json"
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"fold={fold + 1}/{args.folds} counts={counts} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
