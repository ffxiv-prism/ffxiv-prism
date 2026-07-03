from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def inner_fold(glamour_id: str | int, folds: int, seed: int) -> int:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    digest = hashlib.sha256(
        f"inner-cv:{seed}:{glamour_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % folds


def build_fold_assignments(
    base_assignments: dict[str, str],
    *,
    fold: int,
    folds: int,
    seed: int,
) -> dict[str, str]:
    if fold < 0 or fold >= folds:
        raise ValueError("fold must be in [0, folds)")
    output = {}
    for glamour_id, role in base_assignments.items():
        if role != "train":
            output[str(glamour_id)] = "blind_test"
            continue
        output[str(glamour_id)] = (
            "validation"
            if inner_fold(glamour_id, folds, seed) == fold
            else "train"
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build grouped inner-CV manifests using official train groups only. "
            "Every non-train group is excluded as blind_test."
        )
    )
    parser.add_argument("--base-split", type=Path, required=True)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        assignments = build_fold_assignments(
            base_assignments,
            fold=fold,
            folds=args.folds,
            seed=args.seed,
        )
        counts = {
            role: sum(value == role for value in assignments.values())
            for role in ("train", "validation", "blind_test")
        }
        payload = {
            "version": 1,
            "policy": (
                "Inner grouped cross-validation over official train groups only; "
                "official validation and sealed groups are excluded."
            ),
            "source_split": args.base_split.as_posix(),
            "seed": args.seed,
            "fold": fold,
            "folds": args.folds,
            "counts": counts,
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
