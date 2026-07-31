from __future__ import annotations

import argparse
import hashlib
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


def stable_sample_key(row: Mapping[str, Any], seed: str) -> tuple[str, str]:
    example_id = str(row["example_id"])
    digest = hashlib.sha256(f"{seed}|{example_id}".encode("utf-8")).hexdigest()
    return digest, example_id


def select_rule_negative_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: str,
    excluded_example_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    excluded = excluded_example_ids or set()
    candidates = [
        row
        for row in rows
        if row.get("triage_for_review") is False
        and str(row.get("example_id")) not in excluded
    ]
    selected: list[dict[str, Any]] = []
    seen_glamours: set[str] = set()
    for row in sorted(candidates, key=lambda value: stable_sample_key(value, seed)):
        glamour_id = str(row["glamour_id"])
        if glamour_id in seen_glamours:
            continue
        selected.append(dict(row))
        seen_glamours.add(glamour_id)
        if len(selected) == sample_size:
            break
    if len(selected) < sample_size:
        raise ValueError(
            f"requested {sample_size} rule-negative glamours, found "
            f"{len(selected)} after exclusions"
        )
    return selected


def reviewed_example_ids(paths: Sequence[Path]) -> set[str]:
    return {
        str(row["example_id"])
        for path in paths
        for row in read_jsonl(path)
        if row.get("example_id") is not None
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic, one-image-per-glamour audit sample from "
            "FEET rows not queued by the semantic visibility triage rule."
        )
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--exclude-reviews", type=Path, action="append", default=[])
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", default="feet-rule-negative-audit-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.features)
    excluded_ids = reviewed_example_ids(args.exclude_reviews)
    selected = select_rule_negative_sample(
        rows,
        sample_size=args.sample_size,
        seed=args.seed,
        excluded_example_ids=excluded_ids,
    )
    for rank, row in enumerate(selected, start=1):
        row.update(
            {
                "sample_rank": rank,
                "slot": "FEET",
                "visibility": None,
                "usability": None,
                "issue": None,
                "review_source": "manual_parser_negative_audit",
            }
        )
    write_jsonl(args.output, selected)
    summary = {
        "version": 1,
        "features": portable_path(args.features),
        "seed": args.seed,
        "sample_size": len(selected),
        "sample_glamours": len({row["glamour_id"] for row in selected}),
        "excluded_review_files": [portable_path(path) for path in args.exclude_reviews],
        "excluded_example_ids": len(excluded_ids),
        "output": portable_path(args.output),
        "policy": (
            "Independent residual-risk audit only; rule-negative predictions "
            "must not be treated as usable without manual review."
        ),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
