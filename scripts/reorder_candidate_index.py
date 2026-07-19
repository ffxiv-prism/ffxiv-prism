from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import portable_path


DEFAULT_KEY_FIELDS = ("local_path", "candidate_crop")


def reorder_indices(
    source_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    *,
    key_fields: Sequence[str] = DEFAULT_KEY_FIELDS,
) -> np.ndarray:
    """Return source row indexes in reference order, with exact-set checks."""

    fields = tuple(key_fields)
    if not fields:
        raise ValueError("At least one key field is required")

    def key(row: dict[str, Any]) -> tuple[str, ...]:
        missing = [field for field in fields if field not in row]
        if missing:
            raise ValueError(f"Candidate row is missing key fields: {missing}")
        return tuple(str(row[field]) for field in fields)

    source_lookup: dict[tuple[str, ...], int] = {}
    for index, row in enumerate(source_rows):
        row_key = key(row)
        if row_key in source_lookup:
            raise ValueError(f"Duplicate source candidate key: {row_key}")
        source_lookup[row_key] = index

    reference_keys = [key(row) for row in reference_rows]
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError("Reference candidate keys are not unique")
    missing = [row_key for row_key in reference_keys if row_key not in source_lookup]
    if missing:
        raise ValueError(f"Source index is missing {len(missing)} reference candidates")
    extras = set(source_lookup) - set(reference_keys)
    if extras:
        raise ValueError(f"Source index has {len(extras)} extra candidates")
    return np.asarray([source_lookup[row_key] for row_key in reference_keys], dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reorder a candidate index to exactly match reference metadata."
    )
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--reference-metadata", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument(
        "--key-fields",
        nargs="+",
        default=DEFAULT_KEY_FIELDS,
    )
    args = parser.parse_args()

    if args.source_index.resolve() == args.output_index.resolve():
        raise ValueError("Source and output index directories must differ")
    if args.output_index.exists() and any(args.output_index.iterdir()):
        raise FileExistsError(f"Output index is not empty: {args.output_index}")

    source_rows = read_jsonl(args.source_index / "metadata.jsonl")
    reference_rows = read_jsonl(args.reference_metadata)
    source_embeddings = np.load(
        args.source_index / "embeddings.npy",
        mmap_mode="r",
    )
    if source_embeddings.shape[0] != len(source_rows):
        raise ValueError(
            "Embedding/metadata mismatch: "
            f"{source_embeddings.shape[0]} != {len(source_rows)}"
        )
    order = reorder_indices(
        source_rows,
        reference_rows,
        key_fields=args.key_fields,
    )
    reordered_rows = [source_rows[int(index)] for index in order]

    args.output_index.mkdir(parents=True, exist_ok=True)
    np.save(args.output_index / "embeddings.npy", source_embeddings[order])
    write_jsonl(args.output_index / "metadata.jsonl", reordered_rows)
    summary = {
        "source_index": portable_path(args.source_index),
        "reference_metadata": portable_path(args.reference_metadata),
        "output": portable_path(args.output_index),
        "key_fields": list(args.key_fields),
        "candidate_count": len(reordered_rows),
        "embedding_shape": list(source_embeddings.shape),
        "permutation_identity": bool(np.array_equal(order, np.arange(len(order)))),
    }
    write_json(args.output_index / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
