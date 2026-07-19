from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePath
from typing import Sequence

import numpy as np


def path_suffix_key(path: str, suffix_parts: int) -> tuple[str, ...]:
    if suffix_parts < 1:
        raise ValueError("suffix_parts must be positive")
    parts = PurePath(path).parts
    if len(parts) < suffix_parts:
        raise ValueError(f"Path has fewer than {suffix_parts} parts: {path}")
    return tuple(part.casefold() for part in parts[-suffix_parts:])


def reference_order(
    source_paths: Sequence[str],
    reference_paths: Sequence[str],
    *,
    suffix_parts: int = 3,
) -> np.ndarray:
    """Map source embeddings into reference path order by checked suffix keys."""

    source_lookup: dict[tuple[str, ...], int] = {}
    for index, path in enumerate(source_paths):
        key = path_suffix_key(str(path), suffix_parts)
        if key in source_lookup:
            raise ValueError(f"Duplicate source path suffix: {key}")
        source_lookup[key] = index
    reference_keys = [
        path_suffix_key(str(path), suffix_parts)
        for path in reference_paths
    ]
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError("Reference path suffixes are not unique")
    missing = [key for key in reference_keys if key not in source_lookup]
    if missing:
        raise ValueError(f"Source cache is missing {len(missing)} reference paths")
    extras = set(source_lookup) - set(reference_keys)
    if extras:
        raise ValueError(f"Source cache has {len(extras)} extra paths")
    return np.asarray([source_lookup[key] for key in reference_keys], dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Relabel and reorder a query embedding cache to reference crop paths "
            "after validating stable path suffixes."
        )
    )
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--suffix-parts", type=int, default=3)
    args = parser.parse_args()

    if args.source_cache.resolve() == args.output_cache.resolve():
        raise ValueError("Source and output caches must differ")
    source = np.load(args.source_cache, allow_pickle=False)
    reference = np.load(args.reference_cache, allow_pickle=False)
    source_paths = [str(value) for value in source["paths"].tolist()]
    reference_paths = [str(value) for value in reference["paths"].tolist()]
    embeddings = np.asarray(source["embeddings"])
    if embeddings.shape[0] != len(source_paths):
        raise ValueError(
            f"Embedding/path mismatch: {embeddings.shape[0]} != {len(source_paths)}"
        )
    order = reference_order(
        source_paths,
        reference_paths,
        suffix_parts=args.suffix_parts,
    )
    payload = {
        key: np.asarray(source[key])
        for key in source.files
        if key not in {"paths", "embeddings"}
    }
    payload.update(
        {
            "paths": np.asarray(reference_paths),
            "embeddings": embeddings[order],
            "relabel_source_cache": np.asarray(str(args.source_cache)),
            "relabel_reference_cache": np.asarray(str(args.reference_cache)),
            "relabel_suffix_parts": np.asarray(args.suffix_parts),
        }
    )
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_cache, **payload)
    summary = {
        "source_cache": str(args.source_cache),
        "reference_cache": str(args.reference_cache),
        "output_cache": str(args.output_cache),
        "query_count": len(reference_paths),
        "embedding_shape": list(embeddings.shape),
        "suffix_parts": args.suffix_parts,
        "permutation_identity": bool(
            np.array_equal(order, np.arange(len(order), dtype=np.int64))
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
