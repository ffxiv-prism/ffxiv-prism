from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path, resolve_project_path  # noqa: E402
from ffxiv_prism.vision.retrieval import load_candidate_images  # noqa: E402
from scripts.evaluate_glamour_embedding_model import build_encoder  # noqa: E402


DEFAULT_SOURCE = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_OUTPUT = Path("data/indexes/facebook-dinov2-base-color-20260725")
DEFAULT_DB = Path("data/processed/eorzea.sqlite")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Safely extend an existing visual candidate index with rows newly "
            "available from SQLite while preserving all existing metadata."
        )
    )
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-index", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--expected-new-count", type=int, default=None)
    parser.add_argument("--expected-gearset-slug", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.source_index.resolve() == args.output_index.resolve():
        raise ValueError("Source and output index directories must differ")
    if args.output_index.exists() and any(args.output_index.iterdir()):
        raise FileExistsError(f"Output index is not empty: {args.output_index}")

    source_metadata = read_jsonl(args.source_index / "metadata.jsonl")
    source_embeddings = np.load(
        args.source_index / "embeddings.npy",
        mmap_mode="r",
    )
    if source_embeddings.ndim != 2 or source_embeddings.shape[0] != len(
        source_metadata
    ):
        raise ValueError(
            "Source embedding/metadata mismatch: "
            f"{source_embeddings.shape} vs {len(source_metadata)}"
        )

    current_rows = load_candidate_images(db_path=args.db)
    additions = find_new_rows(source_metadata, current_rows)
    validate_additions(
        additions,
        expected_count=args.expected_new_count,
        expected_gearset_slug=args.expected_gearset_slug,
    )
    if not additions:
        raise ValueError("No new candidate rows found; refusing to copy unchanged index")

    manifest_path = args.source_index / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_id = str(source_manifest["model_id"])
    encoder_kind = str(source_manifest.get("encoder_kind") or "auto")
    preprocess = str(source_manifest.get("preprocess") or "color")
    pooling = str(source_manifest.get("pooling") or "cls")
    encoder = build_encoder(
        model_id,
        encoder_kind,
        args.device,
        preprocess,
        pooling,
    )

    addition_paths = [
        resolve_project_path(row.get("local_path"))
        for row in additions
    ]
    missing_paths = [
        row.get("local_path")
        for row, path in zip(additions, addition_paths)
        if path is None or not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"New candidate images are missing: {missing_paths[:5]}"
        )
    new_embeddings = encoder.encode_paths(
        [path for path in addition_paths if path is not None],
        args.batch_size,
    )
    if new_embeddings.ndim != 2 or new_embeddings.shape != (
        len(additions),
        source_embeddings.shape[1],
    ):
        raise ValueError(
            "New embedding shape does not match source index: "
            f"{new_embeddings.shape} vs (*, {source_embeddings.shape[1]})"
        )

    args.output_index.mkdir(parents=True, exist_ok=True)
    output_embeddings_path = args.output_index / "embeddings.npy"
    output_matrix = np.lib.format.open_memmap(
        output_embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(source_metadata) + len(additions),
            int(source_embeddings.shape[1]),
        ),
    )
    output_matrix[: len(source_metadata)] = source_embeddings
    output_matrix[len(source_metadata) :] = new_embeddings
    output_matrix.flush()

    output_metadata = [*source_metadata, *additions]
    write_jsonl(args.output_index / "metadata.jsonl", output_metadata)
    output_manifest = {
        **source_manifest,
        "embedding_count": len(output_metadata),
        "embedding_dimension": int(output_matrix.shape[1]),
        "embeddings": portable_path(output_embeddings_path),
        "metadata": portable_path(args.output_index / "metadata.jsonl"),
        "extended_from": portable_path(args.source_index),
        "new_candidate_count": len(additions),
        "new_gearset_slugs": sorted(gearset_slugs(additions)),
    }
    write_json(args.output_index / "manifest.json", output_manifest)
    write_json(
        args.output_index / "summary.json",
        {
            "source_index": portable_path(args.source_index),
            "output": portable_path(args.output_index),
            "source_candidate_count": len(source_metadata),
            "new_candidate_count": len(additions),
            "candidate_count": len(output_metadata),
            "new_gearset_slugs": sorted(gearset_slugs(additions)),
            "new_by_slot": count_by_slot(additions),
            "embedding_shape": list(output_matrix.shape),
            "preserved_source_rows": True,
        },
    )
    print(
        json.dumps(
            {
                "source_candidate_count": len(source_metadata),
                "new_candidate_count": len(additions),
                "candidate_count": len(output_metadata),
                "new_by_slot": count_by_slot(additions),
                "new_gearset_slugs": sorted(gearset_slugs(additions)),
                "output": str(args.output_index),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def canonical_key(row: dict[str, Any]) -> tuple[str, str]:
    source_url = str(row.get("source_image_url") or "").strip()
    if source_url:
        return ("source_image_url", source_url)
    local_path = str(row.get("local_path") or "").strip()
    if local_path:
        return ("local_path", local_path)
    raise ValueError("Candidate row has neither source_image_url nor local_path")


def find_new_rows(
    source_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_keys = {canonical_key(row) for row in source_rows}
    additions = [
        dict(row)
        for row in current_rows
        if canonical_key(row) not in source_keys
    ]
    addition_keys = [canonical_key(row) for row in additions]
    if len(addition_keys) != len(set(addition_keys)):
        raise ValueError("Current candidate rows contain duplicate new identities")
    return additions


def validate_additions(
    rows: list[dict[str, Any]],
    *,
    expected_count: int | None,
    expected_gearset_slug: str | None,
) -> None:
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} new rows, found {len(rows)}"
        )
    if expected_gearset_slug is not None:
        unexpected = sorted(gearset_slugs(rows) - {expected_gearset_slug})
        missing_expected = any(
            expected_gearset_slug not in {
                str(piece.get("gearset_slug") or "")
                for piece in row.get("pieces") or []
            }
            for row in rows
        )
        if unexpected or missing_expected:
            raise ValueError(
                "New rows do not exclusively belong to "
                f"{expected_gearset_slug!r}; unexpected={unexpected}"
            )


def gearset_slugs(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(piece.get("gearset_slug"))
        for row in rows
        for piece in row.get("pieces") or []
        if piece.get("gearset_slug")
    }


def count_by_slot(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        slot = str(row.get("slot") or "unknown")
        counts[slot] = counts.get(slot, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
