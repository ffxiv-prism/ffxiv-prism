from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import portable_path
from ffxiv_prism.vision.retrieval import select_primary_visual_pieces


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a candidate index while removing unrelated gearset pieces."
    )
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_index
    output = args.output_index
    if source.resolve() == output.resolve():
        raise ValueError("Source and output index directories must differ")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output index is not empty: {output}")

    metadata_path = source / "metadata.jsonl"
    embeddings_path = source / "embeddings.npy"
    rows = read_jsonl(metadata_path)
    embedding_shape = np.load(embeddings_path, mmap_mode="r").shape
    if embedding_shape[0] != len(rows):
        raise ValueError(
            f"Embedding/metadata mismatch: {embedding_shape[0]} != {len(rows)}"
        )

    before_ids = {
        int(piece["piece_id"])
        for row in rows
        for piece in row.get("pieces") or []
    }
    changed_rows = 0
    for row in rows:
        original = list(row.get("pieces") or [])
        selected = select_primary_visual_pieces(original)
        if not selected:
            raise ValueError(f"Metadata row has no candidate pieces: {row.get('image_row_id')}")
        if len(selected) != len(original):
            changed_rows += 1
        row["pieces"] = selected
    after_ids = {
        int(piece["piece_id"])
        for row in rows
        for piece in row.get("pieces") or []
    }

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(embeddings_path, output / "embeddings.npy")
    write_jsonl(output / "metadata.jsonl", rows)

    source_summary_path = source / "summary.json"
    source_summary = (
        json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary_path.is_file()
        else {}
    )
    summary = {
        **source_summary,
        "output": portable_path(output),
        "source_index": portable_path(source),
        "visual_piece_filter": "primary-family-v1",
        "metadata_row_count": len(rows),
        "changed_metadata_row_count": changed_rows,
        "unique_piece_count_before": len(before_ids),
        "unique_piece_count_after": len(after_ids),
        "removed_unrelated_piece_count": len(before_ids - after_ids),
        "embedding_shape": list(embedding_shape),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
