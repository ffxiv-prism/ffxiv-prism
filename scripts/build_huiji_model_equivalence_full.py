from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.huiji_model_equivalence import (
    build_piece_equivalence_from_models,
)
from ffxiv_prism.jsonl import read_jsonl, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build full-library same-model truth from the public HuijiWiki "
            "rule and current extracted Item fields."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/processed/eorzea.sqlite"),
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=Path(
            "data/raw/game_data/xiv_datamining_item_models_raw.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/"
            "piece_appearance_equivalence_huiji_models.json"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "data/processed/"
            "piece_appearance_equivalence_huiji_models_summary.json"
        ),
    )
    args = parser.parse_args()

    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    try:
        pieces = [
            dict(row)
            for row in connection.execute(
                """
                SELECT piece_id, slot, name_en, game_item_id,
                       game_item_ids_json
                FROM gear_pieces
                ORDER BY piece_id
                """
            )
        ]
    finally:
        connection.close()
    items = read_jsonl(args.items)
    payload = build_piece_equivalence_from_models(pieces, items)
    item_sources = sorted(
        {
            str(row["source"])
            for row in items
            if row.get("source")
        }
    )
    payload["source"]["model_fields"] = (
        item_sources[0] if len(item_sources) == 1 else item_sources
    )
    payload["database"] = args.db.as_posix()
    payload["items"] = args.items.as_posix()
    payload["built_at"] = timestamp()
    write_json(args.output, payload)

    summary = {
        key: value
        for key, value in payload.items()
        if key not in {
            "direct_equivalents",
            "same_model_equivalents",
            "exact_model_equivalents",
            "unmapped_pieces",
        }
    }
    summary.update(
        {
            "output": args.output.as_posix(),
            "mapped_rate": (
                payload["mapped_piece_count"] / payload["piece_count"]
                if payload["piece_count"]
                else 0.0
            ),
            "unmapped_examples": payload["unmapped_pieces"][:100],
        }
    )
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
