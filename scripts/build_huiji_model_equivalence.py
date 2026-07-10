from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.huiji_model_equivalence import (
    build_piece_equivalence_from_models,
)
from ffxiv_prism.jsonl import read_jsonl, write_json


DEFAULT_DB = Path("data/processed/eorzea.sqlite")
DEFAULT_ITEMS = Path("data/raw/game_data/cafemaker_item_models_raw.jsonl")
DEFAULT_OUTPUT = Path(
    "data/processed/piece_appearance_equivalence_huiji_models.json"
)
DEFAULT_SUMMARY = Path(
    "data/processed/piece_appearance_equivalence_huiji_models_summary.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact/main-model equivalence by reproducing the public "
            "HuijiWiki Item/Page rule over game Item model fields."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    pieces = load_pieces(args.db)
    items = read_jsonl(args.items)
    payload = build_piece_equivalence_from_models(pieces, items)
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


def load_pieces(path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    piece_id,
                    slot,
                    name_en,
                    game_item_id,
                    game_item_ids_json
                FROM gear_pieces
                ORDER BY piece_id
                """
            )
        ]
    finally:
        connection.close()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
