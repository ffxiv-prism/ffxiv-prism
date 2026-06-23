from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.dataset.sqlite_store import build_eorzea_sqlite, summary_to_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a normalized SQLite database for the Eorzea dataset.")
    parser.add_argument("--db", type=Path, default=None, help="SQLite database output path.")
    parser.add_argument("--gearsets", type=Path, default=None, help="Gearset details JSONL input path.")
    parser.add_argument("--gear-pieces", type=Path, default=None, help="Gear pieces JSONL input path.")
    parser.add_argument("--images", type=Path, default=None, help="QA image index JSONL input path.")
    parser.add_argument("--summary", type=Path, default=None, help="SQLite summary JSON output path.")
    args = parser.parse_args()

    summary = build_eorzea_sqlite(
        db_path=args.db,
        gearsets_path=args.gearsets,
        gear_pieces_path=args.gear_pieces,
        images_path=args.images,
        summary_path=args.summary,
    )
    print(summary_to_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
