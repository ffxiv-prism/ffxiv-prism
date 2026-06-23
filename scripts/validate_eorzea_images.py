from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.dataset.image_qa import build_image_qa_index, summary_to_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local Eorzea image files and build a QA index.")
    parser.add_argument("--index", type=Path, default=None, help="Processed image index JSONL path.")
    parser.add_argument("--output", type=Path, default=None, help="QA image index JSONL output path.")
    parser.add_argument("--summary", type=Path, default=None, help="QA summary JSON output path.")
    parser.add_argument("--missing-manifest", type=Path, default=None, help="Missing image rows JSONL output path.")
    parser.add_argument("--min-width", type=int, default=64, help="Minimum acceptable image width.")
    parser.add_argument("--min-height", type=int, default=64, help="Minimum acceptable image height.")
    args = parser.parse_args()

    summary = build_image_qa_index(
        index_path=args.index,
        output_path=args.output,
        summary_path=args.summary,
        missing_manifest_path=args.missing_manifest,
        min_width=args.min_width,
        min_height=args.min_height,
    )
    print(summary_to_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
