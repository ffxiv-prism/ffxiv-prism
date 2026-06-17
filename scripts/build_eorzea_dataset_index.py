# 功能：根据已采集的图片 manifest 构建本地 Eorzea Collection 图片数据集索引。

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.dataset.eorzea_index import (  # noqa: E402
    build_eorzea_image_index,
    summary_to_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a processed Eorzea Collection image index.")
    parser.add_argument("--manifest", type=Path, default=None, help="Raw image manifest JSONL path.")
    parser.add_argument("--output", type=Path, default=None, help="Processed image index JSONL path.")
    parser.add_argument("--summary", type=Path, default=None, help="Dataset coverage summary JSON path.")
    args = parser.parse_args()

    summary = build_eorzea_image_index(
        manifest_path=args.manifest,
        output_path=args.output,
        summary_path=args.summary,
    )
    print(summary_to_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

