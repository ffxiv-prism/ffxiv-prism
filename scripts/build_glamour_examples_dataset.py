from __future__ import annotations

import argparse
from pathlib import Path

from ffxiv_prism.dataset.glamour_examples import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_SUMMARY,
    build_glamour_examples_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an image-level labeled glamour example manifest.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = build_glamour_examples_dataset(args.input, args.output, args.summary)
    print(
        f"glamours={summary['glamour_records']} images={summary['image_examples']} "
        f"missing={len(summary['missing_images'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
