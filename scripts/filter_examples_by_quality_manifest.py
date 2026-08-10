from __future__ import annotations

import argparse
from pathlib import Path

from ffxiv_prism.jsonl import read_jsonl, write_jsonl
from ffxiv_prism.vision.evaluation_quality import filter_examples_by_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Write only examples accepted by a quality manifest.")
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("data/processed/glamour_examples.jsonl"),
    )
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    examples = read_jsonl(args.examples)
    accepted = filter_examples_by_quality(
        examples,
        read_jsonl(args.quality_manifest),
    )
    write_jsonl(args.output, accepted)
    print(f"source={len(examples)} accepted={len(accepted)} output={args.output}")


if __name__ == "__main__":
    main()
