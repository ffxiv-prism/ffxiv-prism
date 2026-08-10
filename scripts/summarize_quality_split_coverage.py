from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import portable_path
from ffxiv_prism.vision.evaluation_quality import (
    summarize_quality_split_coverage,
)
from ffxiv_prism.vision.evaluation_split import load_split_assignments
from ffxiv_prism.vision.retrieval import VALID_SLOTS


DEFAULT_EXAMPLES = PROJECT_ROOT / "data/processed/glamour_examples.jsonl"
DEFAULT_QUALITY = (
    PROJECT_ROOT / "data/processed/front_fullbody_quality_final_v2.jsonl"
)
DEFAULT_SPLIT = PROJECT_ROOT / "data/processed/glamour_split_v2.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/processed/front_fullbody_quality_final_v2.coverage.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize prediction-independent high-quality image and slot-case "
            "coverage by split."
        )
    )
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--quality-manifest", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-groups",
        type=int,
        default=50,
        help="Minimum sealed glamour groups required for readiness.",
    )
    parser.add_argument(
        "--target-cases",
        type=int,
        default=250,
        help="Minimum sealed slot cases required for readiness.",
    )
    parser.add_argument(
        "--slots",
        nargs="+",
        choices=VALID_SLOTS,
        default=list(VALID_SLOTS),
        help="Slot cases to count.",
    )
    parser.add_argument(
        "--allow-unmatched-labels",
        action="store_true",
        help=(
            "Count slot labels even when they do not resolve to local piece IDs. "
            "By default only locally evaluable labels are counted."
        ),
    )
    args = parser.parse_args()

    examples = read_jsonl(args.examples)
    quality_rows = read_jsonl(args.quality_manifest)
    assignments = load_split_assignments(args.split_manifest)
    summary = summarize_quality_split_coverage(
        examples,
        quality_rows,
        assignments,
        target_groups=args.target_groups,
        target_cases=args.target_cases,
        slots=args.slots,
        require_piece_ids=not args.allow_unmatched_labels,
    )
    payload = {
        "examples": portable_path(args.examples),
        "quality_manifest": portable_path(args.quality_manifest),
        "split_manifest": portable_path(args.split_manifest),
        "output": portable_path(args.output),
        **summary,
    }
    write_json(args.output, payload)
    target = summary["sealed_target"]
    print(
        f"accepted={summary['accepted_image_count']} "
        f"sealed_groups={target['current_groups']}/{target['target_groups']} "
        f"sealed_cases={target['current_cases']}/{target['target_cases']} "
        f"ready={target['ready']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
