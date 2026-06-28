from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import PROCESSED_DIR, portable_path
from ffxiv_prism.vision.evaluation_split import (
    DEFAULT_SPLIT_SEED,
    stable_new_group_role,
)

DEFAULT_BASE_REPORT = PROCESSED_DIR / "glamour_adapter_eval_dinov2_base.json"
DEFAULT_OUTPUT = PROCESSED_DIR / "glamour_split_v2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze existing development groups and reserve new blind groups."
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROCESSED_DIR / "glamour_examples.jsonl",
    )
    parser.add_argument("--base-report", type=Path, default=DEFAULT_BASE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    glamour_ids = sorted(
        {
            str(example["glamour_id"])
            for example in read_jsonl(args.examples)
        }
    )
    assignments = {}
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        assignments.update(
            {
                str(glamour_id): str(role)
                for glamour_id, role in previous.get("assignments", {}).items()
            }
        )
    if args.base_report.is_file():
        base = json.loads(args.base_report.read_text(encoding="utf-8"))
        for glamour_id in base.get("split", {}).get("train", []):
            assignments.setdefault(str(glamour_id), "train")
        for glamour_id in base.get("split", {}).get("validation", []):
            assignments.setdefault(str(glamour_id), "validation")
        for glamour_id in base.get("split", {}).get("test", []):
            assignments.setdefault(str(glamour_id), "development_test")

    for glamour_id in glamour_ids:
        assignments.setdefault(
            glamour_id,
            stable_new_group_role(glamour_id, args.seed),
        )
    counts = Counter(assignments[glamour_id] for glamour_id in glamour_ids)
    payload = {
        "version": 2,
        "seed": args.seed,
        "policy": (
            "Original 50 groups retain train/validation roles; the repeatedly "
            "inspected old test is development_test. New groups use stable hash "
            "assignment and blind_test is excluded from tuning."
        ),
        "source_examples": portable_path(args.examples),
        "base_report": portable_path(args.base_report),
        "assignments": {
            glamour_id: assignments[glamour_id]
            for glamour_id in glamour_ids
        },
        "counts": dict(sorted(counts.items())),
    }
    write_json(args.output, payload)
    print(f"groups={len(glamour_ids)} counts={dict(counts)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
