from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import portable_path
from ffxiv_prism.vision.evaluation_domains import (
    summarize_domain_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report retrieval accuracy and coverage by model race, gender, "
            "and race/gender intersection."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("data/processed/glamour_examples.jsonl"),
    )
    parser.add_argument("--branch", default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--min-groups-per-domain", type=int, default=5)
    parser.add_argument("--min-cases-per-domain", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    evaluation = report["evaluation"]
    if args.branch is not None:
        evaluation = evaluation[args.branch]
    cases = evaluation[f"{args.split}_cases"]
    summary = summarize_domain_evaluation(
        cases,
        read_jsonl(args.examples),
        min_groups_per_domain=args.min_groups_per_domain,
        min_cases_per_domain=args.min_cases_per_domain,
    )
    payload = {
        "report": portable_path(args.report),
        "examples": portable_path(args.examples),
        "branch": args.branch,
        "split": args.split,
        **summary,
    }
    write_json(args.output, payload)
    overall = summary["overall"]
    print(
        f"eligible={overall['eligible_case_count']} "
        f"top1={overall['appearance_recall_at_1']:.3f} "
        f"ready={summary['ready']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
