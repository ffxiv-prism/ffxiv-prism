from __future__ import annotations

import argparse
import json
from pathlib import Path

from ffxiv_prism.jsonl import write_json
from ffxiv_prism.vision.rank_fusion import route_case_results_by_margin
from scripts.evaluate_glamour_matching import summarize


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen score-margin router on any report split."
    )
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--secondary-report", type=Path, required=True)
    parser.add_argument("--split-role", required=True)
    parser.add_argument("--primary-margin-scale", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary = json.loads(
        args.primary_report.read_text(encoding="utf-8")
    )["evaluation"][f"{args.split_role}_cases"]
    secondary = json.loads(
        args.secondary_report.read_text(encoding="utf-8")
    )["evaluation"][f"{args.split_role}_cases"]
    routed = route_case_results_by_margin(
        primary,
        secondary,
        primary_margin_scale=args.primary_margin_scale,
    )
    summary = summarize(routed, args.top_k)
    payload = {
        "strategy": "frozen_relative_top_score_margin_router",
        "split_role": args.split_role,
        "primary_margin_scale": args.primary_margin_scale,
        "primary_report": args.primary_report.as_posix(),
        "secondary_report": args.secondary_report.as_posix(),
        "summary": summary,
        "cases": routed,
    }
    write_json(args.output, payload)
    print(
        f"eligible={summary['eligible_case_count']} "
        f"top1={summary['appearance_hits_at_1']} "
        f"top{args.top_k}={summary['appearance_hits_at_k']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
