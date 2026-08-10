from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import PROCESSED_DIR, portable_path
from ffxiv_prism.vision.evaluation_quality import quality_allowlist
from ffxiv_prism.vision.rank_fusion import route_case_results_by_margin
from scripts.evaluate_glamour_matching import SLOTS, summarize
from scripts.evaluate_mask_ensemble import nested_section


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Route between two retrieval reports using only normalized "
            "Top-1/Top-2 score margins."
        )
    )
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--secondary-report", type=Path, required=True)
    parser.add_argument("--primary-section", default="evaluation")
    parser.add_argument("--secondary-section", default="evaluation")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--quality-manifest", type=Path)
    parser.add_argument(
        "--slots",
        nargs="+",
        choices=SLOTS,
        default=list(SLOTS),
    )
    parser.add_argument("--primary-margin-scale", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "confidence_router_eval.json",
    )
    args = parser.parse_args()
    if args.primary_margin_scale <= 0:
        raise ValueError("primary-margin-scale must be positive")

    primary = nested_section(load_report(args.primary_report), args.primary_section)
    secondary = nested_section(
        load_report(args.secondary_report),
        args.secondary_section,
    )
    accepted = (
        quality_allowlist(read_jsonl(args.quality_manifest))
        if args.quality_manifest is not None
        else None
    )
    primary_cases = filter_cases(
        primary[f"{args.split}_cases"],
        set(args.slots),
        accepted,
    )
    secondary_cases = filter_cases(
        secondary[f"{args.split}_cases"],
        set(args.slots),
        accepted,
    )
    routed = route_case_results_by_margin(
        primary_cases,
        secondary_cases,
        primary_margin_scale=args.primary_margin_scale,
    )
    summary = summarize(routed, args.top_k)
    payload = {
        "strategy": "relative_top_score_margin_router",
        "primary_report": portable_path(args.primary_report),
        "secondary_report": portable_path(args.secondary_report),
        "primary_section": args.primary_section,
        "secondary_section": args.secondary_section,
        "split": args.split,
        "slots": list(args.slots),
        "quality_manifest": portable_path(args.quality_manifest),
        "primary_margin_scale": args.primary_margin_scale,
        "summary": summary,
        "cases": routed,
    }
    write_json(args.output, payload)
    print(
        f"eligible={summary['eligible_case_count']} "
        f"appearance@1={summary['appearance_recall_at_1']:.3f} "
        f"appearance@{args.top_k}={summary['appearance_recall_at_k']:.3f} "
        f"mrr={summary['appearance_mrr']:.3f} output={args.output}"
    )
    return 0


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def filter_cases(
    cases: list[dict[str, Any]],
    slots: set[str],
    accepted: set[str] | None,
) -> list[dict[str, Any]]:
    return [
        case
        for case in cases
        if str(case.get("slot")) in slots
        and (
            accepted is None
            or str(case.get("example_id")) in accepted
        )
    ]


if __name__ == "__main__":
    raise SystemExit(main())
