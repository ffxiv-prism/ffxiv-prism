from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.paths import portable_path
from scripts.evaluate_compact_rank_gate import find_domain_regressions
from scripts.evaluate_retrieval_funnel import summarize_ranked_cases
from scripts.summarize_frozen_rank_gate import domain_hit_table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pool frozen HANDS occupied-slot Top-1 gate reports."
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    if not loaded:
        parser.error("at least one report is required")
    cutoffs = tuple(int(value) for value in loaded[0]["cutoffs"])
    first_gate = loaded[0].get("best_safe_gate")
    if not first_gate:
        raise ValueError("report has no safe occupied-slot gate")
    policy = {
        "occupied_multiplier": float(first_gate["occupied_multiplier"]),
        "max_base_rank": int(first_gate["max_base_rank"]),
    }
    for report in loaded:
        gate = report.get("best_safe_gate")
        if tuple(int(value) for value in report["cutoffs"]) != cutoffs:
            raise ValueError("all reports must use identical cutoffs")
        if not gate or {
            "occupied_multiplier": float(gate["occupied_multiplier"]),
            "max_base_rank": int(gate["max_base_rank"]),
        } != policy:
            raise ValueError("all reports must use the same frozen policy")

    baseline_cases = [
        case for report in loaded for case in report["baseline"]["cases"]
    ]
    gated_cases = [
        case for report in loaded for case in report["best_safe_gate"]["cases"]
    ]
    regressions = find_domain_regressions(baseline_cases, gated_cases, cutoffs)
    gated_by_id = {str(case["example_id"]): case for case in gated_cases}
    top1_gains = [
        str(case["example_id"])
        for case in baseline_cases
        if case.get("status") == "eligible"
        and case.get("rank") != 1
        and gated_by_id[str(case["example_id"])].get("rank") == 1
    ]
    top1_losses = [
        str(case["example_id"])
        for case in baseline_cases
        if case.get("status") == "eligible"
        and case.get("rank") == 1
        and gated_by_id[str(case["example_id"])].get("rank") != 1
    ]
    payload = {
        "slot": "HANDS",
        "fold_reports": [portable_path(path) for path in args.reports],
        "folds": [int(report["fold"]) for report in loaded],
        "cutoffs": list(cutoffs),
        "frozen_policy": policy,
        "accepted_change_count": sum(
            int(report["best_safe_gate"]["accepted_change_count"])
            for report in loaded
        ),
        "top1_change_audit": {
            "gain_count": len(top1_gains),
            "loss_count": len(top1_losses),
            "net_gain": len(top1_gains) - len(top1_losses),
            "gain_example_ids": top1_gains,
            "loss_example_ids": top1_losses,
        },
        "baseline_summary": summarize_ranked_cases(baseline_cases, cutoffs),
        "gated_candidate_summary": summarize_ranked_cases(gated_cases, cutoffs),
        "domain_regressions": regressions,
        "passes_domain_guard": not regressions,
        "domain_hit_table": domain_hit_table(
            baseline_cases, gated_cases, cutoffs
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
