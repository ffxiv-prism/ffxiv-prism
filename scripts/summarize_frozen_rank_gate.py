from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.paths import portable_path
from scripts.evaluate_compact_rank_gate import find_domain_regressions
from scripts.evaluate_retrieval_funnel import summarize_ranked_cases


def domain_hit_table(
    baseline_cases: Sequence[Mapping[str, Any]],
    candidate_cases: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    candidate_by_id = {str(case["example_id"]): case for case in candidate_cases}
    output = {}
    for dimension in ("model_gender", "model_race"):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for case in baseline_cases:
            if case.get("status") == "eligible":
                grouped[str(case.get(dimension) or "unknown")].append(case)
        rows = {}
        for value, cases in sorted(grouped.items()):
            rows[value] = {
                "eligible_case_count": len(cases),
                "baseline_hits": {
                    str(cutoff): sum(
                        case.get("rank") is not None
                        and int(case["rank"]) <= cutoff
                        for case in cases
                    )
                    for cutoff in cutoffs
                },
                "candidate_hits": {
                    str(cutoff): sum(
                        candidate_by_id[str(case["example_id"])].get("rank")
                        is not None
                        and int(
                            candidate_by_id[str(case["example_id"])]["rank"]
                        )
                        <= cutoff
                        for case in cases
                    )
                    for cutoff in cutoffs
                },
            }
        output[dimension] = rows
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pool frozen compact rank-gate reports without retuning."
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.reports
    ]
    if not loaded:
        parser.error("at least one report is required")
    cutoffs = tuple(int(value) for value in loaded[0]["cutoffs"])
    policy = loaded[0]["experiments"][0]["policy"]
    for report in loaded:
        if tuple(int(value) for value in report["cutoffs"]) != cutoffs:
            raise ValueError("all reports must use identical cutoffs")
        if report["experiments"][0]["policy"] != policy:
            raise ValueError("all reports must use the same frozen policy")
        if "baseline_cases" not in report:
            raise ValueError("reports must be generated in single-policy mode")

    baseline_cases = [case for report in loaded for case in report["baseline_cases"]]
    gated_cases = [
        case
        for report in loaded
        for case in report["experiments"][0]["cases"]
    ]
    ungated_cases = [
        case for report in loaded for case in report["ungated_candidate_cases"]
    ]
    regressions = find_domain_regressions(
        baseline_cases, gated_cases, cutoffs
    )
    payload = {
        "slot": loaded[0]["slot"],
        "fold_reports": [portable_path(path) for path in args.reports],
        "folds": [int(report["fold"]) for report in loaded],
        "cutoffs": list(cutoffs),
        "candidate_last_branch_weight": loaded[0][
            "candidate_last_branch_weight"
        ],
        "frozen_policy": policy,
        "baseline_summary": summarize_ranked_cases(baseline_cases, cutoffs),
        "ungated_candidate_summary": summarize_ranked_cases(
            ungated_cases, cutoffs
        ),
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
