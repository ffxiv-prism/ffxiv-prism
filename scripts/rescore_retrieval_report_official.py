from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ffxiv_prism.jsonl import read_jsonl, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore stored ranked candidates using explicit official "
            "same-model piece IDs only."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--relation",
        choices=("same_model", "exact_model"),
        required=True,
    )
    parser.add_argument("--split-role", default="validation")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    examples = {
        str(row["example_id"]): row
        for row in read_jsonl(args.examples)
    }
    case_key = f"{args.split_role}_cases"
    source_cases = report["evaluation"][case_key]
    cases = [
        rescore_case(case, examples, args.top_k)
        for case in source_cases
    ]
    summary = summarize(cases, args.top_k)
    payload = {
        "evaluation_type": "stored_candidates_official_truth_rescore",
        "source_report": args.input.as_posix(),
        "examples": args.examples.as_posix(),
        "split_role": args.split_role,
        "truth_policy": {
            "relation": args.relation,
            "source": (
                "HuijiWiki Item/Page rule over current game Item model fields"
            ),
            "heuristic_name_matching": False,
            "visual_similarity_inference": False,
        },
        "summary": summary,
        "cases": cases,
    }
    write_json(args.output, payload)
    print(
        f"eligible={summary['eligible_case_count']} "
        f"top1={summary['hits_at_1']}/{summary['eligible_case_count']} "
        f"top{args.top_k}={summary['hits_at_k']}/"
        f"{summary['eligible_case_count']} output={args.output}"
    )
    return 0


def rescore_case(
    case: dict[str, Any],
    examples: dict[str, dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    output = {
        "example_id": str(case.get("example_id")),
        "glamour_id": case.get("glamour_id"),
        "slot": case.get("slot"),
        "expected_name": case.get("expected_name"),
        "status": case.get("status"),
        "rank": None,
        "hit_at_1": False,
        "hit_at_k": False,
    }
    if case.get("status") != "eligible":
        return output
    example = examples.get(str(case.get("example_id")))
    label = (
        (example or {}).get("core_slot_labels") or {}
    ).get(str(case.get("slot")))
    expected_ids = {
        int(value)
        for value in (label or {}).get("piece_ids", [])
    }
    output["expected_piece_ids"] = sorted(expected_ids)
    if not expected_ids:
        output["status"] = "missing_truth_piece_id"
        return output
    rank = next(
        (
            offset
            for offset, result in enumerate(
                case.get("results", []),
                start=1,
            )
            if int(result["piece_id"]) in expected_ids
        ),
        None,
    )
    output["rank"] = rank
    output["hit_at_1"] = rank == 1
    output["hit_at_k"] = rank is not None and rank <= top_k
    return output


def summarize(
    cases: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    eligible = [row for row in cases if row["status"] == "eligible"]
    count = len(eligible)
    hits_at_1 = sum(bool(row["hit_at_1"]) for row in eligible)
    hits_at_k = sum(bool(row["hit_at_k"]) for row in eligible)
    return {
        "case_count": len(cases),
        "eligible_case_count": count,
        "status_counts": dict(
            sorted(Counter(row["status"] for row in cases).items())
        ),
        "top_k": top_k,
        "hits_at_1": hits_at_1,
        "hits_at_k": hits_at_k,
        "recall_at_1": hits_at_1 / count if count else 0.0,
        "recall_at_k": hits_at_k / count if count else 0.0,
        "mrr_at_k": (
            sum(
                1.0 / int(row["rank"])
                for row in eligible
                if row.get("rank") is not None
                and int(row["rank"]) <= top_k
            )
            / count
            if count
            else 0.0
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
