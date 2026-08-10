from __future__ import annotations

import argparse
import json
from pathlib import Path

from ffxiv_prism.jsonl import read_jsonl, write_json
from scripts.rescore_retrieval_report_official import (
    rescore_case,
    summarize,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rescore a stored router report with official piece IDs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    examples = {
        str(row["example_id"]): row
        for row in read_jsonl(args.examples)
    }
    cases = [
        rescore_case(case, examples, args.top_k)
        for case in report["cases"]
    ]
    summary = summarize(cases, args.top_k)
    payload = {
        "evaluation_type": "stored_router_official_truth_rescore",
        "source_report": args.input.as_posix(),
        "examples": args.examples.as_posix(),
        "truth_policy": {
            "relation": "same_model",
            "heuristic_name_matching": False,
            "visual_similarity_inference": False,
        },
        "summary": summary,
        "cases": cases,
    }
    write_json(args.output, payload)
    print(
        f"eligible={summary['eligible_case_count']} "
        f"top1={summary['hits_at_1']} "
        f"top{args.top_k}={summary['hits_at_k']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
