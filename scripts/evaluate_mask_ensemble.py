from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import write_json
from ffxiv_prism.paths import PROCESSED_DIR, portable_path
from scripts.evaluate_glamour_matching import summarize
from scripts.train_glamour_adapter import cascade_outcomes

DEFAULT_RAW = PROCESSED_DIR / "glamour_adapter_eval_dinov2_base.json"
DEFAULT_PERSON = PROCESSED_DIR / "glamour_adapter_eval_dinov2_base_mask_person.json"
DEFAULT_OUTPUT = PROCESSED_DIR / "glamour_mask_ensemble_eval.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a person-mask precision head with the raw-image recall cascade."
    )
    parser.add_argument("--raw-report", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--person-report", type=Path, default=DEFAULT_PERSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--precision-section",
        default="adapted",
        help="Dot-separated report section supplying the precision ranking.",
    )
    parser.add_argument(
        "--recall-section",
        default="cascade",
        help="Dot-separated report section supplying the recall ranking.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    raw = load_report(args.raw_report)
    person = load_report(args.person_report)
    if raw["split"] != person["split"]:
        raise ValueError("Raw and person-mask reports must use the same glamour split")
    precision_model = nested_section(person, args.precision_section)
    recall_model = nested_section(raw, args.recall_section)

    evaluation = {}
    for split in ("validation", "test"):
        outcomes = cascade_outcomes(
            precision_model[f"{split}_cases"],
            recall_model[f"{split}_cases"],
            args.top_k,
        )
        evaluation[split] = summarize(outcomes, args.top_k)
        evaluation[f"{split}_cases"] = outcomes

    payload = {
        "strategy": "person_mask_adapted_top1_then_raw_cascade_tail",
        "raw_report": portable_path(args.raw_report),
        "person_report": portable_path(args.person_report),
        "precision_section": args.precision_section,
        "recall_section": args.recall_section,
        "split": raw["split"],
        **evaluation,
    }
    write_json(args.output, payload)
    test = evaluation["test"]
    print(
        f"test_eligible={test['eligible_case_count']} "
        f"appearance@1={test['appearance_recall_at_1']:.3f} "
        f"appearance@{args.top_k}={test['appearance_recall_at_k']:.3f} "
        f"mrr={test['appearance_mrr']:.3f} output={args.output}"
    )
    for slot, row in test["by_slot"].items():
        print(
            f"  {slot}: app@1={row['appearance_recall_at_1']:.3f} "
            f"app@{args.top_k}={row['appearance_recall_at_k']:.3f} "
            f"mrr={row['appearance_mrr']:.3f}"
        )
    return 0


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_section(payload: dict[str, Any], path: str) -> dict[str, Any]:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Report section not found: {path}")
        value = value[part]
    if not isinstance(value, dict):
        raise ValueError(f"Report section is not an object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
