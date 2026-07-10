from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import write_json
from scripts import evaluate_retrieval_funnel as base
from scripts.evaluate_retrieval_funnel_official import (
    argument_path,
    rank_body_cases_official,
)


DEFAULT_OUTPUT = Path(
    "data/processed/retrieval_funnel_v44_huiji_exact_model_strict.json"
)


def main() -> int:
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", str(DEFAULT_OUTPUT)])
    output = argument_path("--output", DEFAULT_OUTPUT)
    base.rank_body_cases = rank_body_cases_official
    result = base.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["evaluation_type"] = (
        "strict_item_disjoint_retrieval_funnel_official_exact_model"
    )
    payload["truth_policy"] = {
        "relation": "exact_model_only",
        "source": (
            "HuijiWiki Item/Page rule over current game Item model fields"
        ),
        "heuristic_name_matching": False,
        "visual_similarity_inference": False,
    }
    write_json(output, payload)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
