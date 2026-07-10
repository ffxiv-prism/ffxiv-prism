from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import write_json
from scripts import evaluate_retrieval_funnel as base
from scripts.evaluate_glamour_matching import (
    ROLE_PROFILES,
    VISIBLE_WEIGHTS,
    available_pieces_by_slot,
    crop_record_weight,
)


DEFAULT_OUTPUT = Path(
    "data/processed/retrieval_funnel_v44_huiji_same_model_strict.json"
)


def rank_body_cases_official(
    examples: Sequence[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    search_results_by_crop: dict[str, list[dict[str, Any]]],
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank only against explicit piece IDs in the rebuilt truth labels."""
    available_ids = available_pieces_by_slot(metadata)["BODY"]
    role_weights = ROLE_PROFILES["balanced"]
    cases = []
    for example in examples:
        label = (example.get("core_slot_labels") or {}).get("BODY")
        root_ids = [
            int(value)
            for value in (label or {}).get("piece_ids", [])
        ]
        base_case = {
            "example_id": str(example["example_id"]),
            "glamour_id": int(example["glamour_id"]),
            "model_gender": example.get("model_gender"),
            "model_race": example.get("model_race"),
            "expected_name": label.get("item_name") if label else None,
            "expected_piece_ids": root_ids,
            "rank": None,
        }
        if not label or label.get("piece_match_type") == "empty":
            cases.append({**base_case, "status": "empty_label"})
            continue
        expected_ids = set(root_ids)
        if not expected_ids:
            cases.append(
                {**base_case, "status": "missing_truth_piece_id"}
            )
            continue
        eligible_ids = expected_ids & available_ids
        if not eligible_ids:
            cases.append({**base_case, "status": "truth_not_in_index"})
            continue
        slot_records = [
            record
            for record in records.get(str(example["example_id"]), [])
            if record["slot"] == "BODY"
            and VISIBLE_WEIGHTS.get(record["visibility"], 0.0) > 0.0
        ]
        if not slot_records:
            cases.append({**base_case, "status": "no_usable_crop"})
            continue

        scores: dict[int, float] = {}
        for record in slot_records:
            weight = crop_record_weight(record, role_weights, {})
            if weight <= 0:
                continue
            for result in search_results_by_crop.get(
                str(record["path"]),
                [],
            ):
                piece_id = int(result["piece_id"])
                contribution = weight * float(result["score"])
                scores[piece_id] = max(
                    scores.get(piece_id, 0.0),
                    contribution,
                )
        ranked_ids = sorted(
            scores,
            key=lambda piece_id: scores[piece_id],
            reverse=True,
        )
        rank = next(
            (
                offset
                for offset, piece_id in enumerate(ranked_ids, start=1)
                if piece_id in eligible_ids
            ),
            None,
        )
        cases.append(
            {
                **base_case,
                "status": "eligible",
                "eligible_expected_piece_ids": sorted(eligible_ids),
                "rank": rank,
            }
        )
    return cases


def main() -> int:
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", str(DEFAULT_OUTPUT)])
    output = argument_path("--output", DEFAULT_OUTPUT)
    base.rank_body_cases = rank_body_cases_official
    result = base.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["evaluation_type"] = (
        "strict_item_disjoint_retrieval_funnel_official_same_model"
    )
    payload["truth_policy"] = {
        "relation": "same_model_including_main_model",
        "source": (
            "HuijiWiki Item/Page rule over current game Item model fields"
        ),
        "heuristic_name_matching": False,
        "visual_similarity_inference": False,
    }
    write_json(output, payload)
    return result


def argument_path(flag: str, default: Path) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(flag) + 1])
    except (ValueError, IndexError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
