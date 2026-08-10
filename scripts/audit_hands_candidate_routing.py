from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402
from ffxiv_prism.vision.evaluation_split import (  # noqa: E402
    load_split_assignments,
)
from scripts.evaluate_glamour_matching import collect_crop_records  # noqa: E402
from scripts.evaluate_retrieval_funnel import (  # noqa: E402
    HANDS_CANDIDATE_CROP_ROUTES,
)
from scripts.train_glamour_query_encoder import (  # noqa: E402
    load_slot_eligibility,
)


def build_candidate_lookup(
    metadata: Sequence[Mapping[str, Any]],
    *,
    slot: str,
) -> dict[int, list[int]]:
    output: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        if str(row.get("slot")) != slot:
            continue
        for piece in row.get("pieces", []):
            output[int(piece["piece_id"])].append(index)
    return dict(output)


def candidate_positive_indices(
    metadata: Sequence[Mapping[str, Any]],
    lookup: Mapping[int, Sequence[int]],
    piece_ids: Sequence[int],
    *,
    query_crop: str,
    model_gender: str,
    route_candidates: bool,
    route_gender: bool,
) -> list[int]:
    allowed_crops = set(
        HANDS_CANDIDATE_CROP_ROUTES.get(query_crop, ())
        if route_candidates
        else ()
    )
    gender = model_gender.casefold() if route_gender else ""
    return sorted(
        {
            int(index)
            for piece_id in piece_ids
            for index in lookup.get(int(piece_id), ())
            if (
                not allowed_crops
                or str(metadata[int(index)].get("candidate_crop"))
                in allowed_crops
            )
            and (
                not gender
                or str(metadata[int(index)].get("model_gender", "")).casefold()
                == gender
            )
        }
    )


def classify_route_coverage(
    *,
    strict_indices: Sequence[int],
    any_gender_indices: Sequence[int],
) -> str:
    if strict_indices:
        return "strict_covered"
    if any_gender_indices:
        return "requires_gender_fallback"
    return "route_uncovered"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit HANDS truth reachability after crop and gender candidate "
            "routing, not merely global index membership."
        )
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--slot-eligibility-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    examples = read_jsonl(args.examples)
    eligibility = load_slot_eligibility(args.slot_eligibility_manifest)
    eligible_examples = [
        example
        for example in examples
        if "HANDS" in eligibility.get(str(example["example_id"]), set())
    ]
    records = collect_crop_records(eligible_examples, args.regions_dir)
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    lookup = build_candidate_lookup(metadata, slot="HANDS")
    assignments = (
        load_split_assignments(args.fold_manifest)
        if args.fold_manifest is not None
        else {}
    )

    rows: list[dict[str, Any]] = []
    for example in eligible_examples:
        example_id = str(example["example_id"])
        label = (example.get("core_slot_labels") or {}).get("HANDS")
        if not label or not label.get("piece_ids"):
            continue
        for record in records.get(example_id, []):
            strict = candidate_positive_indices(
                metadata,
                lookup,
                [int(value) for value in label["piece_ids"]],
                query_crop=str(record.get("name")),
                model_gender=str(example.get("model_gender") or ""),
                route_candidates=True,
                route_gender=True,
            )
            any_gender = candidate_positive_indices(
                metadata,
                lookup,
                [int(value) for value in label["piece_ids"]],
                query_crop=str(record.get("name")),
                model_gender=str(example.get("model_gender") or ""),
                route_candidates=True,
                route_gender=False,
            )
            rows.append(
                {
                    "example_id": example_id,
                    "glamour_id": int(example["glamour_id"]),
                    "fold_role": str(
                        assignments.get(str(example["glamour_id"]))
                        or "unassigned"
                    ),
                    "model_gender": example.get("model_gender"),
                    "query_crop": str(record.get("name")),
                    "expected_name": label.get("item_name"),
                    "piece_ids": [int(value) for value in label["piece_ids"]],
                    "strict_candidate_count": len(strict),
                    "any_gender_candidate_count": len(any_gender),
                    "status": classify_route_coverage(
                        strict_indices=strict,
                        any_gender_indices=any_gender,
                    ),
                }
            )

    status_counts = Counter(row["status"] for row in rows)
    problem_rows = [row for row in rows if row["status"] != "strict_covered"]
    problem_examples = sorted({row["example_id"] for row in problem_rows})
    route_uncovered = status_counts.get("route_uncovered", 0)
    fallback_pairs = status_counts.get("requires_gender_fallback", 0)
    status = (
        "do_not_train"
        if route_uncovered
        else (
            "ready_with_gender_fallback"
            if fallback_pairs
            else "ready_strict"
        )
    )
    payload = {
        "version": 1,
        "status": status,
        "eligible_example_count": len(eligible_examples),
        "checked_crop_pairs": len(rows),
        "status_counts": dict(status_counts),
        "strict_coverage_rate": (
            status_counts.get("strict_covered", 0) / len(rows) if rows else 0.0
        ),
        "any_gender_route_coverage_rate": (
            (len(rows) - route_uncovered) / len(rows) if rows else 0.0
        ),
        "problem_example_count": len(problem_examples),
        "problem_examples": problem_examples,
        "problem_rows": problem_rows,
        "by_query_crop": {
            crop: dict(
                Counter(
                    row["status"]
                    for row in rows
                    if row["query_crop"] == crop
                )
            )
            for crop in sorted({row["query_crop"] for row in rows})
        },
        "by_fold_role": {
            role: dict(
                Counter(
                    row["status"]
                    for row in rows
                    if row["fold_role"] == role
                )
            )
            for role in sorted({row["fold_role"] for row in rows})
        },
        "sources": {
            "examples": portable_path(args.examples),
            "slot_eligibility_manifest": portable_path(
                args.slot_eligibility_manifest
            ),
            "regions_dir": portable_path(args.regions_dir),
            "index_dir": portable_path(args.index_dir),
            "fold_manifest": portable_path(args.fold_manifest),
        },
        "interpretation": (
            "Global candidate coverage is insufficient when serving filters "
            "remove every truth render. Any fallback must be implemented and "
            "re-evaluated explicitly."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "problem_rows"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if status == "do_not_train" else 0


if __name__ == "__main__":
    raise SystemExit(main())
