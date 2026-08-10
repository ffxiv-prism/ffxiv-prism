from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

np.linalg.inv(np.eye(2, dtype=np.float32))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.coverage_profiles import (
    encode_head_coverage_profiles,
    load_head_coverage_index,
    rerank_candidates_with_head_coverage,
)
from ffxiv_prism.matching.inference import (
    HEAD_PROMOTED_CANDIDATE_CROP_ROUTES,
    fuse_crop_results,
)
from ffxiv_prism.paths import portable_path
from ffxiv_prism.vision.human_parsing import HumanParser
from scripts.evaluate_cached_query_retrieval import load_query_embeddings
from scripts.evaluate_glamour_embedding_model import search_crops
from scripts.evaluate_rank_ensemble_retrieval import fuse_branch_searches
from scripts.evaluate_retrieval_funnel import (
    appearance_name_key,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    build_fold_context,
    summarize_ranked_cases,
    validate_cutoffs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-rank HEAD candidates by visible hair/face coverage agreement."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--branch", action="append", nargs=2, metavar=("INDEX_DIR", "QUERY_CACHE"), required=True
    )
    parser.add_argument("--coverage-index", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument(
        "--query-coverage-report",
        type=Path,
        help="Reuse query coverage measurements from an earlier compatible report.",
    )
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument("--blend", type=float, default=0.35)
    parser.add_argument("--max-difference", type=float)
    parser.add_argument("--minimum-query-exposed", type=float)
    parser.add_argument("--minimum-top-difference", type=float)
    parser.add_argument("--maximum-retrieval-margin", type=float)
    parser.add_argument("--branch-rrf-k", type=int, default=1)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()
    if not 0.0 <= args.blend <= 1.0:
        parser.error("--blend must be between zero and one")
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)

    metadata = read_jsonl(args.metadata)
    examples = read_jsonl(args.examples)
    quality_by_id = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
    }
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality_by_id,
        regions_dir=args.regions_dir,
        slot="HEAD",
    )
    retrieval_records = {
        example_id: [
            row for row in rows if str(row.get("name")) == "head_upper_accessory"
        ]
        for example_id, rows in context["records"].items()
    }
    expected_paths = [str(row["local_path"]) for row in metadata]
    searches = []
    branch_manifest = []
    for index_value, cache_value in args.branch:
        index_dir = Path(index_value)
        cache_path = Path(cache_value)
        branch_metadata = read_jsonl(index_dir / "metadata.jsonl")
        if [str(row["local_path"]) for row in branch_metadata] != expected_paths:
            raise ValueError(f"Candidate metadata mismatch in {index_dir}")
        searches.append(
            search_crops(
                branch_metadata,
                np.load(index_dir / "embeddings.npy", mmap_mode="r"),
                retrieval_records,
                load_query_embeddings(cache_path),
                args.candidate_k,
                device=args.device,
                candidate_crop_routes=HEAD_PROMOTED_CANDIDATE_CROP_ROUTES,
            )
        )
        branch_manifest.append(
            {"index_dir": portable_path(index_dir), "query_cache": portable_path(cache_path)}
        )
    merged = fuse_branch_searches(
        searches, rrf_k=args.branch_rrf_k, candidate_k=args.candidate_k
    )

    if args.query_coverage_report is not None:
        previous = json.loads(args.query_coverage_report.read_text(encoding="utf-8"))
        exposure_by_id = {
            str(row["example_id"]): float(row["query_exposed_fraction"])
            for row in previous["diagnostics"]
        }
        person_by_id = {
            str(row["example_id"]): float(row["query_person_fraction"])
            for row in previous["diagnostics"]
        }
    else:
        coverage_records = select_coverage_records(context["records"])
        ordered_ids = sorted(coverage_records)
        human_parser = HumanParser(device=args.device)
        exposed, person = encode_head_coverage_profiles(
            human_parser,
            [coverage_records[value] for value in ordered_ids],
            batch_size=args.batch_size,
        )
        exposure_by_id = dict(zip(ordered_ids, exposed, strict=True))
        person_by_id = dict(zip(ordered_ids, person, strict=True))
    coverage_index = load_head_coverage_index(args.coverage_index)

    baseline_cases, reranked_cases, diagnostics = rank_cases(
        context["examples"],
        retrieval_records,
        merged,
        metadata,
        coverage_index,
        exposure_by_id,
        person_by_id,
        rerank_k=args.rerank_k,
        blend=args.blend,
        minimum_top_difference=args.minimum_top_difference,
        maximum_retrieval_margin=args.maximum_retrieval_margin,
        max_difference=args.max_difference,
        minimum_query_exposed=args.minimum_query_exposed,
    )
    payload = {
        "slot": "HEAD",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "metadata": portable_path(args.metadata),
        "branches": branch_manifest,
        "coverage_index": portable_path(args.coverage_index),
        "candidate_k": args.candidate_k,
        "minimum_top_difference": args.minimum_top_difference,
        "maximum_retrieval_margin": args.maximum_retrieval_margin,
        "rerank_k": args.rerank_k,
        "blend": args.blend,
        "max_difference": args.max_difference,
        "minimum_query_exposed": args.minimum_query_exposed,
        "cutoffs": list(cutoffs),
        "baseline_summary": summarize_ranked_cases(baseline_cases, cutoffs),
        "reranked_summary": summarize_ranked_cases(reranked_cases, cutoffs),
        "baseline_cases": baseline_cases,
        "reranked_cases": reranked_cases,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "baseline": payload["baseline_summary"],
                "reranked": payload["reranked_summary"],
            },
            indent=2,
        )
    )
    return 0


def select_coverage_records(
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    priority = {"head_tight": 0, "head_context": 1, "head_upper_accessory": 2}
    output = {}
    for example_id, rows in records.items():
        usable = [
            row
            for row in rows
            if row.get("name") in priority
            and row.get("visibility") not in {"pose_uncertain", "out_of_frame"}
        ]
        if usable:
            row = min(usable, key=lambda value: priority[str(value["name"])])
            output[str(example_id)] = {"path": row["path"]}
    return output


def rank_cases(
    examples: Sequence[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    search: dict[str, list[dict[str, Any]]],
    metadata: list[dict[str, Any]],
    coverage_index: Any,
    exposure_by_id: dict[str, float],
    person_by_id: dict[str, float],
    *,
    rerank_k: int,
    blend: float,
    max_difference: float | None,
    minimum_query_exposed: float | None,
    minimum_top_difference: float | None,
    maximum_retrieval_margin: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    available_ids = available_pieces_by_slot(metadata)["HEAD"]
    available_keys = available_appearance_keys_by_slot(metadata)["HEAD"]
    baseline_cases = []
    reranked_cases = []
    diagnostics = []
    for example in examples:
        example_id = str(example["example_id"])
        label = (example.get("core_slot_labels") or {}).get("HEAD")
        base = {
            "example_id": example_id,
            "glamour_id": int(example["glamour_id"]),
            "model_gender": example.get("model_gender"),
            "model_race": example.get("model_race"),
            "expected_name": label.get("item_name") if label else None,
            "rank": None,
        }
        status, expected_ids, expected_key = label_status(
            label, available_ids, available_keys
        )
        if status != "eligible" or not records.get(example_id):
            if status == "eligible":
                status = "no_usable_crop"
            row = {**base, "status": status}
            baseline_cases.append(row)
            reranked_cases.append(dict(row))
            continue
        candidates = fuse_crop_results(
            records[example_id], search, top_k=1000, fusion="max"
        )
        keys = {
            int(row["piece_id"]): appearance_name_key(row.get("name_en"))
            for row in candidates
        }
        baseline_ids = [int(row["piece_id"]) for row in candidates]
        baseline_rank = truth_rank(baseline_ids, expected_ids, expected_key, keys)
        reranked = candidates
        if example_id in exposure_by_id:
            reranked = rerank_candidates_with_head_coverage(
                candidates,
                coverage_index,
                float(exposure_by_id[example_id]),
                max_difference=max_difference,
                minimum_query_exposed=minimum_query_exposed,
                rerank_k=rerank_k,
                blend=blend,
                minimum_top_difference=minimum_top_difference,
                maximum_retrieval_margin=maximum_retrieval_margin,
            )
        reranked_ids = [int(row["piece_id"]) for row in reranked]
        reranked_rank = truth_rank(reranked_ids, expected_ids, expected_key, keys)
        baseline_cases.append({**base, "status": status, "rank": baseline_rank})
        reranked_cases.append({**base, "status": status, "rank": reranked_rank})
        diagnostics.append(
            {
                "example_id": example_id,
                "query_exposed_fraction": float(exposure_by_id.get(example_id, -1.0)),
                "query_person_fraction": float(person_by_id.get(example_id, -1.0)),
                "baseline_top_piece_id": baseline_ids[0] if baseline_ids else None,
                "reranked_top_piece_id": reranked_ids[0] if reranked_ids else None,
                "truth_baseline_rank": baseline_rank,
                "truth_reranked_rank": reranked_rank,
            }
        )
    return baseline_cases, reranked_cases, diagnostics

def label_status(
    label: dict[str, Any] | None,
    available_ids: set[int],
    available_keys: set[str],
) -> tuple[str, set[int], str]:
    if not label or label.get("piece_match_type") == "empty":
        return "empty_label", set(), ""
    expected_ids = {int(value) for value in label.get("piece_ids", [])}
    if not expected_ids:
        return "missing_truth_piece_id", set(), ""
    expected_key = appearance_name_key(label.get("item_name"))
    if not (expected_ids & available_ids) and not (
        expected_key and expected_key in available_keys
    ):
        return "truth_not_in_index", expected_ids, expected_key
    return "eligible", expected_ids & available_ids, expected_key


def truth_rank(
    ranked_ids: Sequence[int],
    expected_ids: set[int],
    expected_key: str,
    keys_by_piece: dict[int, str],
) -> int | None:
    return next(
        (
            rank
            for rank, piece_id in enumerate(ranked_ids, start=1)
            if piece_id in expected_ids or keys_by_piece.get(piece_id) == expected_key
        ),
        None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
