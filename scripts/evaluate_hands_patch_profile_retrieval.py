from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.patch_profiles import (
    PatchProfileIndex,
    encode_dino_patch_profiles,
    load_patch_profile_index,
    symmetric_chamfer_score_matrix,
)
from ffxiv_prism.paths import CACHE_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from scripts.evaluate_cached_query_retrieval import build_quality_by_id
from scripts.evaluate_retrieval_funnel import (
    VISIBLE_WEIGHTS,
    appearance_name_key,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    build_fold_context,
    summarize_ranked_cases,
    validate_cutoffs,
)
from scripts.train_glamour_query_encoder import load_slot_eligibility


def select_query_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    query_crop: str,
) -> dict[str, dict[str, Any]]:
    output = {}
    for example_id, rows in records.items():
        selected = [
            row
            for row in rows
            if str(row.get("name")) == query_crop
            and VISIBLE_WEIGHTS.get(str(row.get("visibility")), 0.0) > 0.0
        ]
        if selected:
            output[str(example_id)] = dict(selected[0])
    return output


def rank_piece_ids(
    candidate_profile_indices: np.ndarray,
    profile_scores: np.ndarray,
    patch_index: PatchProfileIndex,
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[list[int], dict[int, str]]:
    if len(candidate_profile_indices) != len(profile_scores):
        raise ValueError("profile index/score length mismatch")
    scores_by_piece: dict[int, float] = {}
    names_by_piece: dict[int, str] = {}
    for profile_index, score in zip(
        candidate_profile_indices,
        profile_scores,
        strict=True,
    ):
        metadata_index = int(patch_index.metadata_indices[int(profile_index)])
        row = metadata[metadata_index]
        for piece in row.get("pieces") or []:
            piece_id = int(piece["piece_id"])
            scores_by_piece[piece_id] = max(
                scores_by_piece.get(piece_id, -float("inf")),
                float(score),
            )
            names_by_piece[piece_id] = str(piece.get("name_en") or "")
    ranked = sorted(scores_by_piece, key=scores_by_piece.get, reverse=True)
    return ranked, names_by_piece


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate local DINO patch-set retrieval for one HANDS crop route."
    )
    parser.add_argument("--examples", type=Path, required=True)
    quality_group = parser.add_mutually_exclusive_group(required=True)
    quality_group.add_argument("--quality-manifest", type=Path)
    quality_group.add_argument("--slot-eligibility-manifest", type=Path)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--candidate-index-dir", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--query-crop", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument(
        "--score-cache",
        type=Path,
        help="Optional dense example-by-profile score cache for fixed-pool reranking.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=24)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()

    cutoffs = validate_cutoffs(args.cutoffs, max(args.cutoffs))
    examples = read_jsonl(args.examples)
    quality_by_id = build_quality_by_id(
        examples,
        slot="HANDS",
        quality_rows=(
            read_jsonl(args.quality_manifest)
            if args.quality_manifest is not None
            else None
        ),
        eligibility=(
            load_slot_eligibility(args.slot_eligibility_manifest)
            if args.slot_eligibility_manifest is not None
            else None
        ),
    )
    metadata = read_jsonl(args.candidate_index_dir / "metadata.jsonl")
    patch_index = load_patch_profile_index(args.patch_index)
    if len(patch_index.profiles) != len(patch_index.metadata_indices):
        raise ValueError("Patch profiles and metadata indices are misaligned")
    if any(
        int(index) < 0 or int(index) >= len(metadata)
        for index in patch_index.metadata_indices
    ):
        raise ValueError("Patch profile metadata index is out of bounds")
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality_by_id,
        regions_dir=args.regions_dir,
        slot="HANDS",
    )
    selected_records = select_query_records(
        context["records"],
        query_crop=args.query_crop,
    )
    ordered_ids = [
        str(example["example_id"])
        for example in context["examples"]
        if str(example["example_id"]) in selected_records
    ]
    if not ordered_ids:
        raise ValueError(f"No usable query records for crop {args.query_crop!r}")

    device = resolve_torch_device(args.device)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        patch_index.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device).eval()
    query_profiles = encode_dino_patch_profiles(
        model,
        [{"path": selected_records[example_id]["path"]} for example_id in ordered_ids],
        projection=patch_index.projection,
        grid_size=patch_index.grid_size,
        batch_size=args.batch_size,
        device=device,
        preprocess=patch_index.preprocess,
    )
    query_profile_by_id = dict(zip(ordered_ids, query_profiles, strict=True))

    example_by_id = {
        str(example["example_id"]): example for example in context["examples"]
    }
    score_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    genders = sorted(
        {
            str(example_by_id[example_id].get("model_gender") or "").casefold()
            for example_id in ordered_ids
        }
    )
    candidate_genders = np.asarray(
        [value.casefold() for value in patch_index.model_genders]
    )
    for gender in genders:
        query_ids = [
            example_id
            for example_id in ordered_ids
            if str(example_by_id[example_id].get("model_gender") or "").casefold()
            == gender
        ]
        candidate_indices = np.flatnonzero(candidate_genders == gender)
        if not len(candidate_indices):
            raise ValueError(f"No candidate patch profiles for gender {gender!r}")
        scores = symmetric_chamfer_score_matrix(
            np.stack([query_profile_by_id[value] for value in query_ids]),
            patch_index.profiles[candidate_indices],
            batch_size=args.score_batch_size,
            device=device,
        )
        for row_index, example_id in enumerate(query_ids):
            score_data[example_id] = (candidate_indices, scores[row_index])

    if args.score_cache is not None:
        dense_scores = np.full(
            (len(ordered_ids), len(patch_index.profiles)),
            -np.inf,
            dtype=np.float16,
        )
        for row_index, example_id in enumerate(ordered_ids):
            candidate_indices, scores = score_data[example_id]
            dense_scores[row_index, candidate_indices] = scores.astype(np.float16)
        args.score_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.score_cache,
            example_ids=np.asarray(ordered_ids),
            scores=dense_scores,
            patch_index=np.asarray(str(args.patch_index)),
            query_crop=np.asarray(args.query_crop),
        )

    available_ids = available_pieces_by_slot(metadata)["HANDS"]
    available_keys = available_appearance_keys_by_slot(metadata)["HANDS"]
    cases = []
    for example in context["examples"]:
        example_id = str(example["example_id"])
        label = (example.get("core_slot_labels") or {}).get("HANDS")
        base = {
            "example_id": example_id,
            "glamour_id": int(example["glamour_id"]),
            "model_gender": example.get("model_gender"),
            "model_race": example.get("model_race"),
            "expected_name": label.get("item_name") if label else None,
            "rank": None,
        }
        status = "eligible"
        expected_ids: set[int] = set()
        expected_key = None
        if not label or label.get("piece_match_type") == "empty":
            status = "empty_label"
        else:
            expected_ids = {int(value) for value in label.get("piece_ids", [])}
            if not expected_ids:
                status = "missing_truth_piece_id"
            expected_key = appearance_name_key(label.get("item_name"))
            if (
                status == "eligible"
                and not (expected_ids & available_ids)
                and not (expected_key and expected_key in available_keys)
            ):
                status = "truth_not_in_index"
        if status == "eligible" and example_id not in score_data:
            status = "no_usable_crop"
        if status != "eligible":
            cases.append({**base, "status": status})
            continue
        candidate_indices, scores = score_data[example_id]
        ranked_ids, names_by_piece = rank_piece_ids(
            candidate_indices,
            scores,
            patch_index,
            metadata,
        )
        rank = next(
            (
                offset
                for offset, piece_id in enumerate(ranked_ids, start=1)
                if piece_id in expected_ids
                or appearance_name_key(names_by_piece.get(piece_id)) == expected_key
            ),
            None,
        )
        cases.append({**base, "status": "eligible", "rank": rank})

    summary = summarize_ranked_cases(cases, cutoffs)
    payload: dict[str, Any] = {
        "slot": "HANDS",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "quality_manifest": portable_path(args.quality_manifest),
        "slot_eligibility_manifest": portable_path(
            args.slot_eligibility_manifest
        ),
        "candidate_index_dir": portable_path(args.candidate_index_dir),
        "patch_index": portable_path(args.patch_index),
        "query_crop": args.query_crop,
        "candidate_crop": patch_index.candidate_crop,
        "model_id": patch_index.model_id,
        "preprocess": patch_index.preprocess,
        "grid_size": patch_index.grid_size,
        "projection_dimension": int(patch_index.projection.shape[1]),
        "score_cache": portable_path(args.score_cache),
        "cutoffs": list(cutoffs),
        "summary": summary,
        "cases": cases,
    }
    if args.baseline_report is not None:
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        baseline_by_id = {
            str(case["example_id"]): case
            for case in baseline["cases"]
            if case.get("status") == "eligible"
        }
        current_by_id = {
            str(case["example_id"]): case
            for case in cases
            if case.get("status") == "eligible"
        }
        payload["baseline_report"] = portable_path(args.baseline_report)
        payload["oracle_union_hits"] = {
            str(cutoff): len(
                {
                    example_id
                    for example_id, case in baseline_by_id.items()
                    if case.get("rank") is not None and int(case["rank"]) <= cutoff
                }
                | {
                    example_id
                    for example_id, case in current_by_id.items()
                    if case.get("rank") is not None and int(case["rank"]) <= cutoff
                }
            )
            for cutoff in cutoffs
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": summary,
                "oracle_union_hits": payload.get("oracle_union_hits"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
