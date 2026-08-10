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
from ffxiv_prism.matching.inference import crop_weight
from ffxiv_prism.paths import portable_path
from scripts.evaluate_retrieval_funnel import (
    SLOTS,
    appearance_name_key,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    build_fold_context,
    summarize_ranked_cases,
    validate_cutoffs,
)


def stable_weighted_rrf(
    piece_lists: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    rrf_k: int,
    top_k: int,
) -> np.ndarray:
    """Fuse ranked piece arrays with Python-dict-equivalent stable tie order."""

    piece_ids, _ = stable_weighted_rrf_with_scores(
        piece_lists,
        weights,
        rrf_k=rrf_k,
        top_k=top_k,
    )
    return piece_ids


def stable_weighted_rrf_with_scores(
    piece_lists: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    rrf_k: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable weighted-RRF piece IDs and their aggregate scores."""

    if len(piece_lists) != len(weights):
        raise ValueError("weights must align with piece_lists")
    if rrf_k < 1 or top_k < 1:
        raise ValueError("rrf_k and top_k must be positive")
    id_parts = []
    score_parts = []
    for piece_ids, weight in zip(piece_lists, weights, strict=True):
        ids = np.asarray(piece_ids, dtype=np.int64)
        ids = ids[ids >= 0]
        if not len(ids) or weight <= 0:
            continue
        id_parts.append(ids)
        score_parts.append(
            float(weight)
            / (rrf_k + np.arange(1, len(ids) + 1, dtype=np.float64))
        )
    if not id_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    all_ids = np.concatenate(id_parts)
    all_scores = np.concatenate(score_parts)
    unique_ids, first_positions, inverse = np.unique(
        all_ids,
        return_index=True,
        return_inverse=True,
    )
    scores = np.bincount(inverse, weights=all_scores, minlength=len(unique_ids))
    # Python's sorted() is stable.  first_positions reproduces the insertion
    # order of the original branch/crop loops when aggregate scores tie.
    order = np.lexsort((first_positions, -scores))
    selected = order[:top_k]
    return unique_ids[selected], scores[selected]


def constrain_piece_ids_to_pool(
    pool_piece_ids: np.ndarray,
    reranked_piece_ids: np.ndarray,
) -> np.ndarray:
    pool = [int(value) for value in pool_piece_ids]
    allowed = set(pool)
    output = [int(value) for value in reranked_piece_ids if int(value) in allowed]
    present = set(output)
    output.extend(value for value in pool if value not in present)
    return np.asarray(output, dtype=np.int64)


def load_compact_piece_ranks(
    path: Path,
    *,
    candidate_k: int,
    include_scores: bool = False,
) -> tuple[list[str], np.ndarray] | tuple[
    list[str], np.ndarray, np.ndarray | None
]:
    with np.load(path, allow_pickle=False) as cached:
        if int(cached["candidate_k"].tolist()) != candidate_k:
            raise ValueError(f"Candidate-k mismatch in {path}")
        paths = [str(value) for value in cached["paths"].tolist()]
        piece_ids = np.asarray(cached["piece_ids"], dtype=np.int64)
        scores = (
            np.asarray(cached["scores"], dtype=np.float32)
            if "scores" in cached
            else None
        )
    if piece_ids.shape != (len(paths), candidate_k):
        raise ValueError(f"Invalid piece-rank cache shape in {path}")
    if scores is not None and scores.shape != piece_ids.shape:
        raise ValueError(f"Invalid score-cache shape in {path}")
    if include_scores:
        return paths, piece_ids, scores
    return paths, piece_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate weighted rank ensembles directly from compact per-crop "
            "piece-rank caches without rebuilding millions of result dictionaries."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--branch-search-cache",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--branch-weights", type=float, nargs="+")
    parser.add_argument("--last-branch-weight-grid", type=float, nargs="+")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--slot", choices=SLOTS, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--branch-rrf-k", type=int, default=1)
    parser.add_argument("--crop-rrf-k", type=int, default=1)
    parser.add_argument(
        "--crop-fusion",
        choices=("rrf", "max"),
        default="rrf",
        help="Fuse multiple crops with weighted RRF or weighted maximum score.",
    )
    parser.add_argument(
        "--fusion-order",
        choices=("branch-then-crop", "crop-then-branch"),
        default="branch-then-crop",
        help=(
            "Fuse branch ranks per crop first, or preserve each branch's raw-score "
            "crop fusion before combining branch ranks."
        ),
    )
    parser.add_argument("--final-pool-branches", type=int, default=2)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()

    if args.branch_weights and args.last_branch_weight_grid:
        parser.error(
            "--branch-weights and --last-branch-weight-grid are mutually exclusive"
        )
    branch_count = len(args.branch_search_cache)
    if not 1 <= args.final_pool_branches <= branch_count:
        parser.error("--final-pool-branches must be within the branch count")
    weight_sets = [tuple(args.branch_weights or (1.0,) * branch_count)]
    if len(weight_sets[0]) != branch_count:
        parser.error("--branch-weights must align with caches")
    if args.last_branch_weight_grid:
        weight_sets = [
            (1.0,) * (branch_count - 1) + (float(weight),)
            for weight in args.last_branch_weight_grid
        ]
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)

    cached = [
        load_compact_piece_ranks(
            path,
            candidate_k=args.candidate_k,
            include_scores=True,
        )
        for path in args.branch_search_cache
    ]
    paths = cached[0][0]
    if any(branch_paths != paths for branch_paths, _, _ in cached[1:]):
        raise ValueError("Compact branch caches do not contain identical query paths")
    piece_ranks = [values for _, values, _ in cached]
    piece_scores = [values for _, _, values in cached]
    if args.fusion_order == "crop-then-branch" and any(
        values is None for values in piece_scores
    ):
        raise ValueError(
            "crop-then-branch fusion requires caches containing raw search scores"
        )
    path_index = {path: index for index, path in enumerate(paths)}

    metadata = [
        row
        for row in read_jsonl(args.metadata)
        if str(row.get("slot")) == args.slot
    ]
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
        slot=args.slot,
    )

    crop_first_branch_rankings = None
    if args.fusion_order == "crop-then-branch":
        crop_first_branch_rankings = _fuse_raw_crops_per_branch(
            piece_ranks,
            piece_scores,
            records_by_example=context["records"],
            path_index=path_index,
            candidate_k=args.candidate_k,
        )
        pool_by_example = _fuse_branch_rankings_by_example(
            crop_first_branch_rankings[: args.final_pool_branches],
            (1.0,) * args.final_pool_branches,
            records_by_example=context["records"],
            branch_rrf_k=args.branch_rrf_k,
            candidate_k=args.candidate_k,
        )
    else:
        pool_by_example = _fuse_rankings_for_examples(
            piece_ranks[: args.final_pool_branches],
            (1.0,) * args.final_pool_branches,
            records_by_example=context["records"],
            path_index=path_index,
            branch_rrf_k=args.branch_rrf_k,
            crop_rrf_k=args.crop_rrf_k,
            crop_fusion=args.crop_fusion,
            candidate_k=args.candidate_k,
        )

    experiments = []
    for weights in weight_sets:
        if args.fusion_order == "crop-then-branch":
            if crop_first_branch_rankings is None:
                raise RuntimeError("Missing crop-first branch rankings")
            reranked_by_example = _fuse_branch_rankings_by_example(
                crop_first_branch_rankings,
                weights,
                records_by_example=context["records"],
                branch_rrf_k=args.branch_rrf_k,
                candidate_k=args.candidate_k,
            )
        else:
            reranked_by_example = _fuse_rankings_for_examples(
                piece_ranks,
                weights,
                records_by_example=context["records"],
                path_index=path_index,
                branch_rrf_k=args.branch_rrf_k,
                crop_rrf_k=args.crop_rrf_k,
                crop_fusion=args.crop_fusion,
                candidate_k=args.candidate_k,
            )
        final_by_example = {
            example_id: constrain_piece_ids_to_pool(
                pool_by_example.get(example_id, np.empty(0, dtype=np.int64)),
                reranked_by_example.get(example_id, np.empty(0, dtype=np.int64)),
            )
            for example_id in pool_by_example
        }
        cases = _rank_cases(
            context["examples"],
            final_by_example,
            metadata,
            slot=args.slot,
        )
        experiments.append(
            {
                "branch_weights": list(weights),
                "summary": summarize_ranked_cases(cases, cutoffs),
                "cases": cases,
            }
        )

    payload = {
        "slot": args.slot,
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "metadata": portable_path(args.metadata),
        "branch_search_caches": [
            portable_path(path) for path in args.branch_search_cache
        ],
        "candidate_k": args.candidate_k,
        "branch_rrf_k": args.branch_rrf_k,
        "crop_rrf_k": args.crop_rrf_k,
        "crop_fusion": args.crop_fusion,
        "fusion_order": args.fusion_order,
        "final_pool_branches": args.final_pool_branches,
        "cutoffs": list(cutoffs),
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": portable_path(args.output),
                "summaries": [
                    {
                        "branch_weights": row["branch_weights"],
                        "summary": row["summary"],
                    }
                    for row in experiments
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _fuse_all_query_crops(
    branch_piece_ranks: Sequence[np.ndarray],
    branch_weights: Sequence[float],
    *,
    branch_rrf_k: int,
    candidate_k: int,
) -> list[np.ndarray]:
    query_count = branch_piece_ranks[0].shape[0]
    return [
        stable_weighted_rrf(
            [branch[query_index] for branch in branch_piece_ranks],
            branch_weights,
            rrf_k=branch_rrf_k,
            top_k=candidate_k,
        )
        for query_index in range(query_count)
    ]


def _fuse_all_query_crops_with_scores(
    branch_piece_ranks: Sequence[np.ndarray],
    branch_weights: Sequence[float],
    *,
    branch_rrf_k: int,
    candidate_k: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    query_count = branch_piece_ranks[0].shape[0]
    fused = [
        stable_weighted_rrf_with_scores(
            [branch[query_index] for branch in branch_piece_ranks],
            branch_weights,
            rrf_k=branch_rrf_k,
            top_k=candidate_k,
        )
        for query_index in range(query_count)
    ]
    return [row[0] for row in fused], [row[1] for row in fused]


def _fuse_rankings_for_examples(
    branch_piece_ranks: Sequence[np.ndarray],
    branch_weights: Sequence[float],
    *,
    records_by_example: Mapping[str, Sequence[dict[str, Any]]],
    path_index: Mapping[str, int],
    branch_rrf_k: int,
    crop_rrf_k: int,
    crop_fusion: str,
    candidate_k: int,
) -> dict[str, np.ndarray]:
    if crop_fusion == "rrf":
        crop_rankings = _fuse_all_query_crops(
            branch_piece_ranks,
            branch_weights,
            branch_rrf_k=branch_rrf_k,
            candidate_k=candidate_k,
        )
        return _fuse_all_examples(
            records_by_example,
            path_index,
            crop_rankings,
            crop_rrf_k=crop_rrf_k,
            candidate_k=candidate_k,
        )
    if crop_fusion == "max":
        crop_rankings, crop_scores = _fuse_all_query_crops_with_scores(
            branch_piece_ranks,
            branch_weights,
            branch_rrf_k=branch_rrf_k,
            candidate_k=candidate_k,
        )
        return _fuse_all_examples_max(
            records_by_example,
            path_index,
            crop_rankings,
            crop_scores,
            candidate_k=candidate_k,
        )
    raise ValueError(f"Unsupported crop fusion: {crop_fusion}")


def _fuse_crops_then_branches(
    branch_piece_ranks: Sequence[np.ndarray],
    branch_piece_scores: Sequence[np.ndarray | None],
    branch_weights: Sequence[float],
    *,
    records_by_example: Mapping[str, Sequence[dict[str, Any]]],
    path_index: Mapping[str, int],
    branch_rrf_k: int,
    candidate_k: int,
) -> dict[str, np.ndarray]:
    """Fuse raw crop scores within each branch, then RRF branch rankings."""

    branch_rankings = _fuse_raw_crops_per_branch(
        branch_piece_ranks,
        branch_piece_scores,
        records_by_example=records_by_example,
        path_index=path_index,
        candidate_k=candidate_k,
    )
    return _fuse_branch_rankings_by_example(
        branch_rankings,
        branch_weights,
        records_by_example=records_by_example,
        branch_rrf_k=branch_rrf_k,
        candidate_k=candidate_k,
    )


def _fuse_raw_crops_per_branch(
    branch_piece_ranks: Sequence[np.ndarray],
    branch_piece_scores: Sequence[np.ndarray | None],
    *,
    records_by_example: Mapping[str, Sequence[dict[str, Any]]],
    path_index: Mapping[str, int],
    candidate_k: int,
) -> list[dict[str, np.ndarray]]:
    """Build each branch's exact raw-score max-fused example rankings once."""

    if len(branch_piece_ranks) != len(branch_piece_scores):
        raise ValueError("Branch ranks and raw scores must align")
    if any(scores is None for scores in branch_piece_scores):
        raise ValueError("Raw branch scores are required for crop-first fusion")
    return [
        _fuse_all_examples_max(
            records_by_example,
            path_index,
            piece_ids,
            scores,
            candidate_k=candidate_k,
        )
        for piece_ids, scores in zip(
            branch_piece_ranks,
            branch_piece_scores,
            strict=True,
        )
        if scores is not None
    ]


def _fuse_branch_rankings_by_example(
    branch_rankings: Sequence[Mapping[str, np.ndarray]],
    branch_weights: Sequence[float],
    *,
    records_by_example: Mapping[str, Sequence[dict[str, Any]]],
    branch_rrf_k: int,
    candidate_k: int,
) -> dict[str, np.ndarray]:
    """Fuse already crop-normalized branch rankings by weighted RRF."""

    if len(branch_rankings) != len(branch_weights):
        raise ValueError("Branch weights must align with branch rankings")
    output = {}
    for example_id in records_by_example:
        piece_lists = [
            branch.get(str(example_id), np.empty(0, dtype=np.int64))
            for branch in branch_rankings
        ]
        ranked = stable_weighted_rrf(
            piece_lists,
            branch_weights,
            rrf_k=branch_rrf_k,
            top_k=candidate_k,
        )
        if len(ranked):
            output[str(example_id)] = ranked
    return output


def _fuse_all_examples(
    records_by_example: Mapping[str, Sequence[dict[str, Any]]],
    path_index: Mapping[str, int],
    crop_rankings: Sequence[np.ndarray],
    *,
    crop_rrf_k: int,
    candidate_k: int,
) -> dict[str, np.ndarray]:
    output = {}
    for example_id, records in records_by_example.items():
        piece_lists = []
        weights = []
        for record in records:
            weight = crop_weight(record)
            query_index = path_index.get(str(record["path"]))
            if weight <= 0 or query_index is None:
                continue
            piece_lists.append(crop_rankings[query_index])
            weights.append(weight)
        if piece_lists:
            output[str(example_id)] = stable_weighted_rrf(
                piece_lists,
                weights,
                rrf_k=crop_rrf_k,
                top_k=candidate_k,
            )
    return output


def _fuse_all_examples_max(
    records_by_example: Mapping[str, Sequence[dict[str, Any]]],
    path_index: Mapping[str, int],
    crop_rankings: Sequence[np.ndarray],
    crop_scores: Sequence[np.ndarray],
    *,
    candidate_k: int,
) -> dict[str, np.ndarray]:
    """Fuse crops by their weighted maximum branch-RRF score."""

    output = {}
    for example_id, records in records_by_example.items():
        scores: dict[int, float] = {}
        for record in records:
            weight = crop_weight(record)
            query_index = path_index.get(str(record["path"]))
            if weight <= 0 or query_index is None:
                continue
            for piece_id, score in zip(
                crop_rankings[query_index],
                crop_scores[query_index],
                strict=True,
            ):
                if int(piece_id) < 0 or not np.isfinite(score):
                    continue
                value = weight * float(score)
                key = int(piece_id)
                if key not in scores or value > scores[key]:
                    scores[key] = value
        if scores:
            ranked = sorted(scores, key=lambda piece_id: -scores[piece_id])
            output[str(example_id)] = np.asarray(
                ranked[:candidate_k],
                dtype=np.int64,
            )
    return output


def _rank_cases(
    examples: Sequence[dict[str, Any]],
    rankings: Mapping[str, np.ndarray],
    metadata: Sequence[dict[str, Any]],
    *,
    slot: str,
) -> list[dict[str, Any]]:
    available_ids = available_pieces_by_slot(list(metadata))[slot]
    available_keys = available_appearance_keys_by_slot(list(metadata))[slot]
    appearance_by_piece = {}
    for row in metadata:
        for piece in row["pieces"]:
            appearance_by_piece.setdefault(
                int(piece["piece_id"]),
                appearance_name_key(piece.get("name_en")),
            )
    cases = []
    for example in examples:
        example_id = str(example["example_id"])
        label = (example.get("core_slot_labels") or {}).get(slot)
        base = {
            "example_id": example_id,
            "glamour_id": int(example["glamour_id"]),
            "model_gender": example.get("model_gender"),
            "model_race": example.get("model_race"),
            "expected_name": label.get("item_name") if label else None,
            "rank": None,
        }
        if not label or label.get("piece_match_type") == "empty":
            cases.append({**base, "status": "empty_label"})
            continue
        expected_ids = {int(value) for value in label.get("piece_ids", [])}
        if not expected_ids:
            cases.append({**base, "status": "missing_truth_piece_id"})
            continue
        expected_key = appearance_name_key(label.get("item_name"))
        eligible_ids = expected_ids & available_ids
        if not eligible_ids and not (
            expected_key and expected_key in available_keys
        ):
            cases.append({**base, "status": "truth_not_in_index"})
            continue
        ranked_ids = rankings.get(example_id)
        if ranked_ids is None or not len(ranked_ids):
            cases.append({**base, "status": "no_usable_crop"})
            continue
        rank = next(
            (
                index
                for index, piece_id in enumerate(ranked_ids, start=1)
                if int(piece_id) in eligible_ids
                or appearance_by_piece.get(int(piece_id)) == expected_key
            ),
            None,
        )
        cases.append({**base, "status": "eligible", "rank": rank})
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
