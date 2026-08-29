from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.paths import portable_path
from ffxiv_prism.matching.inference import fuse_crop_results
from scripts.evaluate_cached_query_retrieval import load_query_embeddings
from scripts.evaluate_glamour_embedding_model import read_jsonl, search_crops
from scripts.evaluate_retrieval_funnel import (
    HANDS_CANDIDATE_CROP_ROUTES,
    HEAD_CANDIDATE_CROP_ROUTES,
    SLOTS,
    build_fold_context,
    rank_body_cases,
    appearance_name_key,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    summarize_ranked_cases,
    validate_cutoffs,
)
from scripts.train_glamour_query_encoder import load_slot_eligibility


def fuse_branch_searches(
    searches: Sequence[dict[str, list[dict[str, Any]]]],
    *,
    rrf_k: int,
    candidate_k: int,
    candidate_pool: dict[str, list[dict[str, Any]]] | None = None,
    branch_weights: Sequence[float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not searches:
        raise ValueError("At least one retrieval branch is required")
    if rrf_k < 1:
        raise ValueError("branch-rrf-k must be positive")
    paths = set(searches[0])
    if any(set(search) != paths for search in searches[1:]):
        raise ValueError("All branches must contain the same query crop paths")
    if candidate_pool is not None and set(candidate_pool) != paths:
        raise ValueError("Candidate pool must contain the same query crop paths")
    weights = tuple(branch_weights or (1.0,) * len(searches))
    if len(weights) != len(searches):
        raise ValueError("branch_weights must match the branch count")
    if any(weight < 0 for weight in weights) or not any(weights):
        raise ValueError("branch_weights must be non-negative with one positive value")

    merged = {}
    for path in sorted(paths):
        scores: dict[int, float] = {}
        exemplar: dict[int, dict[str, Any]] = {}
        allowed_ids = (
            {int(row["piece_id"]) for row in candidate_pool[path]}
            if candidate_pool is not None
            else None
        )
        for search, branch_weight in zip(searches, weights, strict=True):
            if branch_weight == 0:
                continue
            for row in search[path]:
                piece_id = int(row["piece_id"])
                if allowed_ids is not None and piece_id not in allowed_ids:
                    continue
                scores[piece_id] = scores.get(piece_id, 0.0) + branch_weight / (
                    rrf_k + int(row["rank"])
                )
                exemplar.setdefault(piece_id, row)
        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:candidate_k]
        merged[path] = [
            {
                **exemplar[piece_id],
                "rank": rank,
                "score": scores[piece_id],
            }
            for rank, piece_id in enumerate(ranked_ids, start=1)
        ]
    return merged


def save_branch_search_cache(
    path: Path,
    search: dict[str, list[dict[str, Any]]],
    *,
    candidate_k: int,
    index_dir: Path,
    query_cache: Path,
) -> None:
    paths = sorted(search)
    piece_ids = np.full((len(paths), candidate_k), -1, dtype=np.int32)
    scores = np.full((len(paths), candidate_k), np.nan, dtype=np.float32)
    for row_index, query_path in enumerate(paths):
        rows = search[query_path]
        piece_ids[row_index, : len(rows)] = [
            int(row["piece_id"]) for row in rows
        ]
        scores[row_index, : len(rows)] = [float(row["score"]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        paths=np.asarray(paths),
        piece_ids=piece_ids,
        scores=scores,
        candidate_k=np.asarray(candidate_k),
        index_dir=np.asarray(str(index_dir)),
        query_cache=np.asarray(str(query_cache)),
    )


def load_branch_search_cache(
    path: Path,
    metadata: Sequence[dict[str, Any]],
    *,
    candidate_k: int,
    index_dir: Path,
    query_cache: Path,
) -> dict[str, list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as cached:
        if int(cached["candidate_k"].tolist()) != candidate_k:
            raise ValueError(f"Candidate-k mismatch in search cache: {path}")
        if str(cached["index_dir"].tolist()) != str(index_dir):
            raise ValueError(f"Index mismatch in search cache: {path}")
        if str(cached["query_cache"].tolist()) != str(query_cache):
            raise ValueError(f"Query-cache mismatch in search cache: {path}")
        paths = [str(value) for value in cached["paths"].tolist()]
        piece_ids = np.asarray(cached["piece_ids"], dtype=np.int64)
        scores = (
            np.asarray(cached["scores"], dtype=np.float32)
            if "scores" in cached
            else np.zeros(piece_ids.shape, dtype=np.float32)
        )

    exemplar_by_piece: dict[int, dict[str, Any]] = {}
    for row in metadata:
        for piece in row["pieces"]:
            piece_id = int(piece["piece_id"])
            exemplar_by_piece.setdefault(
                piece_id,
                {
                    **piece,
                    "image_row_id": row["image_row_id"],
                    "local_path": row["local_path"],
                    "source_image_url": row["source_image_url"],
                    "model_gender": row.get("model_gender"),
                    "race": row.get("race"),
                    "dye_state": row.get("dye_state"),
                },
            )
    search = {}
    for path_index, query_path in enumerate(paths):
        rows = []
        for rank, (piece_id, score) in enumerate(
            zip(piece_ids[path_index], scores[path_index], strict=True),
            start=1,
        ):
            if piece_id < 0:
                break
            exemplar = exemplar_by_piece.get(int(piece_id))
            if exemplar is None:
                raise ValueError(
                    f"Piece {piece_id} from search cache is absent from metadata"
                )
            rows.append(
                {
                    **exemplar,
                    "piece_id": int(piece_id),
                    "rank": rank,
                    "score": float(score),
                }
            )
        search[query_path] = rows
    return search


def select_slot_index_rows(
    metadata: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    slot: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Select one slot while preserving metadata/embedding row alignment."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Branch index row mismatch: "
            f"embeddings={embeddings.shape}, metadata_rows={len(metadata)}"
        )
    indices = [
        index
        for index, row in enumerate(metadata)
        if str(row.get("slot")) == slot
    ]
    if not indices:
        raise ValueError(f"Branch index contains no {slot} rows")
    selected_metadata = [metadata[index] for index in indices]
    if len(indices) == len(metadata):
        return selected_metadata, embeddings
    return selected_metadata, np.asarray(embeddings[indices], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse frozen retrieval branches by per-crop reciprocal ranks."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--branch",
        action="append",
        nargs=2,
        metavar=("INDEX_DIR", "QUERY_CACHE"),
        required=True,
        help="Repeat for each candidate-index/query-cache branch.",
    )
    parser.add_argument("--examples", type=Path, required=True)
    quality_group = parser.add_mutually_exclusive_group(required=True)
    quality_group.add_argument("--quality-manifest", type=Path)
    quality_group.add_argument("--slot-eligibility-manifest", type=Path)
    parser.add_argument("--slot", choices=SLOTS, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--candidate-crop-routing", action="store_true")
    parser.add_argument("--candidate-gender-routing", action="store_true")
    parser.add_argument("--query-crop")
    parser.add_argument("--candidate-crops", nargs="+")
    parser.add_argument(
        "--query-route",
        action="append",
        nargs="+",
        metavar="CROP",
        help=(
            "Repeat as QUERY_CROP CANDIDATE_CROP [CANDIDATE_CROP ...] to "
            "evaluate multiple explicit query/candidate crop routes."
        ),
    )
    parser.add_argument("--branch-rrf-k", type=int, default=1)
    parser.add_argument(
        "--branch-weights",
        type=float,
        nargs="+",
        help="Optional non-negative RRF weight for each --branch in order.",
    )
    parser.add_argument(
        "--last-branch-weight-grid",
        type=float,
        nargs="+",
        help=(
            "Evaluate several weights for the final branch after loading/searching "
            "all branches once; earlier branch weights stay at 1."
        ),
    )
    parser.add_argument(
        "--branch-search-cache",
        action="append",
        type=Path,
        help=(
            "Optional compact per-crop piece-rank cache; repeat once per branch. "
            "Existing caches are loaded and missing caches are created."
        ),
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Build or validate branch search caches, then stop before fusion.",
    )
    parser.add_argument(
        "--pool-branches",
        type=int,
        help=(
            "Build a fixed candidate pool from the first N branches, then let all "
            "branches rerank only that pool. This preserves deep candidate coverage."
        ),
    )
    parser.add_argument("--pool-rrf-k", type=int, default=1)
    parser.add_argument(
        "--final-pool-branches",
        type=int,
        help=(
            "Freeze the final per-example candidate set from the first N branches, "
            "then let the full ensemble reorder only those pieces."
        ),
    )
    parser.add_argument("--crop-fusion", choices=("max", "rrf"), default="max")
    parser.add_argument("--crop-rrf-k", type=int, default=1)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()

    if bool(args.query_crop) != bool(args.candidate_crops):
        parser.error("--query-crop and --candidate-crops must be used together")
    if args.query_crop and args.candidate_crop_routing:
        parser.error(
            "--query-crop cannot be combined with --candidate-crop-routing"
        )
    if args.query_route and (args.query_crop or args.candidate_crop_routing):
        parser.error(
            "--query-route cannot be combined with --query-crop or "
            "--candidate-crop-routing"
        )
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)
    if args.branch_weights and args.last_branch_weight_grid:
        parser.error(
            "--branch-weights and --last-branch-weight-grid are mutually exclusive"
        )
    branch_weights = tuple(args.branch_weights or (1.0,) * len(args.branch))
    if len(branch_weights) != len(args.branch):
        parser.error("--branch-weights must provide one value per --branch")
    if args.branch_search_cache and len(args.branch_search_cache) != len(args.branch):
        parser.error("Repeat --branch-search-cache once per --branch")
    metadata = [
        row
        for row in read_jsonl(args.metadata)
        if str(row.get("slot")) == args.slot
    ]
    expected_paths = [str(row["local_path"]) for row in metadata]
    examples = read_jsonl(args.examples)
    if args.quality_manifest:
        quality_by_id = {
            str(row["example_id"]): bool(row.get("passed"))
            for row in read_jsonl(args.quality_manifest)
        }
    else:
        eligibility = load_slot_eligibility(args.slot_eligibility_manifest)
        quality_by_id = {
            str(example["example_id"]): (
                args.slot
                in eligibility.get(str(example["example_id"]), set())
            )
            for example in examples
        }
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality_by_id,
        regions_dir=args.regions_dir,
        slot=args.slot,
    )

    query_routes = None
    if args.query_route:
        if any(len(route) < 2 for route in args.query_route):
            parser.error(
                "Each --query-route needs a query crop and at least one "
                "candidate crop"
            )
        query_routes = {
            str(route[0]): tuple(str(crop) for crop in route[1:])
            for route in args.query_route
        }
        if len(query_routes) != len(args.query_route):
            parser.error("--query-route query crop names must be unique")
        allowed_query_crops = set(query_routes)
        context["records"] = {
            example_id: [
                row
                for row in rows
                if str(row.get("name")) in allowed_query_crops
            ]
            for example_id, rows in context["records"].items()
        }
    elif args.query_crop:
        context["records"] = {
            example_id: [
                row
                for row in rows
                if str(row.get("name")) == args.query_crop
            ]
            for example_id, rows in context["records"].items()
        }

    candidate_crop_routes = query_routes
    if args.query_crop:
        candidate_crop_routes = {args.query_crop: tuple(args.candidate_crops or ())}
    if args.candidate_crop_routing:
        candidate_crop_routes = {
            "HEAD": HEAD_CANDIDATE_CROP_ROUTES,
            "HANDS": HANDS_CANDIDATE_CROP_ROUTES,
        }.get(args.slot)
        if candidate_crop_routes is None:
            raise ValueError(f"Candidate crop routing is not defined for {args.slot}")
    candidate_gender_by_example = None
    if args.candidate_gender_routing:
        candidate_gender_by_example = {
            str(example["example_id"]): str(example.get("model_gender", ""))
            for example in context["examples"]
        }

    searches = []
    branch_manifest = []
    for branch_index, (index_value, cache_value) in enumerate(args.branch):
        index_dir = Path(index_value)
        cache_path = Path(cache_value)
        all_branch_metadata = read_jsonl(index_dir / "metadata.jsonl")
        branch_matrix = np.load(index_dir / "embeddings.npy", mmap_mode="r")
        branch_metadata, branch_embeddings = select_slot_index_rows(
            all_branch_metadata,
            branch_matrix,
            args.slot,
        )
        if [str(row["local_path"]) for row in branch_metadata] != expected_paths:
            raise ValueError(f"Candidate metadata mismatch in {index_dir}")
        search_cache_path = (
            args.branch_search_cache[branch_index]
            if args.branch_search_cache
            else None
        )
        if search_cache_path is not None and search_cache_path.is_file():
            search = load_branch_search_cache(
                search_cache_path,
                branch_metadata,
                candidate_k=args.candidate_k,
                index_dir=index_dir,
                query_cache=cache_path,
            )
        else:
            search = search_crops(
                metadata,
                branch_embeddings,
                context["records"],
                load_query_embeddings(cache_path),
                args.candidate_k,
                device=args.device,
                candidate_crop_routes=candidate_crop_routes,
                candidate_gender_by_example=candidate_gender_by_example,
            )
            if search_cache_path is not None:
                save_branch_search_cache(
                    search_cache_path,
                    search,
                    candidate_k=args.candidate_k,
                    index_dir=index_dir,
                    query_cache=cache_path,
                )
        searches.append(search)
        branch_manifest.append(
            {
                "index_dir": portable_path(index_dir),
                "query_cache": portable_path(cache_path),
                "search_cache": portable_path(search_cache_path),
                "weight": branch_weights[branch_index],
            }
        )

    if args.cache_only:
        print(
            json.dumps(
                {"branch_caches": branch_manifest},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    weight_sets = [branch_weights]
    if args.last_branch_weight_grid:
        weight_sets = [
            (1.0,) * (len(searches) - 1) + (float(weight),)
            for weight in args.last_branch_weight_grid
        ]
    pool_weights = weight_sets[0]
    candidate_pool = None
    if args.pool_branches is not None:
        if not 1 <= args.pool_branches <= len(searches):
            parser.error("--pool-branches must be between 1 and the branch count")
        candidate_pool = fuse_branch_searches(
            searches[: args.pool_branches],
            rrf_k=args.pool_rrf_k,
            candidate_k=args.candidate_k,
            branch_weights=pool_weights[: args.pool_branches],
        )
    final_pool_search = None
    if args.final_pool_branches is not None:
        if not 1 <= args.final_pool_branches <= len(searches):
            parser.error("--final-pool-branches must be between 1 and branch count")
        final_pool_search = fuse_branch_searches(
            searches[: args.final_pool_branches],
            rrf_k=args.pool_rrf_k,
            candidate_k=args.candidate_k,
            branch_weights=pool_weights[: args.final_pool_branches],
        )

    def evaluate_weights(weights: tuple[float, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        merged = fuse_branch_searches(
            searches,
            rrf_k=args.branch_rrf_k,
            candidate_k=args.candidate_k,
            candidate_pool=candidate_pool,
            branch_weights=weights,
        )
        if final_pool_search is not None:
            current_cases = rank_cases_with_final_pool(
                context["examples"],
                context["records"],
                final_pool_search,
                merged,
                metadata,
                slot=args.slot,
                candidate_k=args.candidate_k,
                fusion=args.crop_fusion,
                rrf_k=args.crop_rrf_k,
            )
        else:
            current_cases = rank_body_cases(
                context["examples"],
                context["records"],
                merged,
                metadata,
                slot=args.slot,
                fusion=args.crop_fusion,
                rrf_k=args.crop_rrf_k,
            )
        return summarize_ranked_cases(current_cases, cutoffs), current_cases

    common_payload = {
        "slot": args.slot,
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "metadata": portable_path(args.metadata),
        "quality_manifest": portable_path(args.quality_manifest),
        "slot_eligibility_manifest": portable_path(
            args.slot_eligibility_manifest
        ),
        "branches": branch_manifest,
        "candidate_k": args.candidate_k,
        "candidate_crop_routing": args.candidate_crop_routing,
        "candidate_gender_routing": args.candidate_gender_routing,
        "query_routes": query_routes,
        "branch_fusion": "rrf",
        "branch_rrf_k": args.branch_rrf_k,
        "pool_branches": args.pool_branches,
        "pool_rrf_k": args.pool_rrf_k if args.pool_branches is not None else None,
        "final_pool_branches": args.final_pool_branches,
        "crop_fusion": args.crop_fusion,
        "crop_rrf_k": args.crop_rrf_k,
        "cutoffs": list(cutoffs),
    }
    if args.last_branch_weight_grid:
        experiments = []
        for weights in weight_sets:
            summary, cases = evaluate_weights(weights)
            experiments.append(
                {
                    "branch_weights": list(weights),
                    "summary": summary,
                    "cases": cases,
                }
            )
        payload = {
            **common_payload,
            "last_branch_weight_grid": list(args.last_branch_weight_grid),
            "experiments": experiments,
        }
        printed = [
            {"branch_weights": row["branch_weights"], "summary": row["summary"]}
            for row in experiments
        ]
    else:
        summary, cases = evaluate_weights(branch_weights)
        payload = {
            **common_payload,
            "branch_weights": list(branch_weights),
            "summary": summary,
            "cases": cases,
        }
        printed = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "summary": printed}, indent=2))
    return 0


def constrain_to_final_candidate_pool(
    pool_candidates: Sequence[dict[str, Any]],
    reranked_candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    pool_by_id = {int(row["piece_id"]): dict(row) for row in pool_candidates}
    output = [
        dict(row)
        for row in reranked_candidates
        if int(row["piece_id"]) in pool_by_id
    ]
    present = {int(row["piece_id"]) for row in output}
    output.extend(
        row for piece_id, row in pool_by_id.items() if piece_id not in present
    )
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def rank_cases_with_final_pool(
    examples: Sequence[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    pool_search: dict[str, list[dict[str, Any]]],
    reranked_search: dict[str, list[dict[str, Any]]],
    metadata: list[dict[str, Any]],
    *,
    slot: str,
    candidate_k: int,
    fusion: str,
    rrf_k: int,
) -> list[dict[str, Any]]:
    available_ids = available_pieces_by_slot(metadata)[slot]
    available_keys = available_appearance_keys_by_slot(metadata)[slot]
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
        example_records = records.get(example_id, [])
        if not example_records:
            cases.append({**base, "status": "no_usable_crop"})
            continue
        pool_candidates = fuse_crop_results(
            example_records,
            pool_search,
            top_k=candidate_k,
            fusion=fusion,
            rrf_k=rrf_k,
        )
        reranked_candidates = fuse_crop_results(
            example_records,
            reranked_search,
            top_k=candidate_k,
            fusion=fusion,
            rrf_k=rrf_k,
        )
        candidates = constrain_to_final_candidate_pool(
            pool_candidates, reranked_candidates
        )
        rank = next(
            (
                int(row["rank"])
                for row in candidates
                if int(row["piece_id"]) in eligible_ids
                or appearance_name_key(row.get("name_en")) == expected_key
            ),
            None,
        )
        cases.append({**base, "status": "eligible", "rank": rank})
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
