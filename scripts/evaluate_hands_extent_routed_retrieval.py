from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl  # noqa: E402
from ffxiv_prism.paths import CACHE_DIR, portable_path  # noqa: E402
from ffxiv_prism.torch_device import resolve_torch_device  # noqa: E402
from scripts.build_hands_canonical_extent_profiles import (  # noqa: E402
    EXTENT_CLASSES,
)
from scripts.evaluate_glamour_embedding_model import search_crops  # noqa: E402
from scripts.evaluate_retrieval_funnel import (  # noqa: E402
    ROLE_PROFILES,
    VISIBLE_WEIGHTS,
    appearance_name_key,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    build_fold_context,
    crop_record_weight,
    release_cuda,
    summarize_ranked_cases,
    validate_cutoffs,
)
from scripts.train_glamour_query_encoder import (  # noqa: E402
    encode_records,
    load_query_head_from_checkpoint,
    load_slot_eligibility,
)


def load_extent_map(
    profiles: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], str]:
    output = {}
    for row in profiles:
        extent = str(row.get("extent_class") or "")
        if extent not in EXTENT_CLASSES:
            raise ValueError(f"Unsupported extent class: {extent}")
        key = (
            int(row["piece_id"]),
            str(row.get("model_gender") or "").lower(),
        )
        current = output.get(key)
        if current is not None and current != extent:
            raise ValueError(f"Conflicting extent profiles for {key}")
        output[key] = extent
    return output


def partition_candidate_index(
    metadata: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    extent_by_piece_gender: Mapping[tuple[int, str], str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    if len(metadata) != len(embeddings):
        raise ValueError("Candidate metadata and embeddings must align")
    rows_by_class = {extent: [] for extent in EXTENT_CLASSES}
    indexes_by_class = {extent: [] for extent in EXTENT_CLASSES}
    missing: set[tuple[int, str]] = set()
    for index, row in enumerate(metadata):
        if str(row.get("slot")) != "HANDS":
            continue
        gender = str(row.get("model_gender") or "").lower()
        pieces_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for piece in row.get("pieces") or []:
            key = (int(piece["piece_id"]), gender)
            extent = extent_by_piece_gender.get(key)
            if extent is None:
                missing.add(key)
                continue
            pieces_by_class[extent].append(dict(piece))
        for extent, pieces in pieces_by_class.items():
            output_row = dict(row)
            output_row["pieces"] = pieces
            output_row["hands_extent_class"] = extent
            rows_by_class[extent].append(output_row)
            indexes_by_class[extent].append(index)
    arrays = {
        extent: np.asarray(
            embeddings[np.asarray(indexes_by_class[extent], dtype=np.int64)]
        )
        for extent in EXTENT_CLASSES
    }
    diagnostics = {
        "candidate_rows_by_extent": {
            extent: len(rows_by_class[extent]) for extent in EXTENT_CLASSES
        },
        "candidate_pieces_by_extent": {
            extent: len(
                {
                    int(piece["piece_id"])
                    for row in rows_by_class[extent]
                    for piece in row["pieces"]
                }
            )
            for extent in EXTENT_CLASSES
        },
        "missing_piece_gender_count": len(missing),
    }
    return rows_by_class, arrays, diagnostics


def accumulate_rrf_scores(
    records: Sequence[Mapping[str, Any]],
    search_results_by_crop: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    rrf_k: int,
) -> tuple[dict[int, float], dict[int, str]]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    scores: dict[int, float] = {}
    keys: dict[int, str] = {}
    role_weights = ROLE_PROFILES["balanced"]
    for record in records:
        if VISIBLE_WEIGHTS.get(str(record.get("visibility")), 0.0) <= 0:
            continue
        weight = crop_record_weight(record, role_weights, {})
        if weight <= 0:
            continue
        for offset, result in enumerate(
            search_results_by_crop.get(str(record["path"]), ()),
            start=1,
        ):
            piece_id = int(result["piece_id"])
            rank = int(result.get("rank", offset))
            scores[piece_id] = scores.get(piece_id, 0.0) + weight / (rrf_k + rank)
            keys[piece_id] = appearance_name_key(result.get("name_en"))
    return scores, keys


def rank_cases(
    examples: Sequence[Mapping[str, Any]],
    raw_records: Mapping[str, Sequence[Mapping[str, Any]]],
    extent_records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    raw_search: Mapping[str, Sequence[Mapping[str, Any]]],
    extent_search: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    metadata: Sequence[Mapping[str, Any]],
    *,
    extent_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    if extent_weight < 0:
        raise ValueError("extent_weight must not be negative")
    available_ids = available_pieces_by_slot(metadata)["HANDS"]
    available_keys = available_appearance_keys_by_slot(metadata)["HANDS"]
    cases = []
    for example in examples:
        example_id = str(example["example_id"])
        label = (example.get("core_slot_labels") or {}).get("HANDS") or None
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
        expected_ids = {int(value) for value in label.get("piece_ids") or []}
        if not expected_ids:
            cases.append({**base, "status": "missing_truth_piece_id"})
            continue
        expected_key = appearance_name_key(label.get("item_name"))
        eligible_ids = expected_ids & available_ids
        if not eligible_ids and not (expected_key and expected_key in available_keys):
            cases.append({**base, "status": "truth_not_in_index"})
            continue

        raw_scores, keys = accumulate_rrf_scores(
            raw_records.get(example_id, ()), raw_search, rrf_k=rrf_k
        )
        routed_scores: dict[int, float] = {}
        for extent in EXTENT_CLASSES:
            scores, routed_keys = accumulate_rrf_scores(
                extent_records[extent].get(example_id, ()),
                extent_search[extent],
                rrf_k=rrf_k,
            )
            routed_scores.update(scores)
            keys.update(routed_keys)
        piece_ids = set(raw_scores) | set(routed_scores)
        final_scores = {
            piece_id: raw_scores.get(piece_id, 0.0)
            + extent_weight * routed_scores.get(piece_id, 0.0)
            for piece_id in piece_ids
        }
        ranked = sorted(piece_ids, key=final_scores.__getitem__, reverse=True)
        rank = next(
            (
                offset
                for offset, piece_id in enumerate(ranked, start=1)
                if piece_id in eligible_ids or keys.get(piece_id) == expected_key
            ),
            None,
        )
        cases.append({**base, "status": "eligible", "rank": rank})
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate-side HANDS extent routing. Each canonical "
            "candidate competes only against the wrist/medium/long query view "
            "matching its audited extent profile, then adds that evidence to "
            "the raw-query RRF baseline."
        )
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--slot-eligibility-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument(
        "--fold-role", choices=("train", "validation"), default="validation"
    )
    parser.add_argument("--raw-regions-dir", type=Path, required=True)
    for extent in EXTENT_CLASSES:
        parser.add_argument(f"--{extent}-regions-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--extent-profiles", type=Path, required=True)
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--extent-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
    args = parser.parse_args()
    if any(weight < 0 for weight in args.extent_weights):
        parser.error("extent weights must not be negative")
    device = resolve_torch_device(args.device)
    cutoffs = validate_cutoffs(args.cutoffs, args.candidate_k)

    examples = read_jsonl(args.examples)
    eligibility = load_slot_eligibility(args.slot_eligibility_manifest)
    quality_by_id = {
        str(example["example_id"]): (
            "HANDS" in eligibility.get(str(example["example_id"]), set())
        )
        for example in examples
    }
    region_dirs = {
        "raw": args.raw_regions_dir,
        **{extent: getattr(args, f"{extent}_regions_dir") for extent in EXTENT_CLASSES},
    }
    contexts = {
        name: build_fold_context(
            fold=args.fold,
            fold_manifest=args.fold_manifest,
            all_examples=examples,
            quality_by_id=quality_by_id,
            regions_dir=regions_dir,
            slot="HANDS",
            fold_role=args.fold_role,
        )
        for name, regions_dir in region_dirs.items()
    }
    expected_ids = {
        str(example["example_id"]) for example in contexts["raw"]["examples"]
    }
    for name, context in contexts.items():
        if {
            str(example["example_id"]) for example in context["examples"]
        } != expected_ids:
            raise ValueError(f"Extent context does not match raw context: {name}")

    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(args.index_dir / "embeddings.npy", mmap_mode="r")
    extent_map = load_extent_map(read_jsonl(args.extent_profiles))
    partition_rows, partition_embeddings, partition_diagnostics = (
        partition_candidate_index(metadata, candidate_embeddings, extent_map)
    )
    if partition_diagnostics["missing_piece_gender_count"]:
        raise ValueError(
            "Extent profiles do not cover the full candidate index: "
            f"{partition_diagnostics['missing_piece_gender_count']} missing"
        )

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    query_head = load_query_head_from_checkpoint(args.query_checkpoint, model, device)
    query_embeddings = {}
    for name, context in contexts.items():
        query_embeddings[name] = encode_records(
            model,
            query_head,
            context["records"],
            args.batch_size,
            device,
            pose_normalize_hands=False,
        )
        print(f"encoded={name} crops={len(query_embeddings[name])}", flush=True)

    raw_search = search_crops(
        metadata,
        candidate_embeddings,
        contexts["raw"]["records"],
        query_embeddings["raw"],
        args.candidate_k,
        device=str(device),
    )
    extent_search = {}
    for extent in EXTENT_CLASSES:
        extent_search[extent] = search_crops(
            partition_rows[extent],
            partition_embeddings[extent],
            contexts[extent]["records"],
            query_embeddings[extent],
            args.candidate_k,
            device=str(device),
        )
        print(
            f"searched={extent} crops={len(extent_search[extent])}",
            flush=True,
        )

    evaluated = []
    for weight in args.extent_weights:
        cases = rank_cases(
            contexts["raw"]["examples"],
            contexts["raw"]["records"],
            {extent: contexts[extent]["records"] for extent in EXTENT_CLASSES},
            raw_search,
            extent_search,
            metadata,
            extent_weight=weight,
            rrf_k=args.rrf_k,
        )
        summary = summarize_ranked_cases(cases, cutoffs)
        summary["extent_weight"] = weight
        evaluated.append((summary, cases))
        print(
            f"weight={weight:g} top10={summary['hits']['10']} "
            f"top100={summary['hits']['100']} mrr={summary['mrr']:.6f}",
            flush=True,
        )
    best_summary, best_cases = max(
        evaluated,
        key=lambda item: (
            item[0]["hits"].get("10", 0),
            item[0]["mrr"],
            item[0]["hits"].get("100", 0),
        ),
    )
    payload = {
        "version": 1,
        "status": "development_fold_only",
        "slot": "HANDS",
        "fold": args.fold,
        "fold_role": args.fold_role,
        "blind_test_accessed": False,
        "best_summary": best_summary,
        "sweep_summaries": [summary for summary, _ in evaluated],
        "partition_diagnostics": partition_diagnostics,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "query_checkpoint": portable_path(args.query_checkpoint),
        "extent_profiles": portable_path(args.extent_profiles),
        "fold_manifest": portable_path(args.fold_manifest),
        "region_dirs": {
            name: portable_path(path) for name, path in region_dirs.items()
        },
        "cases": best_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "best": best_summary}, indent=2))
    del model, query_embeddings, raw_search, extent_search
    release_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
