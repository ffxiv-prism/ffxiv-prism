from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ffxiv_prism.collectors.eorzea_collection.glamour_parser import appearance_name_key
from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import PROCESSED_DIR, portable_path, resolve_project_path
from ffxiv_prism.vision.retrieval import DEFAULT_INDEX_DIR, ClipImageEncoder, load_index, search_index

SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
VISIBLE_WEIGHTS = {
    "visible": 1.0,
    "partially_visible": 0.72,
    "occluded": 0.45,
    "pose_uncertain": 0.0,
    "out_of_frame": 0.0,
}
ROLE_PROFILES = {
    "context": {"primary": 0.75, "auxiliary": 0.65, "context": 1.0},
    "balanced": {"primary": 1.0, "auxiliary": 0.78, "context": 0.72},
    "primary": {"primary": 1.0, "auxiliary": 0.35, "context": 0.35},
    "overlap": {"primary": 1.0, "auxiliary": 0.95, "context": 0.65},
}
TAG_PENALTY_PROFILES = {
    "none": {},
    "mild": {
        "may_include_face": 0.80,
        "may_include_face_hair": 0.82,
        "may_include_face_hair_shoulders": 0.75,
        "may_include_hair": 0.85,
        "body_hands_overlap": 0.88,
        "legs_feet_overlap": 0.88,
        "may_include_sleeves": 0.88,
        "may_include_boots": 0.88,
        "may_include_lower_legs": 0.88,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end player glamour screenshot matching.")
    parser.add_argument("--examples", type=Path, default=PROCESSED_DIR / "glamour_examples.jsonl")
    parser.add_argument("--regions-dir", type=Path, default=Path("data/debug/outfit-regions/overlap-final-201"))
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "glamour_matching_eval_results.json")
    parser.add_argument("--embedding-cache", type=Path, default=PROCESSED_DIR / "glamour_crop_embeddings.npz")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--role-profile", choices=sorted(ROLE_PROFILES), default="balanced")
    parser.add_argument("--tag-penalty-profile", choices=sorted(TAG_PENALTY_PROFILES), default="mild")
    parser.add_argument("--aggregate", choices=("max", "sum"), default="max")
    parser.add_argument("--sweep", action="store_true", help="Evaluate all built-in weighting profiles and keep the best by appearance recall@1.")
    args = parser.parse_args()

    examples = read_jsonl(args.examples)
    index = load_index(args.index_dir)
    available_piece_ids = available_pieces_by_slot(index.metadata)
    available_appearance_keys = available_appearance_keys_by_slot(index.metadata)
    crop_records = collect_crop_records(examples, args.regions_dir)

    embeddings_by_crop = load_or_build_crop_embeddings(
        crop_records,
        args.embedding_cache,
        index.manifest["model_id"],
        args.device,
    )
    search_results_by_crop = search_all_crops(index, crop_records, embeddings_by_crop, args.candidate_k)

    configs: list[tuple[str, str, str]] = []
    if args.sweep:
        for role_profile in sorted(ROLE_PROFILES):
            for tag_profile in sorted(TAG_PENALTY_PROFILES):
                for aggregate in ("max", "sum"):
                    configs.append((role_profile, tag_profile, aggregate))
    else:
        configs.append((args.role_profile, args.tag_penalty_profile, args.aggregate))

    evaluated = []
    for role_profile, tag_profile, aggregate in configs:
        outcomes = evaluate_config(
            examples,
            crop_records,
            search_results_by_crop,
            available_piece_ids,
            available_appearance_keys,
            args.top_k,
            ROLE_PROFILES[role_profile],
            TAG_PENALTY_PROFILES[tag_profile],
            aggregate,
        )
        summary = summarize(outcomes, args.top_k)
        summary["role_profile"] = role_profile
        summary["tag_penalty_profile"] = tag_profile
        summary["aggregate"] = aggregate
        evaluated.append((summary, outcomes))

    best_summary, best_outcomes = max(
        evaluated,
        key=lambda item: (
            item[0]["appearance_recall_at_1"],
            item[0]["appearance_mrr"],
            item[0]["appearance_recall_at_k"],
            item[0]["exact_recall_at_1"],
        ),
    )
    payload = {
        **best_summary,
        "index_dir": portable_path(args.index_dir),
        "examples": portable_path(args.examples),
        "regions_dir": portable_path(args.regions_dir),
        "embedding_cache": portable_path(args.embedding_cache),
        "candidate_k": args.candidate_k,
        "sweep_results": [summary for summary, _ in evaluated],
        "cases": best_outcomes,
    }
    write_json(args.output, payload)
    print(
        f"cases={best_summary['case_count']} eligible={best_summary['eligible_case_count']} "
        f"exact@1={best_summary['exact_recall_at_1']:.3f} exact@{args.top_k}={best_summary['exact_recall_at_k']:.3f} "
        f"appearance@1={best_summary['appearance_recall_at_1']:.3f} appearance@{args.top_k}={best_summary['appearance_recall_at_k']:.3f} "
        f"appearance_mrr={best_summary['appearance_mrr']:.3f} "
        f"profile={best_summary['role_profile']}/{best_summary['tag_penalty_profile']}/{best_summary['aggregate']}"
    )
    for slot, row in best_summary["by_slot"].items():
        print(
            f"  {slot}: eligible={row['eligible_case_count']} exact@1={row['exact_recall_at_1']:.3f} "
            f"appearance@1={row['appearance_recall_at_1']:.3f} appearance@{args.top_k}={row['appearance_recall_at_k']:.3f} "
            f"appearance_mrr={row['appearance_mrr']:.3f}"
        )
    return 0


def collect_crop_records(examples: list[dict[str, Any]], regions_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        source_path = Path(example["local_path"])
        stem = f"{source_path.parent.name}-{source_path.stem}"
        manifest_path = regions_dir / stem / f"{stem}-regions.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        crops_by_name: dict[str, Path] = {}
        for path_value in manifest.get("crop_paths", []):
            crop_path = resolve_project_path(path_value)
            if crop_path is None:
                continue
            prefix = f"{stem}-"
            crop_stem = crop_path.stem
            crop_name = crop_stem[len(prefix):] if crop_stem.startswith(prefix) else crop_stem.rsplit("-", 1)[-1]
            crops_by_name[crop_name] = crop_path
        for region in manifest.get("regions", []):
            crop_path = crops_by_name.get(region["name"])
            if crop_path is None or not crop_path.is_file():
                continue
            by_example[example["example_id"]].append(
                {
                    "example_id": example["example_id"],
                    "slot": region["slot"],
                    "name": region["name"],
                    "path": crop_path,
                    "visibility": region["visibility"],
                    "confidence": float(region.get("confidence", 0.0)),
                    "role": region.get("role", "primary"),
                    "tags": tuple(region.get("tags", [])),
                }
            )
    return dict(by_example)


def load_or_build_crop_embeddings(
    crop_records: dict[str, list[dict[str, Any]]],
    cache_path: Path,
    model_id: str,
    device: str,
) -> dict[str, np.ndarray]:
    crop_paths = sorted({str(record["path"]) for records in crop_records.values() for record in records})
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        cached_paths = [str(path) for path in cached["paths"].tolist()]
        if cached_paths == crop_paths and str(cached["model_id"].tolist()) == model_id:
            embeddings = cached["embeddings"]
            return {path: embeddings[index] for index, path in enumerate(crop_paths)}

    encoder = ClipImageEncoder(model_id, device)
    paths = [Path(path) for path in crop_paths]
    embeddings = encoder.encode_paths(paths, batch_size=16)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, paths=np.asarray(crop_paths), embeddings=embeddings, model_id=np.asarray(model_id))
    return {path: embeddings[index] for index, path in enumerate(crop_paths)}


def search_all_crops(
    index: Any,
    crop_records: dict[str, list[dict[str, Any]]],
    embeddings_by_crop: dict[str, np.ndarray],
    candidate_k: int,
) -> dict[str, list[dict[str, Any]]]:
    results = {}
    for records in crop_records.values():
        for record in records:
            path = str(record["path"])
            if path in results:
                continue
            results[path] = search_index(index, embeddings_by_crop[path], record["slot"], top_k=candidate_k)
    return results


def evaluate_config(
    examples: list[dict[str, Any]],
    crop_records: dict[str, list[dict[str, Any]]],
    search_results_by_crop: dict[str, list[dict[str, Any]]],
    available_piece_ids: dict[str, set[int]],
    available_appearance_keys: dict[str, set[str]],
    top_k: int,
    role_weights: dict[str, float],
    tag_penalties: dict[str, float],
    aggregate: str,
) -> list[dict[str, Any]]:
    outcomes = []
    for example in examples:
        records = crop_records.get(example["example_id"], [])
        for slot in SLOTS:
            label = example.get("core_slot_labels", {}).get(slot)
            if not label or label.get("piece_match_type") == "empty":
                outcomes.append(base_outcome(example, slot, label, "empty_label"))
                continue
            expected_ids = {int(value) for value in label.get("piece_ids", [])}
            if not expected_ids:
                outcomes.append(base_outcome(example, slot, label, "missing_truth_piece_id"))
                continue
            expected_key = appearance_name_key(label.get("item_name"))
            eligible_ids = expected_ids & available_piece_ids.get(slot, set())
            appearance_eligible = bool(expected_key and expected_key in available_appearance_keys.get(slot, set()))
            if not eligible_ids and not appearance_eligible:
                outcomes.append(base_outcome(example, slot, label, "truth_not_in_index"))
                continue
            slot_records = [
                record
                for record in records
                if record["slot"] == slot and VISIBLE_WEIGHTS.get(record["visibility"], 0.0) > 0.0
            ]
            if not slot_records:
                outcomes.append(base_outcome(example, slot, label, "no_usable_crop"))
                continue

            scores: dict[int, dict[str, Any]] = {}
            evidence_by_piece: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for record in slot_records:
                crop_weight = crop_record_weight(record, role_weights, tag_penalties)
                if crop_weight <= 0:
                    continue
                for result in search_results_by_crop.get(str(record["path"]), []):
                    piece_id = int(result["piece_id"])
                    contribution = crop_weight * float(result["score"])
                    if piece_id not in scores:
                        scores[piece_id] = {
                            "piece_id": piece_id,
                            "score": 0.0,
                            "best_score": -999.0,
                            "best_result": result,
                            "best_crop": record,
                        }
                    if aggregate == "sum":
                        scores[piece_id]["score"] += contribution
                    elif contribution > scores[piece_id]["score"]:
                        scores[piece_id]["score"] = contribution
                    if contribution > scores[piece_id]["best_score"]:
                        scores[piece_id]["best_score"] = contribution
                        scores[piece_id]["best_result"] = result
                        scores[piece_id]["best_crop"] = record
                    evidence_by_piece[piece_id].append(
                        {
                            "crop": record["name"],
                            "crop_weight": crop_weight,
                            "raw_score": float(result["score"]),
                            "weighted_score": contribution,
                            "rank": int(result["rank"]),
                        }
                    )
            ranked = sorted(scores.values(), key=lambda row: row["score"], reverse=True)[:top_k]
            results = []
            for rank, row in enumerate(ranked, start=1):
                best = row["best_result"]
                results.append(
                    {
                        "rank": rank,
                        "piece_id": row["piece_id"],
                        "name_en": best.get("name_en"),
                        "appearance_key": appearance_name_key(best.get("name_en")),
                        "score": row["score"],
                        "image_row_id": best.get("image_row_id"),
                        "local_path": best.get("local_path"),
                        "source_example_id": best.get("source_example_id"),
                        "source_glamour_id": best.get("source_glamour_id"),
                        "neighbor_vote_fraction_5": best.get(
                            "neighbor_vote_fraction_5"
                        ),
                        "neighbor_vote_fraction_10": best.get(
                            "neighbor_vote_fraction_10"
                        ),
                        "neighbor_vote_fraction_20": best.get(
                            "neighbor_vote_fraction_20"
                        ),
                        "best_crop": row["best_crop"]["name"],
                        "evidence": sorted(
                            evidence_by_piece[row["piece_id"]],
                            key=lambda item: item["weighted_score"],
                            reverse=True,
                        )[:3],
                    }
                )
            rank = next((row["rank"] for row in results if int(row["piece_id"]) in eligible_ids), None)
            appearance_rank = next(
                (
                    row["rank"]
                    for row in results
                    if int(row["piece_id"]) in eligible_ids
                    or row.get("appearance_key") == expected_key
                ),
                None,
            )
            outcomes.append(
                {
                    **base_outcome(example, slot, label, "eligible"),
                    "eligible_expected_piece_ids": sorted(eligible_ids),
                    "expected_appearance_key": expected_key,
                    "rank": rank,
                    "appearance_rank": appearance_rank,
                    "exact_hit_at_1": rank == 1,
                    "exact_hit_at_k": rank is not None and rank <= top_k,
                    "appearance_hit_at_1": appearance_rank == 1,
                    "appearance_hit_at_k": appearance_rank is not None and appearance_rank <= top_k,
                    "hit_at_1": appearance_rank == 1,
                    "hit_at_k": appearance_rank is not None and appearance_rank <= top_k,
                    "results": results,
                    "used_crops": [
                        {
                            "name": record["name"],
                            "visibility": record["visibility"],
                            "role": record["role"],
                            "tags": list(record["tags"]),
                            "weight": crop_record_weight(record, role_weights, tag_penalties),
                        }
                        for record in slot_records
                    ],
                }
            )
    return outcomes


def base_outcome(example: dict[str, Any], slot: str, label: dict[str, Any] | None, status: str) -> dict[str, Any]:
    return {
        "example_id": example["example_id"],
        "glamour_id": example["glamour_id"],
        "image_position": example.get("image_position"),
        "source_image": example.get("local_path"),
        "slot": slot,
        "status": status,
        "expected_piece_ids": [int(value) for value in label.get("piece_ids", [])] if label else [],
        "expected_name": label.get("item_name") if label else None,
        "expected_appearance_key": appearance_name_key(label.get("item_name")) if label else None,
        "piece_match_type": label.get("piece_match_type") if label else None,
        "label_type": label.get("label_type") if label else None,
        "rank": None,
        "appearance_rank": None,
        "exact_hit_at_1": False,
        "exact_hit_at_k": False,
        "appearance_hit_at_1": False,
        "appearance_hit_at_k": False,
        "hit_at_1": False,
        "hit_at_k": False,
        "results": [],
    }


def crop_record_weight(record: dict[str, Any], role_weights: dict[str, float], tag_penalties: dict[str, float]) -> float:
    weight = VISIBLE_WEIGHTS.get(record["visibility"], 0.0)
    weight *= role_weights.get(record.get("role", "primary"), 0.0)
    weight *= max(0.35, min(1.0, float(record.get("confidence", 0.0))))
    for tag in record.get("tags", ()):
        weight *= tag_penalties.get(tag, 1.0)
    return weight


def available_pieces_by_slot(metadata: list[dict[str, Any]]) -> dict[str, set[int]]:
    by_slot: dict[str, set[int]] = defaultdict(set)
    for row in metadata:
        slot = row.get("slot")
        for piece in row.get("pieces", []):
            by_slot[slot].add(int(piece["piece_id"]))
    return dict(by_slot)


def available_appearance_keys_by_slot(metadata: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_slot: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        slot = row.get("slot")
        for piece in row.get("pieces", []):
            key = appearance_name_key(piece.get("name_en"))
            if key:
                by_slot[slot].add(key)
    return dict(by_slot)


def summarize(outcomes: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    eligible = [row for row in outcomes if row["status"] == "eligible"]
    by_status = Counter(row["status"] for row in outcomes)

    def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "eligible_case_count": 0,
                "exact_hits_at_1": 0,
                "exact_hits_at_k": 0,
                "appearance_hits_at_1": 0,
                "appearance_hits_at_k": 0,
                "exact_recall_at_1": 0.0,
                "exact_recall_at_k": 0.0,
                "appearance_recall_at_1": 0.0,
                "appearance_recall_at_k": 0.0,
                "exact_mrr": 0.0,
                "appearance_mrr": 0.0,
            }
        return {
            "eligible_case_count": len(rows),
            "exact_hits_at_1": sum(bool(row["exact_hit_at_1"]) for row in rows),
            "exact_hits_at_k": sum(bool(row["exact_hit_at_k"]) for row in rows),
            "appearance_hits_at_1": sum(bool(row["appearance_hit_at_1"]) for row in rows),
            "appearance_hits_at_k": sum(bool(row["appearance_hit_at_k"]) for row in rows),
            "exact_recall_at_1": sum(bool(row["exact_hit_at_1"]) for row in rows) / len(rows),
            "exact_recall_at_k": sum(bool(row["exact_hit_at_k"]) for row in rows) / len(rows),
            "appearance_recall_at_1": sum(bool(row["appearance_hit_at_1"]) for row in rows) / len(rows),
            "appearance_recall_at_k": sum(bool(row["appearance_hit_at_k"]) for row in rows) / len(rows),
            "exact_mrr": sum(1.0 / row["rank"] for row in rows if row["rank"]) / len(rows),
            "appearance_mrr": sum(1.0 / row["appearance_rank"] for row in rows if row["appearance_rank"]) / len(rows),
        }

    by_slot = {slot: summarize_rows([row for row in eligible if row["slot"] == slot]) for slot in SLOTS}
    overall = summarize_rows(eligible)
    return {
        "case_count": len(outcomes),
        "eligible_case_count": len(eligible),
        "top_k": top_k,
        "status_counts": dict(by_status),
        "exact_recall_at_1": overall["exact_recall_at_1"],
        "exact_recall_at_k": overall["exact_recall_at_k"],
        "appearance_recall_at_1": overall["appearance_recall_at_1"],
        "appearance_recall_at_k": overall["appearance_recall_at_k"],
        "exact_mrr": overall["exact_mrr"],
        "appearance_mrr": overall["appearance_mrr"],
        "exact_hits_at_1": overall["exact_hits_at_1"],
        "exact_hits_at_k": overall["exact_hits_at_k"],
        "appearance_hits_at_1": overall["appearance_hits_at_1"],
        "appearance_hits_at_k": overall["appearance_hits_at_k"],
        "by_slot": by_slot,
    }


if __name__ == "__main__":
    raise SystemExit(main())
