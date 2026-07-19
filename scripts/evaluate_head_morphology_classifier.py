from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

np.linalg.inv(np.eye(2, dtype=np.float32))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.coverage_profiles import (
    encode_head_parsing_spatial_profiles,
)
from ffxiv_prism.matching.head_morphology import (
    HEAD_MORPHOLOGY_CLASSES,
    classification_summary,
    fit_balanced_ridge_classifier,
    head_morphology_class,
    morphology_dev_group,
    predict_ridge_probabilities,
    select_confidence_threshold,
)
from ffxiv_prism.paths import portable_path
from ffxiv_prism.vision.evaluation_split import load_split_assignments
from ffxiv_prism.vision.human_parsing import HumanParser
from scripts.evaluate_glamour_matching import collect_crop_records
from scripts.evaluate_head_coverage_reranker import select_coverage_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train-only HEAD morphology router evaluation."
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--feature-kind", choices=("parsing", "query-encoder"), default="parsing"
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--query-checkpoint", type=Path)
    parser.add_argument("--crop-name", default="head_tight")
    parser.add_argument(
        "--taxonomy",
        choices=("three-class", "enclosing-binary", "open-binary"),
        default="three-class",
    )
    parser.add_argument(
        "--ridge-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0]
    )
    parser.add_argument("--target-route-precision", type=float, default=0.95)
    args = parser.parse_args()

    assignments = load_split_assignments(args.fold_manifest)
    quality = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
    }
    examples = [
        row
        for row in read_jsonl(args.examples)
        if quality.get(str(row["example_id"]), False)
        and assignments.get(str(row["glamour_id"])) in {"train", "validation"}
        and (row.get("core_slot_labels") or {}).get("HEAD", {}).get("item_name")
    ]
    all_records = collect_crop_records(examples, args.regions_dir)
    if args.feature_kind == "parsing":
        records = select_coverage_records(all_records)
    else:
        if args.query_checkpoint is None:
            parser.error("--query-checkpoint is required for query-encoder features")
        records = {
            example_id: row
            for example_id, rows in all_records.items()
            for row in rows
            if str(row.get("name")) == args.crop_name
        }
    examples = [row for row in examples if str(row["example_id"]) in records]
    ordered = sorted(examples, key=lambda row: str(row["example_id"]))
    if args.feature_kind == "parsing":
        profiles = encode_head_parsing_spatial_profiles(
            HumanParser(device=args.device),
            [records[str(row["example_id"])] for row in ordered],
            batch_size=args.batch_size,
            grid_size=16,
        ).reshape(len(ordered), -1).astype(np.float32)
        feature_description = "human-parsing-head-layout-16x16-input256-v1"
    else:
        from transformers import AutoModel

        from ffxiv_prism.paths import CACHE_DIR
        from ffxiv_prism.torch_device import resolve_torch_device
        from scripts.train_glamour_query_encoder import (
            encode_records,
            load_query_head_from_checkpoint,
        )

        device = resolve_torch_device(args.device)
        model = AutoModel.from_pretrained(
            args.model_id,
            cache_dir=CACHE_DIR / "huggingface" / "hub",
            local_files_only=True,
        ).to(device)
        query_head = load_query_head_from_checkpoint(
            args.query_checkpoint, model, device
        )
        record_lists = {
            example_id: [record] for example_id, record in records.items()
        }
        embeddings = encode_records(
            model, query_head, record_lists, args.batch_size, device
        )
        profiles = np.stack(
            [embeddings[str(records[str(row["example_id"])]["path"])] for row in ordered]
        ).astype(np.float32)
        feature_description = (
            f"query-encoder:{args.model_id}:{portable_path(args.query_checkpoint)}:"
            f"{args.crop_name}"
        )
    if args.taxonomy == "three-class":
        class_names = HEAD_MORPHOLOGY_CLASSES
    elif args.taxonomy == "enclosing-binary":
        class_names = ("non_enclosing", "enclosing")
    else:
        class_names = ("covered", "open")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    morphology_labels = [
        head_morphology_class(row["core_slot_labels"]["HEAD"]["item_name"])
        for row in ordered
    ]
    labels = np.asarray(
        [
            class_to_index[
                value
                if args.taxonomy == "three-class"
                else (
                    ("enclosing" if value == "enclosing" else "non_enclosing")
                    if args.taxonomy == "enclosing-binary"
                    else ("open" if value == "open" else "covered")
                )
            ]
            for value in morphology_labels
        ],
        dtype=np.int64,
    )
    train_indices = np.asarray(
        [
            index
            for index, row in enumerate(ordered)
            if assignments[str(row["glamour_id"])] == "train"
        ],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [
            index
            for index, row in enumerate(ordered)
            if assignments[str(row["glamour_id"])] == "validation"
        ],
        dtype=np.int64,
    )
    dev_mask = np.asarray(
        [
            morphology_dev_group(
                ordered[index]["core_slot_labels"]["HEAD"]["item_name"]
            )
            for index in train_indices
        ],
        dtype=bool,
    )
    fit_indices = train_indices[~dev_mask]
    dev_indices = train_indices[dev_mask]
    trials = []
    for ridge in args.ridge_grid:
        weights = fit_balanced_ridge_classifier(
            profiles[fit_indices],
            labels[fit_indices],
            ridge=ridge,
            class_count=len(class_names),
        )
        summary = classification_summary(
            labels[dev_indices],
            predict_ridge_probabilities(profiles[dev_indices], weights),
            class_names,
        )
        trials.append({"ridge": ridge, "summary": summary})
    best = max(
        trials,
        key=lambda row: (
            row["summary"]["macro_recall"],
            row["summary"]["accuracy"],
            -row["ridge"],
        ),
    )
    weights = fit_balanced_ridge_classifier(
        profiles[fit_indices],
        labels[fit_indices],
        ridge=float(best["ridge"]), class_count=len(class_names)
    )
    dev_probabilities = predict_ridge_probabilities(profiles[dev_indices], weights)
    confidence_selection = select_confidence_threshold(
        labels[dev_indices],
        dev_probabilities,
        target_precision=args.target_route_precision,
        minimum_count=max(10, int(round(len(dev_indices) * 0.1))),
    )
    test_probabilities = predict_ridge_probabilities(profiles[test_indices], weights)
    test_summary = classification_summary(
        labels[test_indices], test_probabilities, class_names
    )
    payload = {
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "taxonomy": args.taxonomy,
        "classes": list(class_names),
        "feature": feature_description,
        "train_count": len(train_indices),
        "internal_fit_count": len(fit_indices),
        "internal_dev_count": len(dev_indices),
        "test_count": len(test_indices),
        "train_distribution": Counter(labels[train_indices].tolist()),
        "test_distribution": Counter(labels[test_indices].tolist()),
        "ridge_trials": trials,
        "selected_ridge": best["ridge"],
        "target_route_precision": args.target_route_precision,
        "confidence_selection": confidence_selection,
        "selected_minimum_confidence": confidence_selection["threshold"],
        "test_summary": test_summary,
        "test_predictions": [
            {
                "example_id": str(ordered[index]["example_id"]),
                "item_name": ordered[index]["core_slot_labels"]["HEAD"]["item_name"],
                "expected": class_names[int(labels[index])],
                "probabilities": dict(
                    zip(
                        class_names,
                        (float(value) for value in probability),
                        strict=True,
                    )
                ),
            }
            for index, probability in zip(test_indices, test_probabilities, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=dict),
        encoding="utf-8",
    )
    if args.checkpoint_output is not None:
        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.checkpoint_output,
            version=np.asarray(1),
            weights=weights,
            classes=np.asarray(class_names),
            taxonomy=np.asarray(args.taxonomy),
            minimum_confidence=np.asarray(
                float(confidence_selection["threshold"]), dtype=np.float32
            ),
            target_route_precision=np.asarray(
                args.target_route_precision, dtype=np.float32
            ),
            selected_ridge=np.asarray(float(best["ridge"]), dtype=np.float32),
            feature_kind=np.asarray(args.feature_kind),
            model_id=np.asarray(args.model_id),
            query_checkpoint=np.asarray(portable_path(args.query_checkpoint)),
            crop_name=np.asarray(args.crop_name),
        )
    print(json.dumps({"output": str(args.output), "selected_ridge": best["ridge"], "test": test_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
