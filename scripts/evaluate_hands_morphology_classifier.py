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
from ffxiv_prism.matching.hands_morphology import (
    HANDS_MORPHOLOGY_CLASSES,
    hands_morphology_class,
)
from ffxiv_prism.matching.head_morphology import (
    classification_summary,
    fit_balanced_ridge_classifier,
    morphology_dev_group,
    predict_ridge_probabilities,
    select_confidence_threshold,
)
from ffxiv_prism.paths import CACHE_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.evaluation_split import load_split_assignments
from scripts.evaluate_glamour_matching import collect_crop_records
from scripts.train_glamour_query_encoder import (
    encode_records,
    load_query_head_from_checkpoint,
)


CROP_NAMES = (
    "hands_left",
    "hands_right",
    "hands_forearm_left",
    "hands_forearm_right",
    "hands_arm_cover_context",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a train-only HANDS morphology classifier."
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
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
        and (row.get("core_slot_labels") or {}).get("HANDS", {}).get("item_name")
    ]
    all_records = collect_crop_records(examples, args.regions_dir)
    records = {
        example_id: {
            str(row["name"]): row
            for row in rows
            if str(row.get("name")) in CROP_NAMES
        }
        for example_id, rows in all_records.items()
    }
    examples = [
        row
        for row in examples
        if set(records.get(str(row["example_id"]), {})) == set(CROP_NAMES)
    ]
    ordered = sorted(examples, key=lambda row: str(row["example_id"]))
    record_lists = {
        example_id: [by_name[name] for name in CROP_NAMES]
        for example_id, by_name in records.items()
        if set(by_name) == set(CROP_NAMES)
    }

    from transformers import AutoModel

    device = resolve_torch_device(args.device)
    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    query_head = load_query_head_from_checkpoint(args.query_checkpoint, model, device)
    embeddings = encode_records(
        model, query_head, record_lists, args.batch_size, device
    )
    features = np.stack(
        [
            np.concatenate(
                [
                    embeddings[str(records[str(row["example_id"])][name]["path"])]
                    for name in CROP_NAMES
                ]
            )
            for row in ordered
        ]
    ).astype(np.float32)
    class_to_index = {
        name: index for index, name in enumerate(HANDS_MORPHOLOGY_CLASSES)
    }
    labels = np.asarray(
        [
            class_to_index[
                hands_morphology_class(row["core_slot_labels"]["HANDS"]["item_name"])
            ]
            for row in ordered
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
                ordered[index]["core_slot_labels"]["HANDS"]["item_name"]
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
            features[fit_indices],
            labels[fit_indices],
            ridge=ridge,
            class_count=len(HANDS_MORPHOLOGY_CLASSES),
        )
        probabilities = predict_ridge_probabilities(features[dev_indices], weights)
        trials.append(
            {
                "ridge": ridge,
                "summary": classification_summary(
                    labels[dev_indices], probabilities, HANDS_MORPHOLOGY_CLASSES
                ),
            }
        )
    best = max(
        trials,
        key=lambda row: (
            row["summary"]["macro_recall"],
            row["summary"]["accuracy"],
            -row["ridge"],
        ),
    )
    weights = fit_balanced_ridge_classifier(
        features[fit_indices],
        labels[fit_indices],
        ridge=float(best["ridge"]),
        class_count=len(HANDS_MORPHOLOGY_CLASSES),
    )
    dev_probabilities = predict_ridge_probabilities(features[dev_indices], weights)
    confidence_selection = select_confidence_threshold(
        labels[dev_indices],
        dev_probabilities,
        target_precision=args.target_route_precision,
        minimum_count=max(10, int(round(len(dev_indices) * 0.1))),
    )
    test_probabilities = predict_ridge_probabilities(features[test_indices], weights)
    test_summary = classification_summary(
        labels[test_indices], test_probabilities, HANDS_MORPHOLOGY_CLASSES
    )
    payload = {
        "slot": "HANDS",
        "fold": args.fold,
        "fold_manifest": portable_path(args.fold_manifest),
        "query_checkpoint": portable_path(args.query_checkpoint),
        "classes": list(HANDS_MORPHOLOGY_CLASSES),
        "crop_names": list(CROP_NAMES),
        "feature_dimension": int(features.shape[1]),
        "train_count": len(train_indices),
        "internal_fit_count": len(fit_indices),
        "internal_dev_count": len(dev_indices),
        "test_count": len(test_indices),
        "train_distribution": Counter(labels[train_indices].tolist()),
        "test_distribution": Counter(labels[test_indices].tolist()),
        "ridge_trials": trials,
        "selected_ridge": best["ridge"],
        "confidence_selection": confidence_selection,
        "selected_minimum_confidence": confidence_selection["threshold"],
        "test_summary": test_summary,
        "test_predictions": [
            {
                "example_id": str(ordered[index]["example_id"]),
                "item_name": ordered[index]["core_slot_labels"]["HANDS"]["item_name"],
                "expected": HANDS_MORPHOLOGY_CLASSES[int(labels[index])],
                "probabilities": dict(
                    zip(
                        HANDS_MORPHOLOGY_CLASSES,
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
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_ridge": best["ridge"],
                "confidence_selection": confidence_selection,
                "test": test_summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
