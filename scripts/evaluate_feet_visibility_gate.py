from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import read_jsonl, write_json  # noqa: E402
from ffxiv_prism.paths import portable_path, resolve_project_path  # noqa: E402
from ffxiv_prism.vision.human_parsing import HumanParser  # noqa: E402


SHOE_LABELS = frozenset((9, 10))
LOWER_GARMENT_LABELS = frozenset((5, 7))


def semantic_feet_features(segmentation: np.ndarray) -> dict[str, float]:
    labels = np.asarray(segmentation, dtype=np.uint8)
    if labels.ndim != 2:
        raise ValueError("segmentation must be a 2D label map")
    person = labels != 0
    coordinates = np.argwhere(person)
    if not len(coordinates):
        return {
            "person_fraction": 0.0,
            "shoe_share_person": 0.0,
            "lower_garment_bottom_ratio": 0.0,
            "lower_garment_bottom_band_share": 0.0,
            "person_bottom_edge_gap_ratio": 1.0,
        }

    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    person_height = max(1, int(y_max - y_min + 1))
    person_pixels = int(person.sum())
    shoes = np.isin(labels, tuple(SHOE_LABELS))
    lower_garment = np.isin(labels, tuple(LOWER_GARMENT_LABELS))
    lower_coordinates = np.argwhere(lower_garment)
    lower_bottom_ratio = (
        float((lower_coordinates[:, 0].max() - y_min + 1) / person_height)
        if len(lower_coordinates)
        else 0.0
    )
    band_top = int(round(y_min + person_height * 0.8))
    band_top = min(max(int(y_min), band_top), int(y_max))
    bottom_person = person[band_top : y_max + 1, x_min : x_max + 1]
    bottom_lower = lower_garment[
        band_top : y_max + 1,
        x_min : x_max + 1,
    ]
    bottom_person_pixels = int(bottom_person.sum())
    return {
        "person_fraction": float(person.mean()),
        "shoe_share_person": float(shoes.sum() / max(person_pixels, 1)),
        "lower_garment_bottom_ratio": lower_bottom_ratio,
        "lower_garment_bottom_band_share": float(
            bottom_lower.sum() / max(bottom_person_pixels, 1)
        ),
        "person_bottom_edge_gap_ratio": float(
            (labels.shape[0] - 1 - y_max) / max(labels.shape[0], 1)
        ),
    }


def visibility_rule(
    features: Mapping[str, float],
    *,
    min_lower_bottom_ratio: float,
    min_bottom_band_share: float,
    max_shoe_share: float,
    max_edge_gap_ratio: float,
) -> bool:
    weak_shoe_evidence = features["shoe_share_person"] <= max_shoe_share
    floor_length_overlap = (
        features["lower_garment_bottom_ratio"] >= min_lower_bottom_ratio
        and features["lower_garment_bottom_band_share"]
        >= min_bottom_band_share
    )
    source_edge_risk = (
        features["person_bottom_edge_gap_ratio"] <= max_edge_gap_ratio
    )
    return bool(
        weak_shoe_evidence and (floor_length_overlap or source_edge_risk)
    )


def confusion_metrics(
    labels: Sequence[bool],
    predictions: Sequence[bool],
) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    true_positive = sum(label and prediction for label, prediction in zip(
        labels,
        predictions,
        strict=True,
    ))
    false_positive = sum((not label) and prediction for label, prediction in zip(
        labels,
        predictions,
        strict=True,
    ))
    false_negative = sum(label and (not prediction) for label, prediction in zip(
        labels,
        predictions,
        strict=True,
    ))
    true_negative = len(labels) - true_positive - false_positive - false_negative
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (true_positive + true_negative) / max(len(labels), 1)
    return {
        "count": len(labels),
        "positive_count": sum(labels),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def stratified_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for positive in (False, True):
        group = [
            row
            for row in rows
            if bool(row["unusable"]) == positive
        ]
        ordered = sorted(
            group,
            key=lambda row: hashlib.sha256(
                str(row["example_id"]).encode("utf-8")
            ).digest(),
        )
        for index, row in enumerate(ordered):
            assignments[str(row["example_id"])] = (
                "development" if index % 2 == 0 else "holdout"
            )
    return assignments


def tune_rule(
    rows: Sequence[Mapping[str, Any]],
    *,
    objective: str = "balanced",
) -> dict[str, float]:
    if objective not in {"balanced", "triage"}:
        raise ValueError(f"unsupported tuning objective: {objective}")
    candidates = product(
        (0.78, 0.82, 0.86, 0.88, 0.90, 0.92, 0.94),
        (0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10),
        (0.0, 0.01, 0.02),
    )
    best: tuple[tuple[float, ...], dict[str, float]] | None = None
    labels = [bool(row["unusable"]) for row in rows]
    for lower_ratio, band_share, shoe_share, edge_gap in candidates:
        thresholds = {
            "min_lower_bottom_ratio": lower_ratio,
            "min_bottom_band_share": band_share,
            "max_shoe_share": shoe_share,
            "max_edge_gap_ratio": edge_gap,
        }
        predictions = [
            visibility_rule(row["features"], **thresholds)
            for row in rows
        ]
        metrics = confusion_metrics(labels, predictions)
        if objective == "balanced":
            score = (
                metrics["f1"],
                metrics["recall"],
                metrics["precision"],
                -shoe_share,
                band_share,
                lower_ratio,
                -edge_gap,
            )
        else:
            score = (
                metrics["recall"],
                metrics["precision"],
                metrics["f1"],
                -shoe_share,
                band_share,
                lower_ratio,
                -edge_gap,
            )
        if best is None or score > best[0]:
            best = (score, thresholds)
    if best is None:
        raise ValueError("cannot tune a rule without candidate thresholds")
    return best[1]


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    labels = [bool(row["unusable"]) for row in rows]
    predictions = [
        visibility_rule(row["features"], **thresholds)
        for row in rows
    ]
    return confusion_metrics(labels, predictions)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether human-parsing shoe and floor-length-garment cues "
            "can gate semantically hidden FEET labels."
        )
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path(
            "data/processed/glamour_examples_huiji_models_closed.jsonl"
        ),
    )
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    examples = {
        str(row["example_id"]): row for row in read_jsonl(args.examples)
    }
    reviews = [
        row
        for row in read_jsonl(args.reviews)
        if str(row.get("slot")) == "FEET"
    ]
    missing = [
        str(row.get("example_id"))
        for row in reviews
        if str(row.get("example_id")) not in examples
    ]
    if missing:
        raise ValueError(f"review examples missing from manifest: {missing[:5]}")

    parser_model = HumanParser(device=args.device)
    rows = []
    for start in range(0, len(reviews), args.batch_size):
        batch_reviews = reviews[start : start + args.batch_size]
        images = []
        for review in batch_reviews:
            example = examples[str(review["example_id"])]
            image_path = resolve_project_path(example.get("local_path"))
            if image_path is None or not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as source:
                images.append(source.convert("RGB"))
        segmentations = parser_model.parse_native(images)
        for review, segmentation in zip(
            batch_reviews,
            segmentations,
            strict=True,
        ):
            rows.append(
                {
                    "example_id": str(review["example_id"]),
                    "visibility": str(review.get("visibility")),
                    "unusable": str(review.get("usability")) == "unusable",
                    "issue": str(review.get("issue") or "none"),
                    "features": semantic_feet_features(segmentation),
                }
            )
        for image in images:
            image.close()

    assignments = stratified_split(rows)
    for row in rows:
        row["split"] = assignments[str(row["example_id"])]
    development = [row for row in rows if row["split"] == "development"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    thresholds = tune_rule(development, objective="balanced")
    triage_thresholds = tune_rule(development, objective="triage")
    for row in rows:
        row["predicted_unusable"] = visibility_rule(
            row["features"],
            **thresholds,
        )
        row["triage_for_review"] = visibility_rule(
            row["features"],
            **triage_thresholds,
        )
    payload = {
        "version": 1,
        "diagnostic": "feet_semantic_visibility_gate",
        "examples": portable_path(args.examples),
        "reviews": portable_path(args.reviews),
        "parser_model": parser_model.model_id,
        "split_policy": (
            "SHA-256 ordered, label-stratified alternating development/holdout"
        ),
        "thresholds_selected_on": "development",
        "thresholds": thresholds,
        "development": evaluate_rows(development, thresholds),
        "holdout": evaluate_rows(holdout, thresholds),
        "full_sample_diagnostic_only": evaluate_rows(rows, thresholds),
        "high_recall_triage": {
            "thresholds": triage_thresholds,
            "development": evaluate_rows(development, triage_thresholds),
            "holdout": evaluate_rows(holdout, triage_thresholds),
            "full_sample_diagnostic_only": evaluate_rows(
                rows,
                triage_thresholds,
            ),
        },
        "production_gate_ready": False,
        "interpretation_limit": (
            "The holdout contains only four or five unusable cases. Treat "
            "this as a parser-rule feasibility test; validate on a second "
            "independent FEET sample before filtering the training set."
        ),
        "rows": rows,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "thresholds": thresholds,
                "development": payload["development"],
                "holdout": payload["holdout"],
                "full_sample_diagnostic_only": payload[
                    "full_sample_diagnostic_only"
                ],
                "high_recall_triage": payload["high_recall_triage"],
                "output": portable_path(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
