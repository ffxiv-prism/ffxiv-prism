from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HEAD_MORPHOLOGY_CLASSES = ("enclosing", "face", "open")
_ENCLOSING = re.compile(
    r"\b(hood|cowl|helm|helmet|coif|sallet|armet|barbut|greathelm|kabuto|turban|wimple)\b",
    re.IGNORECASE,
)
_FACE = re.compile(
    r"\b(mask|visor|veil|goggles|glasses|spectacles|eyepatch|monocle|headgear|mengu|bandage)\b",
    re.IGNORECASE,
)


def head_morphology_class(item_name: str | None) -> str:
    name = str(item_name or "")
    if _ENCLOSING.search(name):
        return "enclosing"
    if _FACE.search(name):
        return "face"
    return "open"


def morphology_dev_group(item_name: str, fraction: float = 0.2) -> bool:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    digest = hashlib.sha256(f"head-morphology:{item_name}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < fraction


def fit_balanced_ridge_classifier(
    features: np.ndarray,
    labels: Sequence[int],
    *,
    ridge: float,
    class_count: int,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features and labels must align")
    if ridge <= 0.0 or class_count < 2:
        raise ValueError("ridge and class_count must be positive")
    counts = np.bincount(y, minlength=class_count).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("every class needs at least one training example")
    class_weights = len(y) / (class_count * counts)
    sample_scale = np.sqrt(class_weights[y])[:, None]
    x_augmented = np.concatenate(
        [x, np.ones((len(x), 1), dtype=np.float64)], axis=1
    )
    weighted_x = x_augmented * sample_scale
    targets = np.eye(class_count, dtype=np.float64)[y] * sample_scale
    solve_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_x = torch.from_numpy(weighted_x).to(solve_device)
    torch_targets = torch.from_numpy(targets).to(solve_device)
    if len(weighted_x) <= weighted_x.shape[1]:
        dual = torch_x @ torch_x.T
        dual.diagonal().add_(ridge)
        weights = torch_x.T @ torch.linalg.solve(dual, torch_targets)
    else:
        gram = torch_x.T @ torch_x
        gram.diagonal().add_(ridge)
        weights = torch.linalg.solve(gram, torch_x.T @ torch_targets)
    return weights.cpu().numpy().astype(np.float32)


def predict_ridge_probabilities(
    features: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] + 1 != w.shape[0]:
        raise ValueError("feature and classifier dimensions do not align")
    augmented = np.concatenate(
        [x, np.ones((len(x), 1), dtype=np.float32)], axis=1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.from_numpy(augmented).to(device) @ torch.from_numpy(w).to(device)
    return torch.softmax(logits, dim=1).cpu().numpy()


def classification_summary(
    labels: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str] = HEAD_MORPHOLOGY_CLASSES,
) -> dict[str, object]:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(probabilities).argmax(axis=1)
    class_count = len(class_names)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for expected, actual in zip(truth, predicted, strict=True):
        confusion[int(expected), int(actual)] += 1
    recalls = []
    for index in range(class_count):
        denominator = int(confusion[index].sum())
        recalls.append(
            float(confusion[index, index] / denominator) if denominator else 0.0
        )
    return {
        "count": len(truth),
        "accuracy": float((truth == predicted).mean()) if len(truth) else 0.0,
        "macro_recall": float(np.mean(recalls)),
        "class_recall": dict(zip(class_names, recalls, strict=True)),
        "confusion": confusion.tolist(),
    }


def select_confidence_threshold(
    labels: Sequence[int],
    probabilities: np.ndarray,
    *,
    target_precision: float = 0.95,
    minimum_count: int = 10,
) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(truth):
        raise ValueError("labels and probabilities must align")
    if not 0.5 <= target_precision <= 1.0 or minimum_count < 1:
        raise ValueError("invalid confidence-selection constraints")
    predicted = values.argmax(axis=1)
    confidence = values.max(axis=1)
    for threshold in sorted(set(float(value) for value in confidence)):
        selected = confidence >= threshold
        count = int(selected.sum())
        if count < minimum_count:
            continue
        precision = float((predicted[selected] == truth[selected]).mean())
        if precision >= target_precision:
            return {
                "threshold": threshold,
                "count": count,
                "coverage": float(count / len(truth)),
                "precision": precision,
            }
    return {"threshold": 1.0, "count": 0, "coverage": 0.0, "precision": 1.0}


def rerank_candidates_by_enclosing_probability(
    candidates: Sequence[Mapping[str, Any]],
    enclosing_probability: float,
    *,
    rerank_k: int = 10,
    minimum_confidence: float = 0.9,
) -> list[dict[str, Any]]:
    if rerank_k < 1:
        raise ValueError("rerank_k must be positive")
    if not 0.0 <= enclosing_probability <= 1.0:
        raise ValueError("enclosing_probability must be between zero and one")
    if not 0.5 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between 0.5 and one")
    output = [dict(row) for row in candidates]
    confidence = max(enclosing_probability, 1.0 - enclosing_probability)
    if confidence < minimum_confidence:
        return output
    expected_enclosing = enclosing_probability >= 0.5
    leading = output[: min(rerank_k, len(output))]
    for row in leading:
        row["retrieval_score"] = float(row.get("score", 0.0))
        row["predicted_enclosing_probability"] = enclosing_probability
        row["morphology_confidence"] = confidence
        row["candidate_morphology"] = head_morphology_class(row.get("name_en"))
        row["morphology_match"] = (
            (row["candidate_morphology"] == "enclosing") == expected_enclosing
        )
    leading.sort(key=lambda row: bool(row["morphology_match"]), reverse=True)
    output[: len(leading)] = leading
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def rerank_candidates_by_covered_probability(
    candidates: Sequence[Mapping[str, Any]],
    covered_probability: float,
    *,
    rerank_k: int = 10,
    minimum_confidence: float = 0.9,
) -> list[dict[str, Any]]:
    if rerank_k < 1:
        raise ValueError("rerank_k must be positive")
    if not 0.0 <= covered_probability <= 1.0:
        raise ValueError("covered_probability must be between zero and one")
    if not 0.5 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between 0.5 and one")
    output = [dict(row) for row in candidates]
    confidence = max(covered_probability, 1.0 - covered_probability)
    if confidence < minimum_confidence:
        return output
    expected_covered = covered_probability >= 0.5
    leading = output[: min(rerank_k, len(output))]
    for row in leading:
        morphology = head_morphology_class(row.get("name_en"))
        row["retrieval_score"] = float(row.get("score", 0.0))
        row["predicted_covered_probability"] = covered_probability
        row["morphology_confidence"] = confidence
        row["candidate_morphology"] = morphology
        row["morphology_match"] = ((morphology != "open") == expected_covered)
    leading.sort(key=lambda row: bool(row["morphology_match"]), reverse=True)
    output[: len(leading)] = leading
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output
