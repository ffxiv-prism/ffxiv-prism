"""Continuous global/local retrieval fusion, independent of labels and IDs."""

from __future__ import annotations

import numpy as np


def appearance_keys(equivalence):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for pid, others in equivalence.items():
        for other in others:
            a, b = find(int(pid)), find(int(other))
            parent[max(a, b)] = min(a, b)
    return {pid: find(pid) for pid in parent}


def score_ranks(scores):
    """Stable ranks along the last axis, with non-finite entries unavailable."""
    # Match global cosine retrieval's precision and stable tie convention.
    values = np.round(np.asarray(scores), decimals=6)
    ranks = (
        np.argsort(np.argsort(-values, axis=-1, kind="stable"), axis=-1, kind="stable")
        + 1
    )
    return np.where(np.isfinite(values), ranks, 1_000_000).astype(np.float32)


def fused_scores(base_ranks, local_ranks, gate, *, weight, rrf_k, view_fusion="max"):
    """Blend reciprocal ranks across models, local views and global context.

    local_ranks has [..., models, views, appearances]; gate is continuous
    image evidence in [0,1]. No protected prefix, insertion rank or item rule.
    """
    base = np.asarray(base_ranks, dtype=np.float32)
    local = np.asarray(local_ranks, dtype=np.float32)
    gate = np.asarray(gate, dtype=np.float32)
    if weight < 0 or rrf_k < 1 or view_fusion not in {"max", "mean"}:
        raise ValueError("Invalid fusion parameters")
    if local.shape[:-3] != base.shape[:-1] or local.shape[-1] != base.shape[-1]:
        raise ValueError("Global and local appearance axes must align")
    if not np.isfinite(gate).all() or np.any((gate < 0) | (gate > 1)):
        raise ValueError("Local evidence gate must be finite and in [0,1]")
    global_score = np.where(base < 1_000_000, 1 / (rrf_k + base), 0)
    per_view = np.where(local < 1_000_000, 1 / (rrf_k + local), 0).mean(axis=-3)
    local_score = (
        per_view.max(axis=-2) if view_fusion == "max" else per_view.mean(axis=-2)
    )
    available = (base < 1_000_000) | (
        (local < 1_000_000).any(axis=(-3, -2)) & (weight * gate[..., None] > 0)
    )
    return np.where(
        available, global_score + weight * gate[..., None] * local_score, -np.inf
    )


def ranks_of_truth(scores, truth):
    ranks = score_ranks(scores)
    return np.where(truth, ranks, 1_000_000).min(axis=-1)
