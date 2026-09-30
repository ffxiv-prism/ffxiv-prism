"""Image-conditioned local retrieval experts for four kinds of eyewear.

No catalogue identity, name, answer, or output position enters these scores.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ffxiv_prism.matching.aligned_eyes import asset_signature
from ffxiv_prism.matching.local_fusion import score_ranks
from ffxiv_prism.paths import PROJECT_ROOT

EYE_COVERAGE_PROMPTS = (
    "a person wearing an eyepatch or a cloth bandana covering one eye",
    "a person wearing a blindfold or visor covering both eyes",
    "a person wearing glasses or spectacles with transparent lenses",
    "a person wearing goggles with thick round lenses",
    "a person wearing a helmet or a mask covering the face",
    "a person with uncovered eyes and no eyewear",
)
EYE_EXPERT_POLICY = dict(candidate_threshold=.9, weight=1., rrf_k=10)


def grouped_coverage_profiles(profiles, groups):
    """Average render evidence at appearance grain, ignoring repeated links."""
    profiles = np.asarray(profiles, dtype=np.float32)
    if profiles.ndim != 2 or profiles.shape[-1] != len(EYE_COVERAGE_PROMPTS):
        raise ValueError("Expected six eye-coverage classes per render")
    if not np.isfinite(profiles).all() or np.any((profiles < 0) | (profiles > 1)):
        raise ValueError("Invalid coverage profiles")
    return np.stack([
        profiles[sorted(set(ix))].mean(0) if ix
        else np.zeros(profiles.shape[-1], dtype=np.float32)
        for ix in groups
    ])


def eye_expert_scores(local_ranks, query_profiles, candidate_profiles, gate, *,
                      candidate_threshold=.9, weight=1., rrf_k=10):
    """Mix class-conditional ranks using corroborated, continuous query evidence.

    Class membership is image-derived and shared by every candidate. Helmet
    and bare-eye classes are negative evidence, never recall experts. The two
    query views must support the same class; their minimum softmax score gates
    that expert. These relative prompt scores are not calibrated probabilities.
    """
    local = np.asarray(local_ranks, dtype=np.float32)
    query = np.asarray(query_profiles, dtype=np.float32)
    candidates = np.asarray(candidate_profiles, dtype=np.float32)
    gate = np.asarray(gate, dtype=np.float32)
    batch_shape = local.shape[:-3]
    if (local.ndim < 3 or local.shape[-3:-1] != (2, 2)
            or query.shape != batch_shape + (2, 6)
            or candidates.shape != (local.shape[-1], 6)
            or gate.shape != batch_shape):
        raise ValueError("Eye expert query/model/appearance axes must align")
    if (not 0 < candidate_threshold <= 1 or weight < 0 or rrf_k < 1
            or not np.isfinite([weight, rrf_k, candidate_threshold]).all()):
        raise ValueError("Invalid eye expert policy")
    for value in (query, candidates, gate):
        if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("Expert profiles and gates must be finite and in [0,1]")
    if not np.isfinite(local).all() or np.any(local < 1):
        raise ValueError("Local ranks must be positive and finite")
    evidence = np.zeros(batch_shape + (local.shape[-1],), dtype=np.float32)
    confidence = query.min(axis=-2)
    for cls in range(4):
        allowed = candidates[:, cls] >= candidate_threshold
        ranks = score_ranks(np.where(allowed & (local < 1_000_000), -local, -np.inf))
        per_view = np.where(ranks < 1_000_000, 1 / (rrf_k + ranks), 0).mean(axis=-3)
        evidence += confidence[..., cls, None] * per_view.max(axis=-2)
    return weight * gate[..., None] * evidence


def classify_eye_coverage(paths, *, classifier, device, batch_size, cache_dir=None):
    """Cache visual profiles by source content, model assets and prompt contract."""
    model_id = "google/siglip2-base-patch16-224"
    cache_dir = cache_dir or PROJECT_ROOT / "var/cache/eye-coverage-v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    contract = json.dumps((1, model_id, EYE_COVERAGE_PROMPTS, "full-rgb",
                           asset_signature(model_id))).encode()
    keys = [hashlib.sha256(contract + Path(p).read_bytes()).hexdigest() for p in paths]
    found, missing = {}, {}
    for key, path in zip(keys, paths, strict=True):
        cache = cache_dir / (key + ".npy")
        if cache.is_file():
            try:
                value = np.load(cache, allow_pickle=False)
                if (value.shape == (6,) and np.isfinite(value).all()
                        and np.all(value >= 0) and np.isclose(value.sum(), 1, atol=1e-4)):
                    found[key] = value
                    continue
            except (OSError, ValueError, EOFError):
                pass
        missing[key] = path
    if missing:
        profiles = classifier(model_id, EYE_COVERAGE_PROMPTS, list(missing.values()),
                              device=device, batch_size=batch_size)
        for key, value in zip(missing, profiles, strict=True):
            if (np.shape(value) != (6,) or not np.isfinite(value).all()
                    or np.any(value < 0) or not np.isclose(np.sum(value), 1, atol=1e-4)):
                raise ValueError("Invalid classifier coverage profile")
            found[key] = value
            with NamedTemporaryFile(dir=cache_dir, suffix=".npy", delete=False) as stream:
                temporary = Path(stream.name)
                np.save(stream, value, allow_pickle=False)
            try:
                temporary.replace(cache_dir / (key + ".npy"))
            finally:
                temporary.unlink(missing_ok=True)
    return np.stack([found[key] for key in keys]) if keys else np.empty((0, 6))
