"""Image-only local evidence shared by inference and held-out evaluation."""

from __future__ import annotations

import numpy as np

from ffxiv_prism.matching.equipment_slots import candidate_matches_slot
from ffxiv_prism.matching.local_fusion import appearance_keys, fused_scores, score_ranks

LOCAL_POLICIES = {
    # Selected on appearance-disjoint validation folds, before blind_test.
    "HEAD": dict(weight=0.5, rrf_k=10, gate_name="eye_min", view_fusion="max"),
    "LEGS": dict(weight=0.5, rrf_k=60, gate_name="exposed", view_fusion="mean"),
}
LOCAL_NAMES = {
    "HEAD": ("head_eye_left", "head_eye_right", "head_tight", "head_face_accessory"),
    "LEGS": ("legs_waist_thighs", "legs_thighs"),
}


def local_records(records, slot):
    by_name = {r["name"]: r for r in records}
    selected = [by_name.get(n) for n in LOCAL_NAMES[slot]]
    if any(r is None for r in selected):
        return []
    if not all(
        r.get("visibility") == "visible"
        and float(r.get("confidence", 0)) >= 0.55
        and "source_edge_truncation_risk" not in r.get("tags", ())
        for r in selected
    ):
        return []
    return selected


def catalogue(metadata, slot, official, equivalence):
    keys = appearance_keys(equivalence)
    full = []
    for row in metadata:
        if row.get("candidate_crop") != "full":
            continue
        pieces = [p for p in row["pieces"] if candidate_matches_slot(p, slot, official)]
        if pieces:
            full.append(dict(row, pieces=pieces))
    ids = sorted(
        {
            keys.get(int(p["piece_id"]), int(p["piece_id"]))
            for r in full
            for p in r["pieces"]
        }
    )
    return full, ids, keys


def visual_gates(slot, probabilities):
    """Return all validation gate options, along the last axis."""
    p = np.asarray(probabilities)
    if slot == "HEAD":
        from ffxiv_prism.matching.head_detail import HEAD_TYPE_CLASSES

        eye = p[..., HEAD_TYPE_CLASSES.index("eyewear")]
        return np.stack(
            (np.sqrt(eye.prod(-1)), eye.min(-1), np.ones(eye.shape[:-1])), -1
        )
    return np.stack(
        (
            np.sqrt(p[..., 0].prod(-1)),
            np.sqrt((1 - p[..., -1]).prod(-1)),
            np.ones(p.shape[:-2]),
        ),
        -1,
    )


def grouped_eye_ranks(raw_scores, groups):
    """Max over candidate eyes/renders, then rank at appearance grain."""
    raw = np.asarray(raw_scores).max(-1)
    scores = np.stack(
        [
            raw[..., ix].max(-1) if ix else np.full(raw.shape[:-1], -np.inf)
            for ix in groups
        ],
        -1,
    )
    return score_ranks(scores)


def search_ranks(rows, ids, keys):
    columns = {pid: i for i, pid in enumerate(ids)}
    ranks = np.full(len(ids), 1_000_000, dtype=np.float32)
    seen = set()
    for row in rows:
        pid = int(row["piece_id"])
        key = keys.get(pid, pid)
        if key in seen:
            continue
        seen.add(key)
        if key in columns:
            ranks[columns[key]] = len(seen)
    return ranks


def rerank_candidates(candidates, full, ids, keys, local, gate, policy):
    """Score the union of global/local appearances; never pin an output position."""
    base = search_ranks(candidates, ids, keys)
    scores = fused_scores(
        base, local, gate, **{k: policy[k] for k in ("weight", "rrf_k", "view_fusion")}
    )
    representatives = {}
    for row in candidates:
        pid = int(row["piece_id"])
        representatives.setdefault(keys.get(pid, pid), row)
    for row in full:
        for piece in row["pieces"]:
            pid = int(piece["piece_id"])
            representatives.setdefault(
                keys.get(pid, pid),
                dict(
                    piece,
                    local_preview=row["local_path"],
                    score=0.0,
                ),
            )
    result = []
    for i in np.argsort(-scores, kind="stable"):
        if not np.isfinite(scores[i]):
            continue
        row = dict(representatives[ids[i]])
        row["global_retrieval_score"] = row.get("score")
        row["score"] = float(scores[i])
        row["rank"] = len(result) + 1
        row["local_fusion"] = dict(
            previous_rank=int(base[i]) if base[i] < 1_000_000 else None,
            local_ranks=local[..., i].astype(int).tolist(),
        )
        result.append(row)
    return result
