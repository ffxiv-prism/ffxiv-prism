"""Conservative local-detail recall, independent of upper-head adapters.

No item names, evaluation labels or identities select the local views.
Two Top-1 appearance votes and support in the third view's Top-10 are
required. The existing first five appearances are never displaced.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

DETAIL_NAMES = ("head_lower_face_detail", "head_ear_left_detail", "head_ear_right_detail")
MIN_CONFIDENCE = 0.55
PRESERVE_TOP = 5
HEAD_TYPE_CLASSES = ("hat", "helmet", "hair_ornament", "eyewear", "mask", "headset", "ear_ornament", "bare")
HEAD_TYPE_PROMPTS = (
    "a person wearing a hat or cap on their head",
    "a person wearing a helmet covering their head",
    "a person wearing a crown, tiara or hair ornament",
    "a person wearing glasses, goggles or an eyepatch",
    "a person wearing a face mask covering their mouth",
    "a person wearing a headset with a microphone beside their mouth",
    "a person wearing earrings or ear ornaments",
    "a person with a bare face and no head accessories",
)
LOCAL_TYPES = {"eyewear", "mask", "headset", "ear_ornament"}


def detail_presence_decision(profiles: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Require clear lower-face evidence and independent ear-side agreement.

    Confidence thresholds reuse the project's strong-style (0.75) and
    majority (0.5) decisions; probabilities are relative prompt scores.
    """
    decision: dict[str, Any] = {"active": False, "reason": "presence_views_unavailable"}
    if any(name not in profiles for name in DETAIL_NAMES):
        return decision
    predictions = {}
    for name in DETAIL_NAMES:
        values = list(map(float, profiles[name]))
        if len(values) != len(HEAD_TYPE_CLASSES):
            raise ValueError("HEAD type profile must match prompt classes")
        index = max(range(len(values)), key=values.__getitem__)
        predictions[name] = {"class": HEAD_TYPE_CLASSES[index], "confidence": values[index]}
    primary = predictions[DETAIL_NAMES[0]]
    supported = any(
        predictions[name]["class"] == primary["class"]
        and predictions[name]["confidence"] >= 0.5
        for name in DETAIL_NAMES[1:]
    )
    active = primary["class"] in LOCAL_TYPES and primary["confidence"] >= 0.75 and supported
    decision.update(
        active=active, reason="visible_local_accessory" if active else "local_accessory_uncertain",
        predictions=predictions, primary_min_confidence=0.75, ear_min_confidence=0.5,
    )
    return decision


def detail_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    # Crop tags describe geometry; visibility is the authoritative quality gate.
    by_name = {
        str(r.get("name")): r for r in records
        if r.get("name") in DETAIL_NAMES
        and r.get("visibility") == "visible"
        and float(r.get("confidence", 0)) >= MIN_CONFIDENCE
    }
    if any(name not in by_name for name in DETAIL_NAMES):
        return []
    return [dict(by_name[name], role="auxiliary") for name in DETAIL_NAMES]


def recall_detail_candidate(
    candidates: Sequence[dict[str, Any]],
    views: Mapping[str, Sequence[dict[str, Any]]],
    equivalence: Mapping[int, Sequence[int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Insert one corroborated appearance after the protected prefix.

    Caller supplies HEAD-filtered, same-model-deduplicated rankings per view.
    A known appearance already above the insertion point stays in place.
    """
    original = list(candidates)
    diagnostic: dict[str, Any] = {
        "active": False, "ranking_changed": False,
        "reason": "detail_views_unavailable", "preserve_top": PRESERVE_TOP,
        "required_top1_votes": 2, "support_top_k": 10,
    }
    if any(not views.get(name) for name in DETAIL_NAMES):
        return original, diagnostic

    def key(row):
        pid = int(row["piece_id"])
        return min((pid, *equivalence.get(pid, ())))

    tops = {name: key(views[name][0]) for name in DETAIL_NAMES}
    winner, votes = Counter(tops.values()).most_common(1)[0]
    ranks = {
        name: next((i for i, r in enumerate(views[name][:10], 1) if key(r) == winner), None)
        for name in DETAIL_NAMES
    }
    diagnostic.update(top1_appearances=tops, support_ranks=ranks, votes=votes)
    if votes < 2 or any(rank is None for rank in ranks.values()):
        diagnostic["reason"] = "local_views_disagree"
        return original, diagnostic
    diagnostic.update(active=True, appearance_id=winner)
    old_index = next((i for i, r in enumerate(original) if key(r) == winner), None)
    diagnostic["previous_rank"] = old_index + 1 if old_index is not None else None
    if old_index is not None and old_index <= PRESERVE_TOP:
        diagnostic["reason"] = "already_high_ranked"
        return original, diagnostic
    chosen = (
        dict(original[old_index]) if old_index is not None
        else dict(next(views[n][0] for n in DETAIL_NAMES if tops[n] == winner))
    )
    chosen["head_detail_recall"] = True
    chosen["head_detail_support_ranks"] = ranks
    result = [dict(r) for r in original if key(r) != winner]
    result.insert(min(PRESERVE_TOP, len(result)), chosen)
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
    diagnostic.update(ranking_changed=True, reason="local_consensus_recall", rank=chosen["rank"])
    return result, diagnostic
