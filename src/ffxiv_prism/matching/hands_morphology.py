from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


HANDS_MORPHOLOGY_CLASSES = ("minimal", "long_armor", "glove")
_MINIMAL = re.compile(
    r"\b(nails?|claws?|wrist|ringbands?|bracelets?|chaplets?|runes?|dressing|bands?)\b",
    re.IGNORECASE,
)
_LONG_ARMOR = re.compile(
    r"\b(gauntlets?|vambraces?|armguards?|armlets?|kote|tekko|sleeves?)\b",
    re.IGNORECASE,
)


def hands_morphology_class(item_name: str | None) -> str:
    name = str(item_name or "")
    if _MINIMAL.search(name):
        return "minimal"
    if _LONG_ARMOR.search(name):
        return "long_armor"
    return "glove"


HANDS_SHAPE_CLASSES = ("mitten", "fingered_glove", "fingerless", "armor", "bare")
HANDS_SHAPE_PROMPTS = (
    "padded mittens with a single rounded pouch enclosing the fingers together "
    "and a separate thumb, with thick cuffs",
    "fitted gloves with individually separated fingers",
    "fingerless gloves or wrist cuffs with bare exposed fingers",
    "metal armored gauntlets or long arm guards",
    "bare hands without gloves",
)
HANDS_SHAPE_QUERY_CROPS = ("hands_palm_left", "hands_palm_right")
HANDS_SHAPE_MIN_QUERY_CONFIDENCE = 0.70
HANDS_SHAPE_MIN_CONFIDENCE = 0.80


def _shape_profile(values: Sequence[float]) -> tuple[float, ...]:
    profile = tuple(float(value) for value in values)
    if (len(profile) != len(HANDS_SHAPE_CLASSES)
            or any(not math.isfinite(v) or v < 0 for v in profile)
            or sum(profile) <= 0):
        raise ValueError("Invalid hand-shape probability profile")
    total = sum(profile)
    return tuple(value / total for value in profile)


def hands_shape_decision(probabilities: Sequence[Sequence[float]]) -> dict[str, Any]:
    profiles = [_shape_profile(row) for row in probabilities]
    # One confident hand must not override an occluded or contradictory hand.
    active = len(profiles) == 2 and all(
        row[0] >= HANDS_SHAPE_MIN_QUERY_CONFIDENCE for row in profiles
    )
    return {
        "active": active,
        "reason": "bilateral_mitten_shape" if active else "hand_shape_not_confirmed",
        "classes": list(HANDS_SHAPE_CLASSES),
        "probabilities": [dict(zip(HANDS_SHAPE_CLASSES, row)) for row in profiles],
        "minimum_confidence": HANDS_SHAPE_MIN_QUERY_CONFIDENCE,
        "candidate_minimum_confidence": HANDS_SHAPE_MIN_CONFIDENCE,
        "ranking_changed": False,
    }


def rerank_candidates_with_hand_shape(
    candidates: Sequence[Mapping[str, Any]],
    query_probabilities: Sequence[Sequence[float]],
    candidate_probabilities: Mapping[int, Sequence[float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover at most five verified mitten candidates from the existing pool.

    This is an attribute gate, not an item-name prior or a replacement identity
    model. It runs only with bilateral evidence and a strongly contradictory
    baseline winner. Incomplete preview coverage leaves the ranking untouched.
    """
    output = [dict(row) for row in candidates]
    diagnostic = hands_shape_decision(query_probabilities)
    if not diagnostic["active"] or not output:
        return output, diagnostic
    if any(int(row["piece_id"]) not in candidate_probabilities for row in output):
        diagnostic.update(active=False, reason="candidate_preview_unavailable")
        return output, diagnostic
    profiles = {
        int(row["piece_id"]): _shape_profile(candidate_probabilities[int(row["piece_id"])])
        for row in output
    }
    previous_top1 = int(output[0]["piece_id"])
    diagnostic.update(candidate_profile_count=len(profiles), previous_top1_piece_id=previous_top1)
    if profiles[previous_top1][0] > 0.20:
        diagnostic.update(active=False, reason="baseline_shape_not_contradicted")
        return output, diagnostic
    matching = [row for row in output
                if row.get("link_type") == "exact_slot"
                and profiles[int(row["piece_id"])][0] >= HANDS_SHAPE_MIN_CONFIDENCE]
    matching.sort(key=lambda row: -profiles[int(row["piece_id"])][0])
    promoted = matching[:5]
    if not promoted:
        diagnostic.update(active=False, reason="no_verified_shape_candidates")
        return output, diagnostic
    promoted_ids = {int(row["piece_id"]) for row in promoted}
    for row in promoted:
        row["hand_shape_gate"] = {
            "previous_rank": int(row["rank"]),
            "mitten_probability": profiles[int(row["piece_id"])][0],
        }
    output = promoted + [row for row in output if int(row["piece_id"]) not in promoted_ids]
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    diagnostic.update(
        ranking_changed=True, current_top1_piece_id=int(output[0]["piece_id"]),
        promoted_piece_ids=[int(row["piece_id"]) for row in promoted],
    )
    return output, diagnostic
