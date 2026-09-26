"""Experimental whole-foot identity check, NOT enabled by production inference.

The frozen evaluation found a fold/subgroup regression when this check was
isolated from the existing height reranker. Retain it for reproducibility,
not as a promoted model or a reason to change production candidate ranks.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

FOOTWEAR_CONSENSUS_CROPS = (
    "feet_full_left",
    "feet_full_right",
    "feet_full_context",
)


def footwear_consensus_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {str(row.get("name")): row for row in records}
    selected = []
    for name in FOOTWEAR_CONSENSUS_CROPS:
        row = by_name.get(name)
        confidence = float(row.get("confidence", 0.0)) if row is not None else 0.0
        if (
            row is None
            or row.get("visibility") != "visible"
            or not math.isfinite(confidence)
            or confidence < 0.8
            or "source_edge_truncation_risk" in row.get("tags", ())
        ):
            return []
        selected.append(dict(row))
    return selected


def apply_footwear_consensus(
    candidates: Sequence[Mapping[str, Any]],
    support: Sequence[Sequence[Mapping[str, Any]]],
    equivalence: Mapping[int, Sequence[int]],
    *,
    max_base_rank: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Move one existing Top-5 appearance only after three Top-1 votes agree.

    All other candidates retain their relative order and scores. In particular,
    Top-5/10/20 membership is unchanged. Official appearance equivalence, rather
    than names, colours, job or answer labels, defines agreement.
    """
    if max_base_rank < 1:
        raise ValueError("max_base_rank must be positive")
    output = [dict(row) for row in candidates]
    diagnostic = {"ranking_changed": False, "max_base_rank": max_base_rank}
    if len(support) != len(FOOTWEAR_CONSENSUS_CROPS) or any(
        not rows for rows in support
    ):
        return output, {**diagnostic, "reason": "incomplete_support"}

    def identity(row: Mapping[str, Any]) -> set[int]:
        piece_id = int(row["piece_id"])
        return {piece_id, *(int(v) for v in equivalence.get(piece_id, ()))}

    votes = [rows[0] for rows in support]
    diagnostic["support_piece_ids"] = [int(row["piece_id"]) for row in votes]
    if any(row.get("link_type") != "exact_slot" for row in votes):
        return output, {**diagnostic, "reason": "non_exact_slot_support"}
    common = set.intersection(*(identity(row) for row in votes))
    if not common:
        return output, {**diagnostic, "reason": "whole_foot_views_disagree"}
    index = next(
        (
            i
            for i, row in enumerate(output[:max_base_rank])
            if identity(row) & common and row.get("link_type") == "exact_slot"
        ),
        None,
    )
    if index is None:
        return output, {**diagnostic, "reason": "outside_baseline_shortlist"}
    if index == 0:
        return output, {**diagnostic, "reason": "baseline_already_agrees"}
    promoted = output.pop(index)
    promoted["footwear_consensus_gate"] = {"original_rank": index + 1}
    output.insert(0, promoted)
    for rank, row in enumerate(output, 1):
        row["rank"] = rank
    return output, {
        **diagnostic,
        "reason": "whole_foot_views_agree",
        "ranking_changed": True,
        "previous_rank": index + 1,
        "promoted_piece_id": int(promoted["piece_id"]),
    }
