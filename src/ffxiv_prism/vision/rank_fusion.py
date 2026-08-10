from __future__ import annotations

from typing import Any


def relative_ranking_margin(case: dict[str, Any]) -> float:
    results = case.get("results", [])
    if len(results) < 2:
        return 0.0
    first = float(results[0].get("score", 0.0))
    second = float(results[1].get("score", 0.0))
    return (first - second) / max(abs(first), 1e-9)


def route_case_results_by_margin(
    primary_cases: list[dict[str, Any]],
    secondary_cases: list[dict[str, Any]],
    *,
    primary_margin_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Select one complete ranking using only its normalized top-score margin."""
    secondary_by_key = {
        (str(case["example_id"]), str(case["slot"])): case
        for case in secondary_cases
    }
    routed = []
    for primary in primary_cases:
        key = (str(primary["example_id"]), str(primary["slot"]))
        secondary = secondary_by_key.get(key)
        use_secondary = (
            primary.get("status") == "eligible"
            and secondary is not None
            and secondary.get("status") == "eligible"
            and relative_ranking_margin(secondary)
            > relative_ranking_margin(primary) * primary_margin_scale
        )
        selected = secondary if use_secondary else primary
        output = dict(selected)
        output["routing"] = {
            "selected": "secondary" if use_secondary else "primary",
            "primary_margin": relative_ranking_margin(primary),
            "secondary_margin": (
                relative_ranking_margin(secondary)
                if secondary is not None
                else None
            ),
            "primary_margin_scale": primary_margin_scale,
        }
        routed.append(output)
    return routed


def fuse_case_results_rrf(
    primary_cases: list[dict[str, Any]],
    secondary_cases: list[dict[str, Any]],
    *,
    primary_weight: float = 0.6,
    top_k: int = 10,
    rank_constant: float = 10.0,
) -> list[dict[str, Any]]:
    """Fuse two aligned case result lists with weighted reciprocal-rank fusion."""
    secondary_by_key = {
        (str(case["example_id"]), str(case["slot"])): case
        for case in secondary_cases
    }
    fused_cases = []
    for primary in primary_cases:
        key = (str(primary["example_id"]), str(primary["slot"]))
        secondary = secondary_by_key.get(key)
        if primary.get("status") != "eligible" or secondary is None:
            fused_cases.append(dict(primary))
            continue
        by_piece: dict[int, dict[str, Any]] = {}
        for results, weight in (
            (primary.get("results", []), primary_weight),
            (secondary.get("results", []), 1.0 - primary_weight),
        ):
            for rank, result in enumerate(results, start=1):
                piece_id = int(result["piece_id"])
                row = by_piece.setdefault(
                    piece_id,
                    {"result": dict(result), "fusion_score": 0.0},
                )
                row["fusion_score"] += weight / (rank_constant + rank)
        ranked = sorted(
            by_piece.values(),
            key=lambda row: row["fusion_score"],
            reverse=True,
        )[:top_k]
        results = []
        for rank, row in enumerate(ranked, start=1):
            result = dict(row["result"])
            result["rank"] = rank
            result["fusion_score"] = float(row["fusion_score"])
            results.append(result)
        fused_cases.append(update_case_ranks(primary, results, top_k))
    return fused_cases


def update_case_ranks(
    source: dict[str, Any],
    results: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    eligible_ids = {int(value) for value in source.get("eligible_expected_piece_ids", [])}
    expected_key = source.get("expected_appearance_key")
    exact_rank = next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if int(result["piece_id"]) in eligible_ids
        ),
        None,
    )
    appearance_rank = next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if result.get("appearance_key") == expected_key
        ),
        None,
    )
    output = dict(source)
    output.update(
        results=results,
        rank=exact_rank,
        appearance_rank=appearance_rank,
        exact_hit_at_1=exact_rank == 1,
        exact_hit_at_k=exact_rank is not None and exact_rank <= top_k,
        appearance_hit_at_1=appearance_rank == 1,
        appearance_hit_at_k=appearance_rank is not None and appearance_rank <= top_k,
        hit_at_1=appearance_rank == 1,
        hit_at_k=appearance_rank is not None and appearance_rank <= top_k,
    )
    return output


def appearance_union_recall(
    first_cases: list[dict[str, Any]],
    second_cases: list[dict[str, Any]],
) -> float:
    """Return the oracle appearance recall in the union of two result lists."""
    second_by_key = {
        (str(case["example_id"]), str(case["slot"])): case
        for case in second_cases
    }
    eligible = 0
    hits = 0
    for first in first_cases:
        if first.get("status") != "eligible":
            continue
        key = (str(first["example_id"]), str(first["slot"]))
        second = second_by_key.get(key, {})
        expected = first.get("expected_appearance_key")
        eligible += 1
        if any(
            result.get("appearance_key") == expected
            for result in [
                *first.get("results", []),
                *second.get("results", []),
            ]
        ):
            hits += 1
    return hits / eligible if eligible else 0.0
