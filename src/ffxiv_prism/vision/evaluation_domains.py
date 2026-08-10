from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from typing import Any, Iterable, Mapping, Sequence


DOMAIN_DIMENSIONS = ("model_gender", "model_race", "race_gender")


def domain_value(example: Mapping[str, Any], dimension: str) -> str:
    race = str(example.get("model_race") or "unknown").strip().lower()
    gender = str(example.get("model_gender") or "unknown").strip().lower()
    if dimension == "model_race":
        return race
    if dimension == "model_gender":
        return gender
    if dimension == "race_gender":
        return f"{race}|{gender}"
    raise ValueError(f"Unsupported domain dimension: {dimension}")


def build_domain_balance_weights(
    examples: Sequence[Mapping[str, Any]],
    *,
    group_by_glamour: bool = False,
    max_weight: float = 3.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return capped inverse-sqrt race/gender weights with mean weight one."""

    if max_weight < 1.0:
        raise ValueError("max_weight must be at least 1")
    groups_by_domain: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        groups_by_domain[domain_value(example, "race_gender")].add(
            str(example.get("glamour_id"))
        )
    counts = {
        domain: len(groups)
        for domain, groups in groups_by_domain.items()
        if groups
    }
    if not counts:
        return {}, {"domain_group_counts": {}, "domain_weights": {}}

    largest = max(counts.values())
    raw = {
        domain: min(max_weight, sqrt(largest / count))
        for domain, count in counts.items()
    }
    normalizer = (
        sum(counts[domain] * weight for domain, weight in raw.items())
        / sum(counts.values())
    )
    weights_by_domain = {
        domain: weight / normalizer
        for domain, weight in raw.items()
    }
    weights = {}
    for example in examples:
        key = (
            str(example.get("glamour_id"))
            if group_by_glamour
            else str(example.get("example_id"))
        )
        weights[key] = weights_by_domain[
            domain_value(example, "race_gender")
        ]
    return weights, {
        "dimension": "race_gender",
        "domain_group_counts": dict(sorted(counts.items())),
        "domain_weights": dict(sorted(weights_by_domain.items())),
        "max_weight": max_weight,
        "normalization": "capped_inverse_sqrt_with_group_weighted_mean_one",
    }


def summarize_domain_evaluation(
    cases: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    *,
    dimensions: Iterable[str] = DOMAIN_DIMENSIONS,
    min_groups_per_domain: int = 5,
    min_cases_per_domain: int = 10,
) -> dict[str, Any]:
    """Break retrieval results down by gender, race, and their intersection."""

    if min_groups_per_domain < 1 or min_cases_per_domain < 1:
        raise ValueError("minimum domain coverage thresholds must be positive")
    example_by_id = {
        str(example.get("example_id")): example
        for example in examples
    }
    orphan_cases = [
        str(case.get("example_id"))
        for case in cases
        if str(case.get("example_id")) not in example_by_id
    ]
    selected_dimensions = tuple(dimensions)
    dimension_payload = {}
    for dimension in selected_dimensions:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        expected_values = {
            domain_value(example, dimension)
            for example in examples
        }
        for case in cases:
            example = example_by_id.get(str(case.get("example_id")))
            if example is not None:
                grouped[domain_value(example, dimension)].append(case)
        domains = {
            value: _summarize_cases(grouped.get(value, []))
            for value in sorted(expected_values)
        }
        undercovered = [
            value
            for value, summary in domains.items()
            if summary["glamour_group_count"] < min_groups_per_domain
            or summary["eligible_case_count"] < min_cases_per_domain
        ]
        scorable = [
            summary
            for summary in domains.values()
            if summary["eligible_case_count"] > 0
        ]
        dimension_payload[dimension] = {
            "domains": domains,
            "domain_count": len(domains),
            "scorable_domain_count": len(scorable),
            "macro_appearance_recall_at_1": (
                sum(row["appearance_recall_at_1"] for row in scorable)
                / len(scorable)
                if scorable
                else 0.0
            ),
            "worst_domain_appearance_recall_at_1": (
                min(row["appearance_recall_at_1"] for row in scorable)
                if scorable
                else 0.0
            ),
            "undercovered_domains": undercovered,
            "coverage_ready": not undercovered and bool(domains),
        }
    return {
        "overall": _summarize_cases(cases),
        "coverage_policy": {
            "min_groups_per_domain": min_groups_per_domain,
            "min_cases_per_domain": min_cases_per_domain,
        },
        "dimensions": dimension_payload,
        "integrity": {
            "case_count": len(cases),
            "orphan_case_count": len(orphan_cases),
            "orphan_example_ids": sorted(set(orphan_cases)),
        },
        "ready": (
            not orphan_cases
            and bool(dimension_payload)
            and all(
                payload["coverage_ready"]
                for payload in dimension_payload.values()
            )
        ),
    }


def _summarize_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [case for case in cases if case.get("status") == "eligible"]
    groups = {str(case.get("glamour_id")) for case in eligible}
    hits_at_1 = sum(bool(case.get("appearance_hit_at_1")) for case in eligible)
    hits_at_k = sum(bool(case.get("appearance_hit_at_k")) for case in eligible)
    reciprocal_ranks = [
        1.0 / int(case["appearance_rank"])
        if case.get("appearance_rank")
        else 0.0
        for case in eligible
    ]
    count = len(eligible)
    return {
        "case_count": len(cases),
        "eligible_case_count": count,
        "glamour_group_count": len(groups),
        "appearance_hits_at_1": hits_at_1,
        "appearance_hits_at_k": hits_at_k,
        "appearance_recall_at_1": hits_at_1 / count if count else 0.0,
        "appearance_recall_at_k": hits_at_k / count if count else 0.0,
        "appearance_mrr": (
            sum(reciprocal_ranks) / count
            if count
            else 0.0
        ),
        "status_counts": dict(
            sorted(Counter(str(case.get("status")) for case in cases).items())
        ),
    }
