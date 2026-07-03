from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from ffxiv_prism.jsonl import write_json


def nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        value = value[part]
    return value


def fuse_group_cases(
    cases: Iterable[dict[str, Any]],
    *,
    slots: set[str] | None = None,
    top_k: int = 10,
    rank_power: float = 1.0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if case.get("status") != "eligible":
            continue
        slot = str(case["slot"])
        if slots is not None and slot not in slots:
            continue
        grouped[(str(case["glamour_id"]), slot)].append(case)

    outcomes = []
    for (glamour_id, slot), views in sorted(grouped.items()):
        votes: Counter[int] = Counter()
        result_by_piece: dict[int, dict[str, Any]] = {}
        for view in views:
            for result in view.get("results", [])[:top_k]:
                piece_id = int(result["piece_id"])
                rank = max(1, int(result["rank"]))
                votes[piece_id] += 1.0 / (rank**rank_power)
                result_by_piece.setdefault(piece_id, result)

        ranked_piece_ids = sorted(
            votes,
            key=lambda piece_id: (-votes[piece_id], piece_id),
        )
        results = []
        for rank, piece_id in enumerate(ranked_piece_ids[:top_k], start=1):
            result = dict(result_by_piece[piece_id])
            result.update(rank=rank, group_vote=float(votes[piece_id]))
            results.append(result)

        expected_ids = {
            int(piece_id)
            for view in views
            for piece_id in view.get("eligible_expected_piece_ids", [])
        }
        expected_keys = {
            str(view["expected_appearance_key"])
            for view in views
            if view.get("expected_appearance_key")
        }
        exact_rank = next(
            (
                result["rank"]
                for result in results
                if int(result["piece_id"]) in expected_ids
            ),
            None,
        )
        appearance_rank = next(
            (
                result["rank"]
                for result in results
                if int(result["piece_id"]) in expected_ids
                or result.get("appearance_key") in expected_keys
            ),
            None,
        )
        outcomes.append(
            {
                "glamour_id": glamour_id,
                "slot": slot,
                "view_count": len(views),
                "example_ids": sorted(str(view["example_id"]) for view in views),
                "expected_names": sorted(
                    {
                        str(view["expected_name"])
                        for view in views
                        if view.get("expected_name")
                    }
                ),
                "eligible_expected_piece_ids": sorted(expected_ids),
                "rank": exact_rank,
                "appearance_rank": appearance_rank,
                "exact_hit_at_1": exact_rank == 1,
                "exact_hit_at_k": exact_rank is not None and exact_rank <= top_k,
                "appearance_hit_at_1": appearance_rank == 1,
                "appearance_hit_at_k": (
                    appearance_rank is not None and appearance_rank <= top_k
                ),
                "results": results,
            }
        )
    return outcomes


def summarize_group_cases(cases: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    count = len(cases)
    appearance_hits_at_1 = sum(case["appearance_hit_at_1"] for case in cases)
    appearance_hits_at_k = sum(case["appearance_hit_at_k"] for case in cases)
    exact_hits_at_1 = sum(case["exact_hit_at_1"] for case in cases)
    exact_hits_at_k = sum(case["exact_hit_at_k"] for case in cases)
    return {
        "group_case_count": count,
        "image_view_count": sum(case["view_count"] for case in cases),
        "appearance_hits_at_1": appearance_hits_at_1,
        "appearance_hits_at_k": appearance_hits_at_k,
        "appearance_recall_at_1": appearance_hits_at_1 / count if count else 0.0,
        "appearance_recall_at_k": appearance_hits_at_k / count if count else 0.0,
        "exact_hits_at_1": exact_hits_at_1,
        "exact_hits_at_k": exact_hits_at_k,
        "exact_recall_at_1": exact_hits_at_1 / count if count else 0.0,
        "exact_recall_at_k": exact_hits_at_k / count if count else 0.0,
        "top_k": top_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse multiple image rankings from one glamour using reciprocal-rank votes."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--section", default="evaluation")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--slots", nargs="+")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rank-power", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    section = nested_value(report, args.section)
    source_cases = section[f"{args.split}_cases"]
    outcomes = fuse_group_cases(
        source_cases,
        slots=set(args.slots) if args.slots else None,
        top_k=args.top_k,
        rank_power=args.rank_power,
    )
    summary = summarize_group_cases(outcomes, args.top_k)
    payload = {
        "report": args.report.as_posix(),
        "section": args.section,
        "split": args.split,
        "slots": args.slots,
        "strategy": "reciprocal_rank_vote",
        "rank_power": args.rank_power,
        "summary": summary,
        "cases": outcomes,
    }
    write_json(args.output, payload)
    print(
        f"groups={summary['group_case_count']} views={summary['image_view_count']} "
        f"appearance@1={summary['appearance_recall_at_1']:.3f} "
        f"appearance@{args.top_k}={summary['appearance_recall_at_k']:.3f} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
