from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import EORZEA_GLAMOUR_METADATA_DIR, INTERIM_DIR, portable_path
from ffxiv_prism.vision.evaluation_split import stable_new_group_role

DEFAULT_INPUT = EORZEA_GLAMOUR_METADATA_DIR / "glamour_examples_raw.jsonl"
DEFAULT_OUTPUT = INTERIM_DIR / "glamour_expansion_queue.txt"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a bounded new-glamour queue from source-backed related links."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="Also write numbered queue parts of at most this many ids.",
    )
    parser.add_argument(
        "--exclude-file",
        action="append",
        type=Path,
        default=[],
        help="Previously attempted queue file; may be passed more than once.",
    )
    parser.add_argument(
        "--prefer-recent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--source-model-races",
        nargs="+",
        help="Only follow related links from source glamours with these model races.",
    )
    parser.add_argument(
        "--source-model-genders",
        nargs="+",
        help="Only follow related links from source glamours with these model genders.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["train"],
        choices=(
            "train",
            "validation",
            "blind_test",
            "development_test",
            "confirmation_test",
        ),
        help="Keep only new glamour ids assigned to these stable split roles.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    existing = {int(row["glamour_id"]) for row in rows if row.get("glamour_id")}
    excluded = set()
    for path in args.exclude_file:
        if not path.is_file():
            continue
        excluded.update(
            int(value.strip())
            for value in path.read_text(encoding="utf-8").splitlines()
            if value.strip().isdigit()
        )
    frequency: Counter[int] = Counter()
    sources: dict[int, set[int]] = defaultdict(set)
    allowed_races = set(args.source_model_races or [])
    allowed_genders = {
        value.strip().lower() for value in (args.source_model_genders or [])
    }
    eligible_source_count = 0
    for row in rows:
        model_race = str(row.get("model_race") or "").strip()
        model_gender = str(row.get("model_gender") or "").strip().lower()
        if allowed_races and model_race not in allowed_races:
            continue
        if allowed_genders and model_gender not in allowed_genders:
            continue
        eligible_source_count += 1
        source_id = int(row["glamour_id"])
        for url in row.get("related_glamour_urls", []):
            match = re.search(r"/glamour/(\d+)", str(url))
            if not match:
                continue
            glamour_id = int(match.group(1))
            if glamour_id in existing or glamour_id in excluded:
                continue
            frequency[glamour_id] += 1
            sources[glamour_id].add(source_id)

    allowed_roles = set(args.roles)
    candidates = sorted(
        filter_candidate_roles(frequency, allowed_roles),
        key=lambda glamour_id: (
            len(sources[glamour_id]),
            frequency[glamour_id],
            glamour_id if args.prefer_recent else -glamour_id,
        ),
        reverse=True,
    )
    selected = candidates[: max(0, args.limit)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{glamour_id}\n" for glamour_id in selected),
        encoding="utf-8",
    )
    shard_paths = []
    if args.shard_size > 0:
        for start in range(0, len(selected), args.shard_size):
            shard_path = args.output.with_name(
                f"{args.output.stem}.part{start // args.shard_size + 1:03d}"
                f"{args.output.suffix}"
            )
            shard_path.write_text(
                "".join(
                    f"{glamour_id}\n"
                    for glamour_id in selected[start : start + args.shard_size]
                ),
                encoding="utf-8",
            )
            shard_paths.append(portable_path(shard_path))
    summary = {
        "source": portable_path(args.input),
        "existing_glamour_count": len(existing),
        "excluded_attempted_count": len(excluded),
        "discovered_candidate_count": len(candidates),
        "selected_count": len(selected),
        "prefer_recent": args.prefer_recent,
        "source_model_races": sorted(allowed_races),
        "source_model_genders": sorted(allowed_genders),
        "roles": sorted(allowed_roles),
        "eligible_source_glamour_count": eligible_source_count,
        "output": portable_path(args.output),
        "shards": shard_paths,
        "selected": [
            {
                "glamour_id": glamour_id,
                "related_link_count": frequency[glamour_id],
                "source_glamour_count": len(sources[glamour_id]),
                "role": stable_new_group_role(glamour_id),
            }
            for glamour_id in selected
        ],
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(
        f"existing={len(existing)} sources={eligible_source_count} "
        f"discovered={len(candidates)} "
        f"selected={len(selected)} output={args.output}"
    )
    return 0


def filter_candidate_roles(
    glamour_ids: object,
    allowed_roles: set[str],
) -> list[int]:
    return [
        int(glamour_id)
        for glamour_id in glamour_ids
        if stable_new_group_role(int(glamour_id)) in allowed_roles
    ]


if __name__ == "__main__":
    raise SystemExit(main())
