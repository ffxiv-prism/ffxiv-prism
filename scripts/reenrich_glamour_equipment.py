from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ffxiv_prism.collectors.eorzea_collection.glamour_parser import (
    enrich_equipment_from_sqlite,
)
from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import (
    EORZEA_GLAMOUR_METADATA_DIR,
    PROCESSED_DIR,
    portable_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh local piece and appearance-family matches without refetching pages."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=EORZEA_GLAMOUR_METADATA_DIR / "glamour_examples_raw.jsonl",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--db", type=Path, default=PROCESSED_DIR / "eorzea.sqlite")
    parser.add_argument(
        "--appearance-equivalence",
        type=Path,
        help="Optional JSON produced by build_piece_appearance_equivalence.py.",
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    output = args.output or args.input
    summary_path = args.summary or output.with_name(f"{output.stem}_enrichment_summary.json")
    rows = read_jsonl(args.input)
    appearance_equivalence: dict[int, list[int]] = {}
    if args.appearance_equivalence:
        payload = json.loads(args.appearance_equivalence.read_text(encoding="utf-8"))
        appearance_equivalence = {
            int(piece_id): [int(value) for value in values]
            for piece_id, values in payload.get("direct_equivalents", {}).items()
        }
    match_types: Counter[str] = Counter()
    equivalence_rules: Counter[str] = Counter()
    for row in rows:
        enrich_equipment_from_sqlite(
            row,
            args.db,
            appearance_equivalence=appearance_equivalence,
        )
        for item in row.get("equipment", []):
            match_types[str(item.get("piece_match_type") or "unknown")] += 1
            rule = item.get("appearance_equivalence_rule")
            if rule:
                equivalence_rules[str(rule)] += 1

    write_jsonl(output, rows)
    write_json(
        summary_path,
        {
            "database": portable_path(args.db),
            "appearance_equivalence": portable_path(args.appearance_equivalence),
            "equivalence_rule_counts": dict(sorted(equivalence_rules.items())),
            "input": portable_path(args.input),
            "match_type_counts": dict(sorted(match_types.items())),
            "output": portable_path(output),
            "records": len(rows),
        },
    )
    print(
        f"records={len(rows)} embedding_equivalences="
        f"{equivalence_rules.get('canonical_embedding_similarity', 0)} output={output}"
    )


if __name__ == "__main__":
    main()
