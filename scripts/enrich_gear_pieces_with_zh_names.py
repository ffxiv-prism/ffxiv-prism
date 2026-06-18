from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import append_jsonl, read_jsonl, write_json  # noqa: E402
from ffxiv_prism.paths import EORZEA_METADATA_DIR, GAME_DATA_RAW_DIR  # noqa: E402


DEFAULT_PIECES = EORZEA_METADATA_DIR / "gear_pieces_raw.jsonl"
DEFAULT_FILTERED_MAPPING = GAME_DATA_RAW_DIR / "cafemaker_item_names_for_gear_pieces_raw.jsonl"
DEFAULT_FULL_MAPPING = GAME_DATA_RAW_DIR / "cafemaker_item_names_raw.jsonl"
DEFAULT_SUMMARY = EORZEA_METADATA_DIR / "gear_pieces_zh_name_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add official Chinese item names to Eorzea Collection gear piece rows."
    )
    parser.add_argument("--pieces", type=Path, default=DEFAULT_PIECES, help="Input gear piece JSONL.")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Item name mapping JSONL. Defaults to the filtered mapping if present, then the full mapping.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PIECES,
        help="Output JSONL. Defaults to updating gear_pieces_raw.jsonl in place.",
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Summary JSON path.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create an in-place backup file.")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between progress updates.",
    )
    args = parser.parse_args()

    started_at = time.monotonic()
    mapping_path = args.mapping or default_mapping_path()
    mapping = build_mapping(read_jsonl(mapping_path))
    rows = read_jsonl(args.pieces)
    total = len(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp_output.write_text("", encoding="utf-8")

    stats = Counter()
    unique_names: set[str] = set()
    unmatched_names: set[str] = set()
    ambiguous_names: set[str] = set()
    last_progress_at = 0.0

    for index, row in enumerate(rows, start=1):
        name_en = str(row.get("name") or row.get("name_en") or "").strip()
        unique_names.add(name_en)
        enriched = enrich_row(row, name_en, mapping)
        status = str(enriched.get("name_zh_match_status") or "unmapped")
        stats[status] += 1
        if status == "unmapped":
            unmatched_names.add(name_en)
        elif status == "ambiguous":
            ambiguous_names.add(name_en)
        append_jsonl(tmp_output, [enriched])

        now = time.monotonic()
        if args.progress_interval > 0 and now - last_progress_at >= args.progress_interval:
            print_progress(index, total, stats, started_at, final=False)
            last_progress_at = now

    if args.output.resolve() == args.pieces.resolve() and not args.no_backup:
        backup = args.pieces.with_suffix(args.pieces.suffix + ".before_zh_names")
        if not backup.exists():
            shutil.copy2(args.pieces, backup)
    tmp_output.replace(args.output)

    summary = {
        "pieces": str(args.pieces),
        "mapping": str(mapping_path),
        "output": str(args.output),
        "rows_total": total,
        "unique_english_names": len({name for name in unique_names if name}),
        "mapping_english_names": len(mapping),
        "matched_rows": stats["matched"],
        "ambiguous_rows": stats["ambiguous"],
        "unmapped_rows": stats["unmapped"],
        "matched_unique_names": count_unique_status(rows, mapping, "matched"),
        "ambiguous_unique_names": len({name for name in ambiguous_names if name}),
        "unmapped_unique_names": len({name for name in unmatched_names if name}),
        "unmapped_name_examples": sorted(name for name in unmatched_names if name)[:50],
        "ambiguous_name_examples": sorted(name for name in ambiguous_names if name)[:50],
        "finished_at": timestamp(),
    }
    write_json(args.summary, summary)
    print_progress(total, total, stats, started_at, final=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_mapping(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int | None, str | None]] = set()
    for row in rows:
        name_en = str(row.get("name_en") or "").strip()
        name_zh = str(row.get("name_zh") or "").strip()
        if not name_en or not name_zh:
            continue
        item_id = row.get("game_item_id")
        try:
            item_id = int(item_id) if item_id not in (None, "") else None
        except (TypeError, ValueError):
            item_id = None
        key = normalize_key(name_en)
        seen_key = (key, item_id, name_zh)
        if seen_key in seen:
            continue
        seen.add(seen_key)
        mapping[key].append(
            {
                "game_item_id": item_id,
                "name_en": name_en,
                "name_zh": name_zh,
                "source": row.get("source") or "https://cafemaker.wakingsands.com/Item",
            }
        )
    return dict(mapping)


def default_mapping_path() -> Path:
    if DEFAULT_FILTERED_MAPPING.exists():
        return DEFAULT_FILTERED_MAPPING
    return DEFAULT_FULL_MAPPING


def enrich_row(row: dict[str, Any], name_en: str, mapping: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    enriched = dict(row)
    enriched.pop("name_zh_source_url", None)
    enriched["name_en"] = name_en or None

    matches = mapping.get(normalize_key(name_en), [])
    if not matches:
        enriched["name_zh"] = None
        enriched["name_zh_source"] = None
        enriched["name_zh_match_status"] = "unmapped"
        enriched["game_item_id"] = None
        enriched["game_item_ids"] = []
        enriched["name_zh_candidates"] = []
        return enriched

    names_zh = sorted({str(match["name_zh"]) for match in matches if match.get("name_zh")})
    item_ids = sorted(
        {
            int(match["game_item_id"])
            for match in matches
            if match.get("game_item_id") not in (None, "")
        }
    )
    sources = sorted({str(match.get("source")) for match in matches if match.get("source")})

    enriched["game_item_id"] = item_ids[0] if len(item_ids) == 1 else None
    enriched["game_item_ids"] = item_ids
    enriched["name_zh_source"] = sources[0] if sources else "https://cafemaker.wakingsands.com/Item"
    enriched["name_zh_candidates"] = names_zh

    if len(names_zh) == 1:
        enriched["name_zh"] = names_zh[0]
        enriched["name_zh_match_status"] = "matched"
    else:
        enriched["name_zh"] = None
        enriched["name_zh_match_status"] = "ambiguous"
    return enriched


def count_unique_status(rows: list[dict[str, Any]], mapping: dict[str, list[dict[str, Any]]], status: str) -> int:
    names = {str(row.get("name") or row.get("name_en") or "").strip() for row in rows}
    count = 0
    for name in names:
        matches = mapping.get(normalize_key(name), [])
        names_zh = {str(match["name_zh"]) for match in matches if match.get("name_zh")}
        current_status = "unmapped"
        if len(names_zh) == 1:
            current_status = "matched"
        elif len(names_zh) > 1:
            current_status = "ambiguous"
        if current_status == status:
            count += 1
    return count


def normalize_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def print_progress(current: int, total: int, stats: Counter[str], started_at: float, final: bool) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = current / elapsed if current else 0.0
    percent = (current / total * 100.0) if total else 100.0
    line = (
        f"Progress: {current}/{total} ({percent:5.1f}%) "
        f"matched={stats['matched']} ambiguous={stats['ambiguous']} "
        f"unmapped={stats['unmapped']} rate={rate:5.1f}/s"
    )
    if final:
        print(f"\r{line}", file=sys.stderr, flush=True)
    else:
        print(f"\r{line}", end="", file=sys.stderr, flush=True)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
