from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bs4 import BeautifulSoup  # noqa: E402

from ffxiv_prism.jsonl import append_jsonl, read_jsonl, write_json  # noqa: E402
from ffxiv_prism.paths import EORZEA_METADATA_DIR, GAME_DATA_RAW_DIR  # noqa: E402


DEFAULT_PIECES = EORZEA_METADATA_DIR / "gear_pieces_raw.jsonl"
DEFAULT_CACHE = GAME_DATA_RAW_DIR / "lodestone_equip_conflicts_raw.jsonl"
DEFAULT_SUMMARY = EORZEA_METADATA_DIR / "gear_pieces_lodestone_equip_conflict_summary.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
SLOT_TERMS = {
    "head": "HEAD",
    "headgear": "HEAD",
    "arm": "HANDS",
    "arms": "HANDS",
    "hand": "HANDS",
    "hands": "HANDS",
    "glove": "HANDS",
    "gloves": "HANDS",
    "leg": "LEGS",
    "legs": "LEGS",
    "foot": "FEET",
    "feet": "FEET",
    "shoe": "FEET",
    "shoes": "FEET",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Use Lodestone item notes as a fallback equip-conflict source.")
    parser.add_argument("--pieces", type=Path, default=DEFAULT_PIECES, help="Input gear piece JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PIECES, help="Output gear piece JSONL.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="Lodestone lookup cache JSONL.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Summary JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional lookup limit for testing.")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between network requests.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Network timeout.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create an in-place backup file.")
    args = parser.parse_args()

    rows = read_jsonl(args.pieces)
    cache = load_cache(args.cache)
    pending = fallback_candidates(rows)
    if args.limit is not None:
        pending = pending[: args.limit]

    looked_up = 0
    for row in pending:
        url = str(row.get("source_item_url") or "")
        if not url or url in cache:
            continue
        record = fetch_lodestone_record(url, timeout=args.timeout)
        append_jsonl(args.cache, [record])
        cache[url] = record
        looked_up += 1
        if args.delay > 0:
            time.sleep(args.delay)

    updated_rows, stats = apply_lodestone_cache(rows, cache)
    write_jsonl_with_optional_backup(args.pieces, args.output, updated_rows, args.no_backup)
    summary = {
        "input": str(args.pieces),
        "output": str(args.output),
        "cache": str(args.cache),
        "fallback_candidate_rows": len(pending),
        "looked_up_this_run": looked_up,
        **stats,
        "finished_at": timestamp(),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def fallback_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("source_item_url")
        and not row.get("official_occupied_slots")
        and row.get("visual_inferred_blocked_slots")
    ]


def fetch_lodestone_record(url: str, timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
        phrase = extract_conflict_phrase(html)
        blocked_slots = blocked_slots_from_phrase(phrase)
        return {
            "source_item_url": url,
            "status": "matched" if phrase else "no_conflict_note",
            "conflict_phrase": phrase,
            "blocked_slots": blocked_slots,
            "checked_at": timestamp(),
        }
    except Exception as exc:
        return {
            "source_item_url": url,
            "status": "failed",
            "error": str(exc),
            "blocked_slots": [],
            "checked_at": timestamp(),
        }


def extract_conflict_phrase(html: str) -> str | None:
    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    for line in text.splitlines():
        if re.search(r"Cannot equip with .+ gear\.", line, re.I):
            return " ".join(line.split())
    return None


def blocked_slots_from_phrase(phrase: str | None) -> list[str]:
    if not phrase:
        return []
    phrase = phrase.casefold()
    result: list[str] = []
    for term, slot in SLOT_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\b", phrase) and slot not in result:
            result.append(slot)
    return result


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        url = str(row.get("source_item_url") or "")
        if url:
            cache[url] = row
    return cache


def apply_lodestone_cache(
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    updated: list[dict[str, Any]] = []
    stats = {"rows_confirmed": 0, "rows_cleared": 0, "rows_failed": 0}
    for row in rows:
        current = dict(row)
        if current.get("official_occupied_slots") or not current.get("visual_inferred_blocked_slots"):
            updated.append(current)
            continue
        record = cache.get(str(current.get("source_item_url") or ""))
        if not record:
            updated.append(current)
            continue
        if record.get("status") == "failed":
            stats["rows_failed"] += 1
            updated.append(current)
            continue
        blocked_slots = list(record.get("blocked_slots") or [])
        if blocked_slots:
            occupied_slots = unique([current.get("slot"), *blocked_slots])
            current["occupied_slots"] = occupied_slots
            current["blocked_slots"] = blocked_slots
            current["equip_conflict_source"] = "lodestone_item_note"
            current["equip_conflict_note"] = record.get("conflict_phrase")
            stats["rows_confirmed"] += 1
        else:
            current["occupied_slots"] = [current.get("slot")] if current.get("slot") else []
            current["blocked_slots"] = []
            current["equip_conflict_source"] = None
            current["equip_conflict_note"] = None
            stats["rows_cleared"] += 1
        updated.append(current)
    return updated, stats


def write_jsonl_with_optional_backup(
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    no_backup: bool,
) -> None:
    if output_path.resolve() == input_path.resolve() and input_path.exists() and not no_backup:
        backup = input_path.with_suffix(input_path.suffix + ".before_lodestone_equip_notes")
        if not backup.exists():
            shutil.copy2(input_path, backup)
    output_path.write_text("", encoding="utf-8")
    append_jsonl(output_path, rows)


def unique(values: list[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
