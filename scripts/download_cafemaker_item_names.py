from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.boilmaster import (  # noqa: E402
    DEFAULT_API_ROOT,
    fetch_sheet_page,
    next_after,
    row_fields,
    row_id,
)
from ffxiv_prism.jsonl import (  # noqa: E402
    append_jsonl,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from ffxiv_prism.paths import GAME_DATA_RAW_DIR  # noqa: E402


DEFAULT_BASE_URL = f"{DEFAULT_API_ROOT}/Item"
DEFAULT_OUTPUT = GAME_DATA_RAW_DIR / "cafemaker_item_names_raw.jsonl"
DEFAULT_SUMMARY = GAME_DATA_RAW_DIR / "cafemaker_item_names_summary.json"
ITEM_FIELDS = (
    "Name@lang(chs)",
    "Name@lang(en)",
    "EquipSlotCategory@as(raw)",
    "ItemUICategory@lang(en).Name",
    "DyeCount",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache Boilmaster Item names, equipment slots, and dye channel counts."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Boilmaster Item sheet endpoint.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Output summary JSON path.")
    parser.add_argument("--page-size", type=int, default=1000, help="Rows requested per page.")
    parser.add_argument("--limit-pages", type=int, default=None, help="Optional page limit for debugging.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Network timeout per request.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between requests.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per page.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing .tmp output file.")
    parser.add_argument("--progress-interval", type=float, default=5.0, help="Seconds between progress updates.")
    args = parser.parse_args()
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive")

    started_at = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.resume and tmp_output.exists():
        existing_rows = read_jsonl(tmp_output)
        if any("dye_count" not in row for row in existing_rows):
            raise ValueError("Legacy Item download lacks dye_count; rerun without --resume")
        written = len(existing_rows)
        after = resume_after(existing_rows)
        pages_fetched = 0
        print(
            f"Resuming after row_id={after} with {written} cached rows.",
            file=sys.stderr,
            flush=True,
        )
    else:
        tmp_output.write_text("", encoding="utf-8")
        after = None
        pages_fetched = 0
        written = 0

    last_progress_at = 0.0
    schema: str | None = None
    game_version: str | None = None

    while True:
        payload = fetch_sheet_page(
            args.base_url,
            fields=ITEM_FIELDS,
            after=after,
            limit=args.page_size,
            timeout=args.timeout,
            retries=args.retries,
        )
        rows = payload["rows"]
        if not rows:
            break
        pages_fetched += 1
        schema = str(payload.get("schema") or "") or schema
        game_version = str(payload.get("version") or "") or game_version

        normalized_rows = [normalize_item_row(row, source=args.base_url) for row in rows]
        written += append_jsonl(tmp_output, normalized_rows)
        after = next_after(rows, after)

        now = time.monotonic()
        if args.progress_interval > 0 and now - last_progress_at >= args.progress_interval:
            print_progress(pages_fetched, written, started_at, final=False)
            last_progress_at = now

        if args.limit_pages is not None and pages_fetched >= args.limit_pages:
            break
        if args.delay > 0:
            time.sleep(args.delay)

    deduped_rows, duplicate_rows_removed = dedupe_rows(read_jsonl(tmp_output))
    if not deduped_rows:
        raise RuntimeError(
            f"Boilmaster returned no Item rows; refusing to replace existing output: {args.output}"
        )
    wearable_rows = [row for row in deduped_rows if (row.get("equip_slot_category_id") or 0) > 0]
    known_dyes = [row for row in wearable_rows if row.get("dye_count") is not None]
    if wearable_rows and not known_dyes:
        raise RuntimeError("No wearable DyeCount values returned; refusing to replace existing output")
    write_jsonl_atomic(args.output, deduped_rows)
    tmp_output.unlink(missing_ok=True)
    final_counts = inspect_output(args.output)
    written = final_counts["rows"]
    summary = {
        "source": args.base_url,
        "fields": list(ITEM_FIELDS),
        "output": str(args.output),
        "rows_written": written,
        "duplicate_rows_removed": duplicate_rows_removed,
        "empty_name_en": final_counts["empty_name_en"],
        "wearable_rows_with_dye_count": len(known_dyes),
        "wearable_rows_missing_dye_count": len(wearable_rows) - len(known_dyes),
        "dyeable_wearable_rows": sum(row["dye_count"] > 0 for row in known_dyes),
        "pages_fetched": pages_fetched,
        "page_size": args.page_size,
        "results_total": written,
        "last_row_id": after,
        "schema": schema,
        "game_version": game_version,
        "finished_at": timestamp(),
    }
    write_json_atomic(args.summary, summary)
    print_progress(pages_fetched, written, started_at, final=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def normalize_item_row(row: object, source: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    item_id = row_id(row)
    fields = row_fields(row)
    return {
        "game_item_id": item_id,
        "name_zh": str(fields.get("Name@lang(chs)") or "").strip() or None,
        "name_en": str(fields.get("Name@lang(en)") or "").strip() or None,
        "equip_slot_category_id": normalize_int(fields.get("EquipSlotCategory@as(raw)")),
        "item_ui_category_en": normalize_nested_name(fields.get("ItemUICategory@lang(en)")),
        "dye_count": normalize_dye_count(fields.get("DyeCount")),
        "source": source,
    }


def normalize_dye_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    count = normalize_int(value)
    return count if count is not None and count >= 0 else None


def normalize_int(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalize_nested_name(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    fields = value.get("fields")
    if not isinstance(fields, dict):
        return None
    return str(fields.get("Name") or "").strip() or None


def resume_after(rows: list[dict[str, Any]]) -> int | None:
    if not rows:
        return None
    item_ids = [normalize_int(row.get("game_item_id")) for row in rows]
    if any(item_id is None for item_id in item_ids):
        raise ValueError("Cannot resume Item download with a missing game_item_id")
    return max(item_id for item_id in item_ids if item_id is not None)


def inspect_output(path: Path) -> dict[str, int]:
    rows = 0
    empty_name_en = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            row = json.loads(line)
            if not row.get("name_en"):
                empty_name_en += 1
    return {"rows": rows, "empty_name_en": empty_name_en}


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    duplicate_rows_removed = 0
    for row in rows:
        item_id = row.get("game_item_id")
        if isinstance(item_id, int):
            if item_id in seen_ids:
                duplicate_rows_removed += 1
                continue
            seen_ids.add(item_id)
        deduped.append(row)
    return deduped, duplicate_rows_removed


def print_progress(page: int, written: int, started_at: float, final: bool) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = written / elapsed if written else 0.0
    line = f"Progress: pages={page} rows={written} rate={rate:5.1f}/s"
    if final:
        print(f"\r{line}", file=sys.stderr, flush=True)
    else:
        print(f"\r{line}", end="", file=sys.stderr, flush=True)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
