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
DEFAULT_OUTPUT = GAME_DATA_RAW_DIR / "cafemaker_item_models_raw.jsonl"
DEFAULT_SUMMARY = GAME_DATA_RAW_DIR / "cafemaker_item_models_summary.json"
ITEM_FIELDS = (
    "Name@lang(chs)",
    "Name@lang(en)",
    "ModelMain",
    "ModelSub",
    "EquipSlotCategory@as(raw)",
    "ItemUICategory@as(raw)",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cache XIVAPI Item model fields used by HuijiWiki's explicit "
            "same-model rule."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--limit-pages", type=int)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.resume and temporary.exists():
        cached_rows = read_jsonl(temporary)
        written = len(cached_rows)
        after = resume_after(cached_rows)
        pages_fetched = 0
        print(
            f"Resuming after row_id={after} with {written} cached rows.",
            file=sys.stderr,
            flush=True,
        )
    else:
        temporary.write_text("", encoding="utf-8")
        written = 0
        after = None
        pages_fetched = 0

    started = time.monotonic()
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
        normalized = [normalize_item_row(row, source=args.base_url) for row in rows]
        written += append_jsonl(temporary, normalized)
        after = next_after(rows, after)

        now = time.monotonic()
        if args.progress_interval > 0 and now - last_progress_at >= args.progress_interval:
            print_progress(pages_fetched, written, started, final=False)
            last_progress_at = now
        if args.limit_pages is not None and pages_fetched >= args.limit_pages:
            break
        if args.delay > 0:
            time.sleep(args.delay)

    deduped, duplicate_rows_removed = dedupe_by_id(read_jsonl(temporary))
    if not deduped:
        raise RuntimeError(
            f"XIVAPI returned no Item rows; refusing to replace existing output: {args.output}"
        )
    write_jsonl_atomic(args.output, deduped)
    temporary.unlink(missing_ok=True)
    summary = {
        "source": args.base_url,
        "fields": list(ITEM_FIELDS),
        "output": args.output.as_posix(),
        "rows_written": len(deduped),
        "rows_with_model_main": sum(any(row.get("model_main") or []) for row in deduped),
        "duplicate_rows_removed": duplicate_rows_removed,
        "pages_fetched": pages_fetched,
        "page_size": args.page_size,
        "results_total": len(deduped),
        "last_row_id": after,
        "schema": schema,
        "game_version": game_version,
        "finished_at": timestamp(),
    }
    write_json_atomic(args.summary, summary)
    print_progress(pages_fetched, len(deduped), started, final=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def normalize_item_row(row: object, source: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    fields = row_fields(row)
    return {
        "game_item_id": row_id(row),
        "name_zh": normalize_text(fields.get("Name@lang(chs)")),
        "name_en": normalize_text(fields.get("Name@lang(en)")),
        "model_main": unpack_model(fields.get("ModelMain")),
        "model_sub": unpack_model(fields.get("ModelSub")),
        "equip_slot_category_id": normalize_int(fields.get("EquipSlotCategory@as(raw)")),
        "item_ui_category_id": normalize_int(fields.get("ItemUICategory@as(raw)")),
        "source": source,
    }


def unpack_model(value: object) -> list[int]:
    packed = normalize_int(value)
    if packed is None:
        return []
    return [(packed >> (16 * index)) & 0xFFFF for index in range(4)]


def normalize_int(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalize_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def resume_after(rows: list[dict[str, Any]]) -> int | None:
    ids = [normalize_int(row.get("game_item_id")) for row in rows]
    if any(item_id is None for item_id in ids):
        raise ValueError("Cannot resume Item model download with a missing game_item_id")
    return max((item_id for item_id in ids if item_id is not None), default=None)


def dedupe_by_id(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    duplicates = 0
    for row in rows:
        item_id = row.get("game_item_id")
        if not isinstance(item_id, int) or item_id in seen:
            duplicates += 1
            continue
        seen.add(item_id)
        output.append(row)
    return output, duplicates


def print_progress(page: int, written: int, started: float, final: bool) -> None:
    elapsed = max(time.monotonic() - started, 0.001)
    line = f"Progress: pages={page} rows={written} rate={written / elapsed:5.1f}/s"
    if final:
        print(f"\r{line}", file=sys.stderr, flush=True)
    else:
        print(f"\r{line}", end="", file=sys.stderr, flush=True)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
