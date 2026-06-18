from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import append_jsonl, write_json  # noqa: E402
from ffxiv_prism.paths import GAME_DATA_RAW_DIR  # noqa: E402


DEFAULT_BASE_URL = "https://cafemaker.wakingsands.com/Item"
DEFAULT_OUTPUT = GAME_DATA_RAW_DIR / "cafemaker_item_names_raw.jsonl"
DEFAULT_SUMMARY = GAME_DATA_RAW_DIR / "cafemaker_item_names_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache CafeMaker/WakingSands Item names for English-to-Chinese item name mapping."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="CafeMaker Item endpoint.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Output summary JSON path.")
    parser.add_argument("--page-size", type=int, default=1000, help="Rows per request.")
    parser.add_argument("--limit-pages", type=int, default=None, help="Optional page limit for debugging.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Network timeout per request.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between requests.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per page.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing .tmp output file.")
    parser.add_argument("--progress-interval", type=float, default=5.0, help="Seconds between progress updates.")
    args = parser.parse_args()

    started_at = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.resume and tmp_output.exists():
        written = count_lines(tmp_output)
        pages_fetched = written // args.page_size
        if written % args.page_size:
            raise ValueError(
                f"Cannot resume partial page in {tmp_output}: {written} rows is not divisible by {args.page_size}."
            )
        page = pages_fetched + 1
        print(f"Resuming from page {page} after {written} cached rows.", file=sys.stderr, flush=True)
    else:
        tmp_output.write_text("", encoding="utf-8")
        page = 1
        pages_fetched = 0

    page_total: int | None = None
    total_rows: int | None = None
    empty_name_en = 0
    last_progress_at = 0.0

    while True:
        if args.limit_pages is not None and page > args.limit_pages:
            break
        payload = fetch_page(
            args.base_url,
            page=page,
            page_size=args.page_size,
            timeout=args.timeout,
            retries=args.retries,
        )
        pagination = payload.get("Pagination") or {}
        rows = payload.get("Results") or []
        if not isinstance(rows, list):
            raise TypeError("CafeMaker response Results must be a list")
        pages_fetched += 1

        if page_total is None:
            page_total = int(pagination.get("PageTotal") or 0) or None
            total_rows = int(pagination.get("ResultsTotal") or 0) or None

        normalized_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = normalize_item_row(row)
            if not normalized.get("name_en"):
                empty_name_en += 1
            normalized_rows.append(normalized)
        written += append_jsonl(tmp_output, normalized_rows)

        now = time.monotonic()
        if args.progress_interval > 0 and now - last_progress_at >= args.progress_interval:
            print_progress(pages_fetched, page_total, written, total_rows, started_at, final=False)
            last_progress_at = now

        if not pagination.get("PageNext"):
            break
        if args.limit_pages is not None and pages_fetched >= args.limit_pages:
            break
        page += 1
        if args.delay > 0:
            time.sleep(args.delay)

    tmp_output.replace(args.output)
    final_counts = inspect_output(args.output)
    written = final_counts["rows"]
    empty_name_en = final_counts["empty_name_en"]
    summary = {
        "source": args.base_url,
        "columns": ["ID", "Name", "Name_en"],
        "output": str(args.output),
        "rows_written": written,
        "empty_name_en": empty_name_en,
        "pages_fetched": pages_fetched,
        "page_size": args.page_size,
        "results_total": total_rows,
        "finished_at": timestamp(),
    }
    write_json(args.summary, summary)
    print_progress(pages_fetched, page_total, written, total_rows, started_at, final=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def fetch_page(base_url: str, page: int, page_size: int, timeout: float, retries: int) -> dict[str, Any]:
    query = urlencode(
        {
            "columns": "ID,Name,Name_en",
            "limit": str(page_size),
            "page": str(page),
        }
    )
    url = f"{base_url}?{query}"
    last_error: Exception | None = None
    for attempt in range(1, max(retries, 1) + 1):
        try:
            request = Request(url, headers={"User-Agent": "ffxiv-prism/0.1 item-name-cache"})
            with urlopen(request, timeout=timeout) as response:
                data = response.read()
            payload = json.loads(data.decode("utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("CafeMaker response must be a JSON object")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def normalize_item_row(row: dict[str, Any]) -> dict[str, Any]:
    item_id = row.get("ID")
    try:
        item_id = int(item_id) if item_id not in (None, "") else None
    except (TypeError, ValueError):
        item_id = None
    return {
        "game_item_id": item_id,
        "name_zh": str(row.get("Name") or "").strip() or None,
        "name_en": str(row.get("Name_en") or "").strip() or None,
        "source": DEFAULT_BASE_URL,
    }


def count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _line in handle:
            count += 1
    return count


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


def print_progress(
    page: int,
    page_total: int | None,
    written: int,
    total_rows: int | None,
    started_at: float,
    final: bool,
) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = written / elapsed if written else 0.0
    if page_total:
        percent = min(page / page_total * 100.0, 100.0)
        page_text = f"{page}/{page_total} ({percent:5.1f}%)"
    else:
        page_text = str(page)
    total_text = str(total_rows) if total_rows is not None else "unknown"
    line = f"Progress: pages={page_text} rows={written}/{total_text} rate={rate:5.1f}/s"
    if final:
        print(f"\r{line}", file=sys.stderr, flush=True)
    else:
        print(f"\r{line}", end="", file=sys.stderr, flush=True)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
