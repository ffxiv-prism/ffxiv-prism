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

from ffxiv_prism.jsonl import append_jsonl, read_jsonl, write_json
from ffxiv_prism.paths import GAME_DATA_RAW_DIR


DEFAULT_BASE_URL = "https://cafemaker.wakingsands.com/Item"
DEFAULT_OUTPUT = GAME_DATA_RAW_DIR / "cafemaker_item_models_raw.jsonl"
DEFAULT_SUMMARY = GAME_DATA_RAW_DIR / "cafemaker_item_models_summary.json"
ITEM_COLUMNS = [
    "ID",
    "Name",
    "Name_en",
    "ModelMain",
    "ModelSub",
    "EquipSlotCategoryTargetID",
    "ItemUICategoryTargetID",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cache Item model fields used by HuijiWiki's explicit "
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
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.resume and temporary.exists():
        written = count_lines(temporary)
        if written % args.page_size:
            raise ValueError(
                f"Cannot resume a partial page: {temporary} has {written} rows"
            )
        page = written // args.page_size + 1
    else:
        temporary.write_text("", encoding="utf-8")
        written = 0
        page = 1

    started = time.monotonic()
    pages_fetched = 0
    results_total: int | None = None
    while args.limit_pages is None or pages_fetched < args.limit_pages:
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
        normalized = [
            normalize_item_row(row)
            for row in rows
            if isinstance(row, dict)
        ]
        written += append_jsonl(temporary, normalized)
        pages_fetched += 1
        results_total = normalize_int(pagination.get("ResultsTotal"))
        elapsed = max(time.monotonic() - started, 0.001)
        print(
            f"\rpages={pages_fetched} rows={written}/{results_total or '?'} "
            f"rate={written / elapsed:.1f}/s",
            end="",
            file=sys.stderr,
            flush=True,
        )
        if not pagination.get("PageNext"):
            break
        page += 1
        if args.delay:
            time.sleep(args.delay)
    print(file=sys.stderr)

    deduped = dedupe_by_id(read_jsonl(temporary))
    args.output.write_text("", encoding="utf-8")
    append_jsonl(args.output, deduped)
    temporary.unlink(missing_ok=True)
    summary = {
        "source": args.base_url,
        "columns": ITEM_COLUMNS,
        "output": args.output.as_posix(),
        "rows_written": len(deduped),
        "rows_with_model_main": sum(
            bool(row.get("model_main")) for row in deduped
        ),
        "pages_fetched": pages_fetched,
        "page_size": args.page_size,
        "results_total": results_total,
        "finished_at": timestamp(),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def fetch_page(
    base_url: str,
    *,
    page: int,
    page_size: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    query = urlencode(
        {
            "columns": ",".join(ITEM_COLUMNS),
            "limit": page_size,
            "page": page,
        }
    )
    request = Request(
        f"{base_url}?{query}",
        headers={"User-Agent": "ffxiv-prism/0.1 same-model-cache"},
    )
    error: Exception | None = None
    for attempt in range(1, max(retries, 1) + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("CafeMaker response must be an object")
            return payload
        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    assert error is not None
    raise error


def normalize_item_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_item_id": normalize_int(row.get("ID")),
        "name_zh": normalize_text(row.get("Name")),
        "name_en": normalize_text(row.get("Name_en")),
        "model_main": parse_model(row.get("ModelMain")),
        "model_sub": parse_model(row.get("ModelSub")),
        "equip_slot_category_id": normalize_int(
            row.get("EquipSlotCategoryTargetID")
        ),
        "item_ui_category_id": normalize_int(
            row.get("ItemUICategoryTargetID")
        ),
        "source": DEFAULT_BASE_URL,
    }


def parse_model(value: Any) -> list[int]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        values: list[Any] = value.split(",")
    elif isinstance(value, (list, tuple)):
        values = list(value)
    elif isinstance(value, dict):
        values = [
            value.get(str(index), value.get(index))
            for index in range(4)
        ]
    else:
        values = [value]
    output = []
    for item in values:
        parsed = normalize_int(item)
        if parsed is None:
            return []
        output.append(parsed)
    return output


def normalize_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def dedupe_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        item_id = row.get("game_item_id")
        if not isinstance(item_id, int) or item_id in seen:
            continue
        seen.add(item_id)
        output.append(row)
    return output


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
