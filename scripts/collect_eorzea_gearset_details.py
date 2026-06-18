# 功能：采集 gearset 详情页中的 Equipment 物品、真实槽位和多槽占用约束。
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.collector import EorzeaCollectionCollector, timestamp  # noqa: E402
from ffxiv_prism.collectors.eorzea_collection.parser import parse_gearset_detail  # noqa: E402
from ffxiv_prism.collectors.eorzea_collection.url_utils import absolute_url, extract_slug  # noqa: E402
from ffxiv_prism.jsonl import append_jsonl, read_jsonl  # noqa: E402
from ffxiv_prism.paths import EORZEA_METADATA_DIR, INTERIM_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Eorzea gearset equipment metadata.")
    parser.add_argument(
        "--queue",
        type=Path,
        default=INTERIM_DIR / "gearset_detail_queue.jsonl",
        help="JSONL queue containing source_page_url rows.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for debugging.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel detail page workers.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between page requests.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Network timeout per request.")
    parser.add_argument("--resume", action="store_true", help="Skip gearsets already present in output.")
    parser.add_argument("--include-images", action="store_true", help="Keep generated image candidates in details output.")
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Clear output files before running. Implied when --resume is not used.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between progress updates.",
    )
    parser.add_argument(
        "--respect-robots",
        action="store_true",
        help="Respect robots.txt. This project defaults to ignoring it after user approval.",
    )
    args = parser.parse_args()

    queue_rows = read_jsonl(args.queue)
    urls = [absolute_url(str(row["source_page_url"])) for row in queue_rows if row.get("source_page_url")]
    urls = unique(urls)
    if args.limit is not None:
        urls = urls[: args.limit]

    details_path = EORZEA_METADATA_DIR / "gearset_details_raw.jsonl"
    pieces_path = EORZEA_METADATA_DIR / "gear_pieces_raw.jsonl"
    EORZEA_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset_output or not args.resume:
        details_path.write_text("", encoding="utf-8")
        pieces_path.write_text("", encoding="utf-8")

    existing_slugs = set()
    if args.resume:
        existing_slugs = {str(row.get("slug")) for row in read_jsonl(details_path) if row.get("slug")}
    pending_urls = [url for url in urls if extract_slug(url, "gearset") not in existing_slugs]

    started_at = time.monotonic()
    parsed_count = 0
    error_count = 0
    last_progress_at = 0.0
    print(
        f"Starting gearset detail collection: total={len(urls)}, already={len(existing_slugs)}, pending={len(pending_urls)}",
        file=sys.stderr,
        flush=True,
    )
    print_progress(len(existing_slugs), len(urls), parsed_count, error_count, started_at, final=False)

    if args.workers <= 1:
        for url in pending_urls:
            result = collect_one(url, args.timeout, args.respect_robots, args.include_images)
            parsed_count, error_count = write_result(result, details_path, pieces_path, parsed_count, error_count)
            now = time.monotonic()
            if args.progress_interval > 0 and now - last_progress_at >= args.progress_interval:
                print_progress(len(existing_slugs) + parsed_count, len(urls), parsed_count, error_count, started_at, final=False)
                last_progress_at = now
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(collect_one, url, args.timeout, args.respect_robots, args.include_images): url
                for url in pending_urls
            }
            for future in concurrent.futures.as_completed(futures):
                parsed_count, error_count = write_result(
                    future.result(),
                    details_path,
                    pieces_path,
                    parsed_count,
                    error_count,
                )
                now = time.monotonic()
                if args.progress_interval > 0 and now - last_progress_at >= args.progress_interval:
                    print_progress(len(existing_slugs) + parsed_count, len(urls), parsed_count, error_count, started_at, final=False)
                    last_progress_at = now

    print_progress(len(existing_slugs) + parsed_count, len(urls), parsed_count, error_count, started_at, final=True)
    print(
        json.dumps(
            {
                "gearset_total": len(urls),
                "already_collected": len(existing_slugs),
                "newly_collected": parsed_count,
                "errors": error_count,
                "details_path": str(details_path),
                "pieces_path": str(pieces_path),
                "finished_at": timestamp(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def collect_one(
    url: str,
    timeout: float,
    respect_robots: bool,
    include_images: bool,
) -> dict[str, object]:
    slug = extract_slug(url, "gearset")
    try:
        collector = EorzeaCollectionCollector(
            delay_seconds=0.0,
            timeout_seconds=timeout,
            respect_robots=respect_robots,
        )
        html, _from_cache = collector._fetch_html(url, "gearset", slug)
        gearset = parse_gearset_detail(html, url)
        gearset_payload = gearset.to_dict()
        if not include_images:
            gearset_payload["images"] = []
        return {"ok": True, "gearset": gearset_payload, "url": url}
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def write_result(
    result: dict[str, object],
    details_path: Path,
    pieces_path: Path,
    parsed_count: int,
    error_count: int,
) -> tuple[int, int]:
    if result.get("ok"):
        gearset = result["gearset"]
        if not isinstance(gearset, dict):
            raise TypeError("gearset result must be a dict")
        append_jsonl(details_path, [gearset])
        append_jsonl(pieces_path, gearset_piece_rows(gearset))
        return parsed_count + 1, error_count

    append_jsonl(
        EORZEA_METADATA_DIR / "errors.jsonl",
        [
            {
                "stage": "collect_gearset_details",
                "url": result.get("url"),
                "error": result.get("error"),
                "error_type": result.get("error_type"),
                "created_at": timestamp(),
            }
        ],
    )
    return parsed_count, error_count + 1


def gearset_piece_rows(gearset: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for piece in gearset.get("pieces", []):
        if not isinstance(piece, dict):
            continue
        row = {
            "gearset_slug": gearset.get("slug"),
            "source_page_url": gearset.get("source_page_url"),
            "title": gearset.get("title"),
            "present_slots": gearset.get("present_slots"),
            "missing_slots": gearset.get("missing_slots"),
            **piece,
        }
        rows.append(row)
    return rows


def print_progress(current: int, total: int, newly_collected: int, errors: int, started_at: float, final: bool) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = newly_collected / elapsed if newly_collected > 0 else 0.0
    remaining = max(total - current, 0)
    eta = format_duration(remaining / rate) if rate > 0 else "unknown"
    percent = (current / total * 100.0) if total else 100.0
    line = (
        f"Progress: {current}/{total} ({percent:5.1f}%) "
        f"new={newly_collected} errors={errors} rate={rate:5.1f}/s eta={eta}"
    )
    if final:
        print(f"\r{line}", file=sys.stderr, flush=True)
    else:
        print(f"\r{line}", end="", file=sys.stderr, flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
