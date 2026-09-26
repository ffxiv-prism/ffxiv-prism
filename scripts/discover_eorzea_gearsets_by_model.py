# 功能：按 male/female 列表流顺序发现 Eorzea Collection gearset 详情页，避免跟进无关筛选分页。
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.collector import (  # noqa: E402
    EorzeaCollectionCollector,
    timestamp,
)
from ffxiv_prism.collectors.eorzea_collection.visible_browser import (  # noqa: E402
    DEFAULT_PROFILE_DIR,
    close_visible_driver,
    create_visible_driver,
    fetch_verified_html,
)
from ffxiv_prism.jsonl import read_jsonl, write_jsonl_atomic, write_json_atomic  # noqa: E402
from ffxiv_prism.paths import EORZEA_HTML_DIR, INTERIM_DIR
from ffxiv_prism.collectors.eorzea_collection.discovery import scan_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover gearset URLs from male/female Eorzea list pages.")
    parser.add_argument("--max-pages", type=int, default=500, help="Maximum pages to scan per model.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between network requests in seconds.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Network timeout per request in seconds.")
    parser.add_argument("--empty-stop", type=int, default=2, help="Stop after this many consecutive empty pages.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch fresh list pages even when an HTML cache entry exists.",
    )
    parser.add_argument(
        "--models",
        default="male,female",
        help="Comma-separated model filters to scan. Defaults to male,female.",
    )
    parser.add_argument(
        "--respect-robots",
        action="store_true",
        help="Respect robots.txt. This project defaults to ignoring it after user approval.",
    )
    parser.add_argument(
        "--browser-verification",
        action="store_true",
        help=(
            "Use a visible persistent Chrome for uncached/refreshed pages and "
            "wait for the user to complete Cloudflare verification."
        ),
    )
    parser.add_argument(
        "--browser-profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help="Persistent Chrome profile used by --browser-verification.",
    )
    parser.add_argument(
        "--browser-wait-seconds",
        type=float,
        default=300.0,
        help="Maximum time to wait for manual browser verification per page.",
    )
    parser.add_argument("--discovery-mode", choices=("auto", "incremental", "full"), default="auto",
                        help="Auto initializes an unknown library fully, otherwise uses a bounded incremental scan.")
    parser.add_argument("--head-pages", type=int, default=3)
    parser.add_argument("--max-head-pages", type=int, default=8)
    parser.add_argument("--history-pages", type=int, default=3)
    parser.add_argument("--known-page-stop", type=int, default=2)
    parser.add_argument("--state-file", type=Path, default=PROJECT_ROOT / "var/cache/eorzea_discovery/state.json")
    args = parser.parse_args()
    if not (1 <= args.head_pages <= args.max_head_pages <= args.max_pages):
        parser.error("Require 1 <= head-pages <= max-head-pages <= max-pages")
    if min(args.history_pages, args.known_page_stop, args.empty_stop) < 1:
        parser.error("history-pages, known-page-stop and empty-stop must be positive")

    collector = EorzeaCollectionCollector(
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        respect_robots=args.respect_robots,
    )

    metadata = PROJECT_ROOT / "data/raw/eorzea_collection/metadata"
    raw_path = metadata / "gearset_discovery_by_model_raw.jsonl"
    queue_path = INTERIM_DIR / "gearset_detail_queue.jsonl"
    state = json.loads(args.state_file.read_text(encoding="utf-8")) if args.state_file.is_file() else {"version": 1, "models": {}}
    if state.get("version") != 1:
        raise ValueError("Unsupported discovery state version")
    records = {row["source_page_url"]: row for row in read_jsonl(raw_path) if row.get("source_page_url")}
    # Existing snapshots count as a known library even if page caches/state were
    # removed or this is a fresh checkout on a second machine.
    for filename in ("gearset_details_raw.jsonl", "gearset_discovery_raw.jsonl"):
        for row in read_jsonl(metadata / filename):
            if url := row.get("source_page_url"):
                records.setdefault(url, {"source_page_url": url})
    for row in read_jsonl(queue_path):
        if url := row.get("source_page_url"):
            records.setdefault(url, {"source_page_url": url})
    database = PROJECT_ROOT / "data/processed/eorzea.sqlite"
    if not records and database.is_file():
        with closing(sqlite3.connect(database)) as conn:
            for (url,) in conn.execute("SELECT source_page_url FROM gearsets WHERE source_page_url IS NOT NULL"):
                records.setdefault(url, {"source_page_url": url})
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models or any(model not in {"male", "female"} for model in models):
        parser.error("models must contain male and/or female")
    known_links = set(records)
    known_seeds = {row["source_image_url"] for row in read_jsonl(metadata / "image_url_checks_raw.jsonl")
                   if row.get("accessible") and row.get("source_image_url")}
    mode = args.discovery_mode
    initializing = not known_links or bool(state.get("initialization_pending"))
    full = mode == "full" or (mode == "auto" and initializing)
    if mode == "incremental" and initializing:
        full = True
    if full:
        if not state.get("full_pending"):
            state["full_pending"] = list(models)
            for model in models:
                state.setdefault("models", {})[model] = {"full_next_page": 1}
        if initializing:
            state["initialization_pending"] = True
        write_json_atomic(args.state_file, state)
    driver = None
    reports = {}
    baseline_count = len(records)
    try:
        for model in models:
            if full and model not in state.get("full_pending", []):
                continue
            model_state = state.setdefault("models", {}).setdefault(model, {})
            if full and model_state.get("full_complete"):
                state["full_pending"].remove(model)
                continue

            def fetch(page: int) -> tuple[str, bool]:
                nonlocal driver
                url = f"https://ffxiv.eorzeacollection.com/gearsets?page={page}&model={model}"
                cache_path = EORZEA_HTML_DIR / "gearset-list-by-model" / f"{model}-{page}.html"
                if args.browser_verification and (args.refresh or not cache_path.is_file()):
                    if driver is None:
                        driver = create_visible_driver(args.browser_profile_dir, manual_handoff_url=url)
                    html, _ = fetch_verified_html(driver, url, wait_seconds=args.browser_wait_seconds)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(html, encoding="utf-8")
                    return html, False
                return collector._fetch_html(url, "gearset-list-by-model", f"{model}-{page}", force_refresh=args.refresh)

            def checkpoint(page: int, parsed: dict, from_cache: bool) -> None:
                for url in parsed["links"]:
                    previous = records.get(url, {})
                    records[url] = {**previous, "source_page_url": url,
                        "discovered_at": previous.get("discovered_at") or timestamp(),
                        "last_seen_at": timestamp(),
                        "source_list_url": f"https://ffxiv.eorzeacollection.com/gearsets?page={page}&model={model}",
                        "source_model_filter": model, "source_from_cache": str(from_cache).lower()}
                # Merge, never replace the historical catalogue with the small
                # scanned prefix. Failed detail jobs remain in the queue.
                write_jsonl_atomic(raw_path, records.values())
                write_jsonl_atomic(queue_path, [{"source_page_url": url} for url in records])
                write_json_atomic(args.state_file, state)
                print(f"discovery model={model} page={page} mode={'full' if full else 'incremental'} known={len(records)}", flush=True)

            reports[model] = scan_model(model=model, state=model_state,
                known_links=known_links, known_seeds=known_seeds, fetch=fetch, checkpoint=checkpoint,
                full=full, max_pages=args.max_pages, head_pages=args.head_pages,
                max_head_pages=args.max_head_pages, known_stop=args.known_page_stop,
                tail_pages=args.history_pages, empty_stop=args.empty_stop)
            if full and reports[model]["full_complete"]:
                state["full_pending"].remove(model)
            write_json_atomic(args.state_file, state)
    finally:
        if driver is not None:
            close_visible_driver(driver)
    if full and not state.get("full_pending"):
        state["initialization_pending"] = False
        write_json_atomic(args.state_file, state)
    summary = {"mode": "full" if full else "incremental", "models": reports,
        "new_gearsets": len(records) - baseline_count, "known_gearsets": len(records),
        "list_pages_checked": sum(len(r["pages_checked"]) for r in reports.values()),
        "remote_pages": sum(r["remote_pages"] for r in reports.values()),
        "cached_pages": sum(r["cached_pages"] for r in reports.values()),
        "initialization_pending": bool(state.get("initialization_pending")),
        "full_scan_pending": state.get("full_pending", []) if full else [],
        "finished_at": timestamp()}
    write_json_atomic(args.state_file.with_name("last_scan_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if full and state.get("full_pending") else 0


def unseen_links(links: list[str], seen: set[str]) -> list[str]:
    """Return page links not observed earlier in the same model stream."""

    return [link for link in links if link not in seen]


if __name__ == "__main__":
    raise SystemExit(main())
