# 功能：采集 Eorzea Collection gearsets 列表页，生成后续详情页下载队列。

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.collector import (  # noqa: E402
    EorzeaCollectionCollector,
    print_summary,
)
from ffxiv_prism.jsonl import append_jsonl  # noqa: E402
from ffxiv_prism.paths import INTERIM_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover Eorzea Collection gearset detail URLs.")
    parser.add_argument(
        "--start-url",
        default="https://ffxiv.eorzeacollection.com/gearsets",
        help="Gearset list URL to start from.",
    )
    parser.add_argument("--max-pages", type=int, default=1, help="Maximum list pages to crawl.")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between network requests in seconds.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Network timeout per request in seconds.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch fresh list pages even when an HTML cache entry exists.",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        default=True,
        help="Skip robots.txt checks. Enabled by default for this project after user approval.",
    )
    parser.add_argument(
        "--respect-robots",
        action="store_true",
        help="Respect robots.txt checks for this run.",
    )
    args = parser.parse_args()

    collector = EorzeaCollectionCollector(
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        respect_robots=args.respect_robots,
    )
    gearset_urls, summary = collector.discover_gearsets(
        args.start_url,
        max_pages=args.max_pages,
        force_refresh=args.refresh,
    )
    append_jsonl(
        INTERIM_DIR / "gearset_detail_queue.jsonl",
        [{"source_page_url": url} for url in gearset_urls],
    )
    print_summary(summary, {"gearset_url_count": len(gearset_urls)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
