# 功能：按 male/female 列表流顺序发现 Eorzea Collection gearset 详情页，避免跟进无关筛选分页。
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.collector import (  # noqa: E402
    CollectionSummary,
    EorzeaCollectionCollector,
    print_summary,
    timestamp,
)
from ffxiv_prism.collectors.eorzea_collection.parser import extract_gearset_links  # noqa: E402
from ffxiv_prism.jsonl import append_jsonl  # noqa: E402
from ffxiv_prism.paths import INTERIM_DIR  # noqa: E402


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
    args = parser.parse_args()

    collector = EorzeaCollectionCollector(
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        respect_robots=args.respect_robots,
    )

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    page_count = 0
    empty_pages = 0
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    for model in models:
        model_empty_pages = 0
        for page in range(1, args.max_pages + 1):
            url = f"https://ffxiv.eorzeacollection.com/gearsets?page={page}&model={model}"
            html, from_cache = collector._fetch_html(
                url,
                "gearset-list-by-model",
                f"{model}-{page}",
                force_refresh=args.refresh,
            )
            page_count += 1
            links = extract_gearset_links(html, url)
            if not links:
                model_empty_pages += 1
                empty_pages += 1
                if model_empty_pages >= args.empty_stop:
                    break
                continue
            model_empty_pages = 0
            for link in links:
                if link in seen:
                    continue
                seen.add(link)
                rows.append(
                    {
                        "source_page_url": link,
                        "discovered_at": timestamp(),
                        "source_list_url": url,
                        "source_model_filter": model,
                        "source_from_cache": str(from_cache).lower(),
                    }
                )

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    queue_path = INTERIM_DIR / "gearset_detail_queue.jsonl"
    queue_path.write_text("", encoding="utf-8")
    append_jsonl(queue_path, [{"source_page_url": row["source_page_url"]} for row in rows])

    raw_path = PROJECT_ROOT / "data" / "raw" / "eorzea_collection" / "metadata" / "gearset_discovery_by_model_raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    append_jsonl(raw_path, rows)

    print_summary(
        CollectionSummary(pages_fetched=page_count - empty_pages, pages_from_cache=0),
        {
            "gearset_url_count": len(rows),
            "list_pages_checked": page_count,
            "empty_pages": empty_pages,
            "models": models,
            "queue_path": str(queue_path),
            "raw_path": str(raw_path),
            "finished_at": timestamp(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
