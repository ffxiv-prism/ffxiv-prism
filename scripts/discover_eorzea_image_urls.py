# 功能：从 Eorzea Collection 套装页建立已验证的静态图片 URL 清单，不下载图片正文。
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


DEFAULT_URLS = [
    "https://ffxiv.eorzeacollection.com/gearset/vanadielian-fending",
]


def split_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover and verify Eorzea Collection static image URLs.")
    parser.add_argument("urls", nargs="*", default=DEFAULT_URLS, help="Gearset detail URLs to inspect.")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between network requests in seconds.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Network timeout per request in seconds.")
    parser.add_argument(
        "--races",
        default=None,
        help="Comma-separated race slugs to test, e.g. hyur,viera. Defaults to races exposed on the page.",
    )
    parser.add_argument(
        "--genders",
        default=None,
        help="Comma-separated gender slugs to test, e.g. male,female. Defaults to genders exposed on the page.",
    )
    parser.add_argument(
        "--parts",
        default=None,
        help="Comma-separated image parts to test, e.g. front,back,head,body,hands,legs,feet.",
    )
    parser.add_argument(
        "--dye-states",
        default=None,
        help="Comma-separated dye states to test, e.g. normal,dyed.",
    )
    parser.add_argument(
        "--guess-missing-model-variants",
        action="store_true",
        help="Generate unlisted race/gender candidates. Verification keeps only reachable URLs.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Write generated candidates as accessible without checking them. Use only for offline debugging.",
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

    combined = None
    gearsets = []
    url_records = []
    for url in args.urls:
        gearset, records, summary = collector.collect_gearset_image_urls(
            url,
            verify=not args.skip_verify,
            guess_missing_model_variants=args.guess_missing_model_variants,
            races=split_csv(args.races),
            genders=split_csv(args.genders),
            parts=split_csv(args.parts),
            dye_states=split_csv(args.dye_states),
        )
        combined = summary if combined is None else combined.add(**summary.__dict__)
        gearsets.append(gearset)
        url_records.extend(records)

    print_summary(
        combined,
        {
            "gearset_slugs": [gearset.slug for gearset in gearsets],
            "accessible_url_record_count": len(url_records),
            "manifest_path": "data/raw/eorzea_collection/metadata/image_url_manifest_raw.jsonl",
            "check_log_path": "data/raw/eorzea_collection/metadata/image_url_checks_raw.jsonl",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
