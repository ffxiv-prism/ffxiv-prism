# 功能：从少量 Eorzea Collection 套装 URL 开始采集本地图片和元数据。

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
    parser = argparse.ArgumentParser(description="Pilot collect Eorzea Collection gearset images.")
    parser.add_argument("urls", nargs="*", default=DEFAULT_URLS, help="Gearset detail URLs to collect.")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between network requests in seconds.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Network timeout per request in seconds.")
    parser.add_argument("--image-workers", type=int, default=1, help="Parallel image download workers.")
    parser.add_argument(
        "--races",
        default=None,
        help="Comma-separated fixed race slugs to collect, e.g. hyur,viera. Defaults to races exposed on the page.",
    )
    parser.add_argument(
        "--genders",
        default=None,
        help="Comma-separated fixed gender slugs to collect, e.g. male,female. Defaults to genders exposed on the page.",
    )
    parser.add_argument(
        "--include-piece-glamours",
        action="store_true",
        help="Deprecated: player glamour pages are not part of the canonical gear dataset.",
    )
    parser.add_argument(
        "--guess-missing-model-variants",
        action="store_true",
        help="Generate unlisted race/gender image URL candidates. This can be slow if many URLs do not exist.",
    )
    parser.add_argument(
        "--max-glamours-per-piece",
        type=int,
        default=3,
        help="Maximum glamour detail pages to collect per gear piece when variants are enabled.",
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
        image_workers=args.image_workers,
        respect_robots=args.respect_robots,
    )

    combined = None
    gearsets = []
    variants = []
    images = []
    for url in args.urls:
        gearset, glamour_variants, downloaded_images, summary = collector.collect_gearset(
            url,
            include_piece_glamours=args.include_piece_glamours,
            max_glamours_per_piece=args.max_glamours_per_piece,
            guess_missing_model_variants=args.guess_missing_model_variants,
            races=split_csv(args.races),
            genders=split_csv(args.genders),
        )
        combined = summary if combined is None else combined.add(**summary.__dict__)
        gearsets.append(gearset)
        variants.extend(glamour_variants)
        images.extend(downloaded_images)

    print_summary(
        combined,
        {
            "gearset_slugs": [gearset.slug for gearset in gearsets],
            "variant_count": len(variants),
            "image_record_count": len(images),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
