# 功能：从已验证的静态图片 URL 清单按需下载图片并写入下载 manifest。
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.collector import (  # noqa: E402
    EorzeaCollectionCollector,
    CollectionSummary,
    print_summary,
)
from ffxiv_prism.collectors.eorzea_collection.models import ImageCandidate  # noqa: E402
from ffxiv_prism.jsonl import append_jsonl, read_jsonl  # noqa: E402
from ffxiv_prism.paths import EORZEA_METADATA_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Download images from a verified Eorzea image URL manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EORZEA_METADATA_DIR / "image_url_manifest_raw.jsonl",
        help="Verified image URL manifest JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of manifest rows to download.")
    parser.add_argument("--gearset-slug", default=None, help="Only download rows for this gearset slug.")
    parser.add_argument(
        "--image-variants",
        nargs="+",
        choices=("normal", "dyed"),
        default=("normal",),
        help="Image variants to download. Defaults to normal only.",
    )
    parser.add_argument("--image-workers", type=int, default=1, help="Parallel image download workers.")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between network requests in seconds.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Network timeout per request in seconds.")
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

    rows = [
        row
        for row in read_jsonl(args.manifest)
        if row.get("accessible", True)
        and (row.get("image_variant") or row.get("dye_state") or "normal")
        in set(args.image_variants)
    ]
    if args.gearset_slug:
        rows = [row for row in rows if row.get("gearset_slug") == args.gearset_slug]
    if args.limit is not None:
        rows = rows[: args.limit]

    fields = ImageCandidate.__dataclass_fields__.keys()
    candidates = []
    for row in rows:
        if not row.get("source_image_url") or not row.get("source_page_url") or not row.get("image_kind"):
            continue
        payload = {field: row.get(field) for field in fields}
        payload["dye_state"] = (
            row.get("image_variant") or row.get("dye_state") or "normal"
        )
        candidates.append(ImageCandidate(**payload))

    collector = EorzeaCollectionCollector(
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        image_workers=args.image_workers,
        respect_robots=args.respect_robots,
    )
    progress = ImageDownloadProgress(total=len(candidates))
    progress.print(final=not candidates)
    downloaded_images = collector.download_images(candidates, progress=progress.update)
    progress.print(final=True)
    append_jsonl(EORZEA_METADATA_DIR / "image_manifest_raw.jsonl", [image.to_dict() for image in downloaded_images])

    summary = CollectionSummary(
        images_downloaded=sum(1 for image in downloaded_images if not image.from_cache),
        images_from_cache=sum(1 for image in downloaded_images if image.from_cache),
    )
    print_summary(
        summary,
        {
            "candidate_count": len(candidates),
            "image_record_count": len(downloaded_images),
            "manifest_path": str(args.manifest),
        },
    )
    return 0


class ImageDownloadProgress:
    def __init__(self, total: int, interval_seconds: float = 5.0) -> None:
        self.total = total
        self.interval_seconds = interval_seconds
        self.started_at = time.monotonic()
        self.last_print_at = 0.0
        self.downloaded = 0
        self.from_cache = 0
        self.failed = 0
        self.current = 0
        self.final_printed = False

    def update(self, current: int, total: int, record: object | None) -> None:
        self.current = current
        self.total = total
        if record is None:
            self.failed += 1
        elif getattr(record, "from_cache", False):
            self.from_cache += 1
        else:
            self.downloaded += 1

        now = time.monotonic()
        if now - self.last_print_at >= self.interval_seconds or current >= total:
            self.print(final=current >= total)
            self.last_print_at = now

    def print(self, final: bool) -> None:
        if final and self.final_printed:
            return
        elapsed = max(time.monotonic() - self.started_at, 0.001)
        current = min(self.current, self.total)
        rate = current / elapsed if current else 0.0
        remaining = max(self.total - current, 0)
        eta = format_duration(remaining / rate) if rate > 0 else "unknown"
        percent = (current / self.total * 100.0) if self.total else 100.0
        line = (
            f"Progress: {current}/{self.total} ({percent:5.1f}%) "
            f"downloaded={self.downloaded} cached={self.from_cache} failed={self.failed} "
            f"rate={rate:5.1f}/s eta={eta}"
        )
        if final:
            print(f"\r{line}", file=sys.stderr, flush=True)
            self.final_printed = True
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


if __name__ == "__main__":
    raise SystemExit(main())
