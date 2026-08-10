# 功能：从已验证的静态图片 URL 清单按需下载图片并写入下载 manifest。
from __future__ import annotations

import argparse
import json
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
from ffxiv_prism.matching.inference import current_best_configs  # noqa: E402
from ffxiv_prism.paths import (  # noqa: E402
    EORZEA_METADATA_DIR,
    portable_path,
    resolve_project_path,
)


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
        "--current-best-only",
        action="store_true",
        help="Download only the equipment images referenced by current-best indexes.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report current-best image coverage without downloading.",
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

    rows = [
        row
        for row in read_jsonl(args.manifest)
        if row.get("accessible", True)
        and (row.get("image_variant") or row.get("dye_state") or "normal")
        in set(args.image_variants)
    ]
    if args.gearset_slug:
        rows = [row for row in rows if row.get("gearset_slug") == args.gearset_slug]

    required_images = (
        current_best_required_images() if args.current_best_only or args.check else {}
    )
    if required_images:
        rows_by_url = {
            str(row.get("source_image_url")): row
            for row in rows
            if row.get("source_image_url")
        }
        missing_manifest_urls = sorted(set(required_images) - set(rows_by_url))
        if missing_manifest_urls:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "current-best image URLs are missing from the download manifest",
                        "missing_url_count": len(missing_manifest_urls),
                        "examples": missing_manifest_urls[:10],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        rows = [rows_by_url[url] for url in sorted(required_images)]

    if args.check:
        report = current_best_image_coverage(required_images)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 2

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
    if required_images and args.limit is None:
        report = current_best_image_coverage(required_images)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 2
    return 0


def required_image_paths_from_metadata(
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    required: dict[str, Path] = {}
    for row in rows:
        url = str(row.get("source_image_url") or "")
        local_path = resolve_project_path(row.get("local_path"))
        if not url or local_path is None:
            continue
        existing = required.get(url)
        if existing is not None and existing != local_path:
            raise ValueError(
                f"One source image URL maps to multiple local paths: {url}"
            )
        required[url] = local_path
    return required


def current_best_required_images() -> dict[str, Path]:
    index_dirs = {
        path
        for config in current_best_configs().values()
        for path in (config.index_dir, *config.ensemble_index_dirs)
    }
    required: dict[str, Path] = {}
    for index_dir in sorted(index_dirs, key=lambda path: path.as_posix()):
        metadata_path = index_dir / "metadata.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Current-best index metadata is missing: {metadata_path}. "
                "Run `git lfs pull` first."
            )
        for url, path in required_image_paths_from_metadata(
            read_jsonl(metadata_path)
        ).items():
            existing = required.get(url)
            if existing is not None and existing != path:
                raise ValueError(
                    f"Current-best indexes disagree on the local path for {url}"
                )
            required[url] = path
    return required


def current_best_image_coverage(
    required_images: dict[str, Path],
) -> dict[str, object]:
    missing = [
        path
        for path in required_images.values()
        if not path.is_file()
    ]
    return {
        "ok": not missing,
        "required_image_count": len(required_images),
        "available_image_count": len(required_images) - len(missing),
        "missing_image_count": len(missing),
        "missing_examples": [
            portable_path(path)
            for path in missing[:10]
        ],
    }


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
