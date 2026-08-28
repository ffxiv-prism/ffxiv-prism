# 功能：从已验证的静态图片 URL 清单按需下载图片并写入下载 manifest。
from __future__ import annotations

import argparse
import base64
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
from ffxiv_prism.collectors.eorzea_collection.models import (  # noqa: E402
    DownloadedImageRecord,
    ImageCandidate,
)
from ffxiv_prism.collectors.eorzea_collection.visible_browser import (  # noqa: E402
    DEFAULT_PROFILE_DIR,
    CdpChrome,
    close_visible_driver,
    create_visible_driver,
    fetch_verified_html,
)
from ffxiv_prism.jsonl import append_jsonl, read_jsonl  # noqa: E402
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
    parser.add_argument(
        "--browser-verification",
        action="store_true",
        help="Download cache misses through a visible normal Chrome session.",
    )
    parser.add_argument(
        "--browser-profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
    )
    parser.add_argument("--browser-wait-seconds", type=float, default=600.0)
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
    if args.browser_verification:
        downloaded_images = download_images_browser(
            collector,
            candidates,
            profile_dir=args.browser_profile_dir,
            wait_seconds=args.browser_wait_seconds,
            progress=progress,
        )
    else:
        downloaded_images = collector.download_images(candidates, progress=progress.update)
    progress.print(final=True)
    image_manifest_path = EORZEA_METADATA_DIR / "image_manifest_raw.jsonl"
    downloaded_rows = [image.to_dict() for image in downloaded_images]
    appended_rows = select_manifest_rows_to_append(
        read_jsonl(image_manifest_path),
        downloaded_rows,
    )
    append_jsonl(image_manifest_path, appended_rows)

    summary = CollectionSummary(
        images_downloaded=sum(1 for image in downloaded_images if not image.from_cache),
        images_from_cache=sum(1 for image in downloaded_images if image.from_cache),
        errors=progress.failed,
    )
    print_summary(
        summary,
        {
            "candidate_count": len(candidates),
            "image_record_count": len(downloaded_images),
            "manifest_rows_appended": len(appended_rows),
            "manifest_path": str(args.manifest),
        },
    )
    if required_images and args.limit is None:
        report = current_best_image_coverage(required_images)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 2
    return 0


def download_images_browser(
    collector: EorzeaCollectionCollector,
    candidates: list[ImageCandidate],
    *,
    profile_dir: Path,
    wait_seconds: float,
    progress: "ImageDownloadProgress",
) -> list[DownloadedImageRecord]:
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate.source_image_url in seen:
            continue
        seen.add(candidate.source_image_url)
        unique_candidates.append(candidate)

    records = []
    driver = None
    total = len(unique_candidates)
    try:
        for index, candidate in enumerate(unique_candidates, start=1):
            target = collector.image_target(candidate)
            record = None
            try:
                if target.is_file():
                    record = collector.downloaded_image_record(
                        candidate,
                        target,
                        True,
                    )
                else:
                    if driver is None:
                        driver = create_visible_driver(profile_dir)
                        fetch_verified_html(
                            driver,
                            candidate.source_image_url,
                            wait_seconds=wait_seconds,
                        )
                    payload = browser_fetch_image(driver, candidate.source_image_url)
                    if browser_payload_needs_verification(payload):
                        fetch_verified_html(
                            driver,
                            candidate.source_image_url,
                            wait_seconds=wait_seconds,
                        )
                        payload = browser_fetch_image(
                            driver,
                            candidate.source_image_url,
                        )
                    content_type = str(payload.get("content_type") or "")
                    status = payload.get("status")
                    if not isinstance(status, int) or not 200 <= status < 400:
                        raise RuntimeError(f"HTTP {status}: {content_type}")
                    if not content_type.startswith("image/"):
                        raise RuntimeError(
                            f"Expected image content, received {content_type or 'unknown'}"
                        )
                    encoded = payload.get("base64")
                    if not isinstance(encoded, str) or not encoded:
                        raise RuntimeError("Chrome returned an empty image payload")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    temporary.write_bytes(base64.b64decode(encoded))
                    temporary.replace(target)
                    record = collector.downloaded_image_record(
                        candidate,
                        target,
                        False,
                    )
            except Exception as exc:
                collector._log_error(
                    "download_image_browser",
                    candidate.source_image_url,
                    exc,
                )
            if record is not None:
                records.append(record)
            progress.update(index, total, record)
    finally:
        if driver is not None:
            close_visible_driver(driver)
    return records


def browser_fetch_image(driver: CdpChrome, url: str) -> dict[str, object]:
    encoded_url = json.dumps(url, ensure_ascii=True)
    expression = f"""
    (async () => {{
      try {{
        const response = await fetch({encoded_url}, {{
          method: 'GET', credentials: 'include', cache: 'no-store'
        }});
        const contentType = response.headers.get('content-type');
        if (!response.ok) {{
          return {{status: response.status, content_type: contentType, base64: null}};
        }}
        const bytes = new Uint8Array(await response.arrayBuffer());
        let binary = '';
        for (let index = 0; index < bytes.length; index += 32768) {{
          binary += String.fromCharCode(...bytes.subarray(index, index + 32768));
        }}
        return {{status: response.status, content_type: contentType, base64: btoa(binary)}};
      }} catch (error) {{
        return {{status: null, content_type: null, base64: null, error: String(error)}};
      }}
    }})()
    """
    payload = driver.evaluate(expression, await_promise=True)
    if not isinstance(payload, dict):
        raise RuntimeError("Chrome returned an invalid image payload")
    return payload


def browser_payload_needs_verification(payload: dict[str, object]) -> bool:
    return (
        payload.get("status") == 403
        and str(payload.get("content_type") or "").casefold().startswith("text/html")
    )


def select_manifest_rows_to_append(
    existing_rows: list[dict[str, object]],
    downloaded_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Avoid growing the append-only audit log with unchanged cache hits."""

    known_urls = {
        str(row.get("source_image_url"))
        for row in existing_rows
        if row.get("source_image_url")
    }
    return [
        row
        for row in downloaded_rows
        if not row.get("from_cache")
        or str(row.get("source_image_url") or "") not in known_urls
    ]


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
    from ffxiv_prism.matching.inference import current_best_configs

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
