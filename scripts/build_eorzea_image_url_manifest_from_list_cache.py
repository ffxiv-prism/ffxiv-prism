# 功能：从 gearset 列表页缓存中的真实静态图片 seed 生成全量候选 URL，并验证可访问链接。
from __future__ import annotations

import argparse
import concurrent.futures
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.image_url import (  # noqa: E402
    DYE_STATES,
    GEAR_IMAGE_PARTS,
    GEARSETS_IMAGE_HOST,
    GENDER_SLUGS,
    RACE_SLUGS,
    generate_static_image_urls,
    parse_gearsets_image_url,
)
from ffxiv_prism.collectors.eorzea_collection.parser import parse_html  # noqa: E402
from ffxiv_prism.collectors.eorzea_collection.collector import timestamp  # noqa: E402
from ffxiv_prism.collectors.eorzea_collection.visible_browser import (  # noqa: E402
    DEFAULT_PROFILE_DIR,
    CdpChrome,
    close_visible_driver,
    create_visible_driver,
    fetch_verified_html,
)
from ffxiv_prism.jsonl import append_jsonl, read_jsonl, write_jsonl_atomic  # noqa: E402
from ffxiv_prism.paths import EORZEA_HTML_DIR, EORZEA_METADATA_DIR  # noqa: E402


USER_AGENT = "ffxiv-prism-local-dataset/0.1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify image URL manifest from cached gearset list pages.")
    parser.add_argument(
        "--list-cache-dir",
        type=Path,
        default=EORZEA_HTML_DIR / "gearset-list-by-model",
        help="Directory containing cached gearset list HTML pages.",
    )
    parser.add_argument("--workers", type=int, default=48, help="Parallel URL check workers.")
    parser.add_argument("--timeout", type=float, default=12.0, help="Timeout per URL check.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows to flush per JSONL batch.")
    parser.add_argument("--limit", type=int, default=None, help="Optional candidate limit for debugging.")
    parser.add_argument("--resume", action="store_true", help="Skip URLs already present in image_url_checks_raw.jsonl.")
    parser.add_argument("--retry-unavailable", action="store_true", help="Recheck previously unavailable images while retaining successful checks.")
    parser.add_argument("--retry-limit", type=int, default=200, help="Maximum historical failed URLs retried per run.")
    parser.add_argument("--missing-retry-days", type=float, default=7, help="Cooldown for historical HTTP 404/410 results.")
    parser.add_argument("--error-retry-hours", type=float, default=6, help="Cooldown for transient failures.")
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Clear checks and manifest outputs before running. Implied when --resume is not used.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between progress updates. Use 0 to print only batch/final summaries.",
    )
    parser.add_argument(
        "--model-scope",
        choices=("exposed", "all"),
        default="exposed",
        help="Use race/gender pairs exposed in list seeds, or generate every known race/gender pair.",
    )
    parser.add_argument("--races", default=",".join(RACE_SLUGS), help="Comma-separated race slugs to generate.")
    parser.add_argument("--genders", default=",".join(GENDER_SLUGS), help="Comma-separated gender slugs to generate.")
    parser.add_argument("--parts", default=",".join(GEAR_IMAGE_PARTS), help="Comma-separated image parts to generate.")
    parser.add_argument("--dye-states", default=",".join(DYE_STATES), help="Comma-separated dye states to generate.")
    parser.add_argument(
        "--browser-verification",
        action="store_true",
        help="Verify pending static image URLs through a visible normal Chrome session.",
    )
    parser.add_argument(
        "--browser-profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
    )
    parser.add_argument("--browser-wait-seconds", type=float, default=600.0)
    parser.add_argument("--browser-batch-size", type=int, default=64)
    args = parser.parse_args()
    if min(args.retry_limit, args.missing_retry_days, args.error_retry_hours) < 0:
        parser.error("Retry limits and cooldowns must be nonnegative")

    checks_path = EORZEA_METADATA_DIR / "image_url_checks_raw.jsonl"
    manifest_path = EORZEA_METADATA_DIR / "image_url_manifest_raw.jsonl"
    EORZEA_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    use_cached_checks = args.resume and not args.reset_output
    cached_check_rows = read_jsonl(checks_path) if use_cached_checks else []
    seed_urls = discover_seed_urls(args.list_cache_dir)
    candidate_urls = generate_candidate_urls(
        seed_urls,
        model_scope=args.model_scope,
        races=split_csv(args.races),
        genders=split_csv(args.genders),
        parts=split_csv(args.parts),
        dye_states=split_csv(args.dye_states),
    )
    # A bounded list refresh must not hide old URL families or their retry backlog.
    candidate_urls = sorted(set(candidate_urls) | set(candidate_urls_from_checks(cached_check_rows)))
    if args.limit is not None:
        candidate_urls = candidate_urls[: args.limit]
    if not candidate_urls:
        raise RuntimeError(
            "No image URL candidates were found. Refusing to replace the existing manifest. "
            f"list_cache_dir={args.list_cache_dir}, seed_urls={len(seed_urls)}, "
            f"cached_checks={len(cached_check_rows)}"
        )

    if args.reset_output or not args.resume:
        checks_path.write_text("", encoding="utf-8")

    candidate_url_set = set(candidate_urls)
    cached_rows_by_url: dict[str, dict[str, object]] = {}
    if use_cached_checks:
        for row in cached_check_rows:
            url = row.get("source_image_url")
            if not url or not parse_gearsets_image_url(str(url)):
                continue
            cached_rows_by_url[str(url)] = row
    retry_urls = select_retry_urls(
        {url: row for url, row in cached_rows_by_url.items() if url in candidate_url_set},
        enabled=args.retry_unavailable, browser=args.browser_verification,
        limit=args.retry_limit, missing_days=args.missing_retry_days,
        error_hours=args.error_retry_hours,
    )
    deferred_retry_count = sum(not row.get("accessible") for url, row in cached_rows_by_url.items()
                               if url in candidate_url_set) - len(retry_urls)
    checked_rows_by_url: dict[str, dict[str, object]] = {}
    if use_cached_checks:
        for url, row in cached_rows_by_url.items():
            if url not in candidate_url_set:
                continue
            if url in retry_urls:
                continue
            checked_rows_by_url[url] = row
    checked_urls = set(checked_rows_by_url)
    pending_urls = [url for url in candidate_urls if url not in checked_urls]

    checked_count = 0
    already_accessible_count = sum(1 for row in checked_rows_by_url.values() if row.get("accessible"))
    already_unavailable_count = len(checked_rows_by_url) - already_accessible_count
    accessible_count = already_accessible_count
    unavailable_count = already_unavailable_count
    batch: list[dict[str, object]] = []
    new_rows_by_url: dict[str, dict[str, object]] = {}
    started_at = time.monotonic()
    total_count = len(candidate_urls)
    already_checked_count = len(checked_urls)
    last_progress_at = 0.0

    print(
        "Starting URL verification: "
        f"scope={args.model_scope}, candidates={total_count}, "
        f"already_checked={already_checked_count}, pending={len(pending_urls)}, "
        f"historical_retries={len(retry_urls)}, deferred_retries={deferred_retry_count}, "
        f"workers={max(1, args.workers)}",
        file=sys.stderr,
        flush=True,
    )
    print_progress(
        current=already_checked_count,
        total=total_count,
        newly_checked=0,
        accessible=accessible_count,
        unavailable=unavailable_count,
        started_at=started_at,
        final=False,
    )

    def register_check(check: dict[str, object]) -> None:
        nonlocal checked_count, accessible_count, unavailable_count, last_progress_at
        row = image_url_row(check)
        batch.append(row)
        new_rows_by_url[str(row["source_image_url"])] = row
        checked_count += 1
        if row["accessible"]:
            accessible_count += 1
        else:
            unavailable_count += 1
        if len(batch) >= args.batch_size:
            append_jsonl(checks_path, batch)
            batch.clear()
            print_progress(
                current=already_checked_count + checked_count,
                total=total_count,
                newly_checked=checked_count,
                accessible=accessible_count,
                unavailable=unavailable_count,
                started_at=started_at,
                final=False,
            )
            last_progress_at = time.monotonic()
        elif args.progress_interval > 0:
            now = time.monotonic()
            if now - last_progress_at >= args.progress_interval:
                print_progress(
                    current=already_checked_count + checked_count,
                    total=total_count,
                    newly_checked=checked_count,
                    accessible=accessible_count,
                    unavailable=unavailable_count,
                    started_at=started_at,
                    final=False,
                )
                last_progress_at = now

    if args.browser_verification:
        for check in check_image_urls_browser(
            pending_urls,
            profile_dir=args.browser_profile_dir,
            wait_seconds=args.browser_wait_seconds,
            batch_size=args.browser_batch_size,
        ):
            register_check(check)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(check_image_url, url, args.timeout): url for url in pending_urls}
            for future in concurrent.futures.as_completed(futures):
                register_check(future.result())

    if batch:
        append_jsonl(checks_path, batch)

    elapsed_seconds = time.monotonic() - started_at
    checked_rows_by_url.update(new_rows_by_url)
    all_checked_rows_by_url = dict(cached_rows_by_url)
    all_checked_rows_by_url.update(checked_rows_by_url)
    manifest_scope_urls = retained_manifest_urls(
        candidate_url_set,
        cached_rows_by_url.values(),
        preserve_cached=use_cached_checks,
    )
    manifest_rows = build_manifest_rows(
        all_checked_rows_by_url.values(),
        manifest_scope_urls,
    )
    if not manifest_rows:
        raise RuntimeError(
            "No accessible normal image URLs were found. "
            f"Refusing to replace the existing manifest: {manifest_path}"
        )
    write_jsonl_atomic(manifest_path, manifest_rows)
    print_progress(
        current=already_checked_count + checked_count,
        total=total_count,
        newly_checked=checked_count,
        accessible=accessible_count,
        unavailable=unavailable_count,
        started_at=started_at,
        final=True,
    )
    print(
        json.dumps(
            {
                "seed_url_count": len(seed_urls),
                "model_scope": args.model_scope,
                "candidate_url_count": len(candidate_urls),
                "already_checked_count": already_checked_count,
                "newly_checked_count": checked_count,
                "historical_retries": len(retry_urls),
                "deferred_historical_retries": deferred_retry_count,
                "new_url_count": sum(url not in cached_rows_by_url for url in pending_urls),
                "accessible_url_count": accessible_count,
                "unavailable_url_count": unavailable_count,
                "manifest_url_count": len(manifest_rows),
                "elapsed_seconds": round(elapsed_seconds, 2),
                "checks_path": str(checks_path),
                "manifest_path": str(manifest_path),
                "finished_at": timestamp(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def select_retry_urls(rows: dict[str, dict], *, enabled: bool, browser: bool = False,
                      limit: int = 200, missing_days: float = 7, error_hours: float = 6,
                      now: datetime | None = None) -> set[str]:
    """Retry the oldest eligible failures; successful URLs never consume requests."""
    now = now or datetime.now(UTC)
    eligible = []
    for url, row in rows.items():
        if row.get("accessible") or not (enabled or (browser and browser_retryable_check(row))):
            continue
        try:
            checked = datetime.fromisoformat(str(row.get("checked_at") or "").replace("Z", "+00:00"))
            if checked.tzinfo is None:
                checked = checked.replace(tzinfo=UTC)
        except ValueError:
            checked = datetime.min.replace(tzinfo=UTC)
        cooldown = timedelta(days=missing_days) if row.get("status") in {404, 410} else timedelta(hours=error_hours)
        if now - checked >= cooldown:
            eligible.append((checked, url))
    return {url for _, url in sorted(eligible)[:limit]}


def discover_seed_urls(cache_dir: Path) -> list[str]:
    seed_urls: list[str] = []
    seen: set[str] = set()
    for path in sorted(cache_dir.glob("*.html")):
        html = path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_html(html, "https://ffxiv.eorzeacollection.com/gearsets")
        for image in parsed.images:
            for url in image.urls:
                meta = parse_gearsets_image_url(url)
                if not meta or url in seen:
                    continue
                seed_urls.append(url)
                seen.add(url)
    return seed_urls


def candidate_urls_from_checks(rows: Iterable[dict[str, object]]) -> list[str]:
    urls = {
        str(row.get("source_image_url"))
        for row in rows
        if isinstance(row, dict)
        and row.get("source_image_url")
        and parse_gearsets_image_url(str(row.get("source_image_url")))
    }
    return sorted(urls)


def retained_manifest_urls(
    candidate_urls: set[str],
    cached_rows: Iterable[dict[str, object]],
    *,
    preserve_cached: bool,
) -> set[str]:
    """Keep verified history during a resumable refresh.

    List pages are a discovery source, not an authoritative deletion signal. A
    gearset can temporarily disappear from pagination while its static image is
    still used by a published candidate index. In resume mode the manifest must
    therefore be monotonic with respect to previously checked, parseable URLs.
    The latest cached/new check row still decides whether each URL is accessible.
    """

    urls = set(candidate_urls)
    if not preserve_cached:
        return urls
    urls.update(
        str(row.get("source_image_url"))
        for row in cached_rows
        if isinstance(row, dict)
        and row.get("source_image_url")
        and parse_gearsets_image_url(str(row.get("source_image_url")))
    )
    return urls


def generate_candidate_urls(
    seed_urls: list[str],
    model_scope: str,
    races: tuple[str, ...],
    genders: tuple[str, ...],
    parts: tuple[str, ...],
    dye_states: tuple[str, ...],
) -> list[str]:
    if model_scope == "all":
        return generate_static_image_urls(
            seed_urls,
            races=races,
            genders=genders,
            parts=parts,
            dye_states=dye_states,
        )

    folder_to_seed: dict[str, str] = {}
    folder_to_pairs: dict[str, set[tuple[str, str]]] = {}
    for url in seed_urls:
        meta = parse_gearsets_image_url(url)
        if not meta:
            continue
        folder = url.rsplit("/", 1)[0]
        if meta.race not in races or meta.model_gender not in genders:
            continue
        folder_to_seed.setdefault(folder, url)
        folder_to_pairs.setdefault(folder, set()).add((meta.race, meta.model_gender))

    urls: list[str] = []
    seen: set[str] = set()
    for folder, seed_url in sorted(folder_to_seed.items()):
        pairs = sorted(folder_to_pairs[folder])
        for race, gender in pairs:
            for url in generate_static_image_urls(
                [seed_url],
                races=(race,),
                genders=(gender,),
                parts=parts,
                dye_states=dye_states,
            ):
                if url not in seen:
                    urls.append(url)
                    seen.add(url)
    return urls


def check_image_url(url: str, timeout: float) -> dict[str, object]:
    status: int | None = None
    content_type: str | None = None
    content_length: int | None = None
    error: str | None = None
    try:
        status, content_type, content_length = request_metadata(url, "HEAD", timeout)
        if status in {403, 405, 501}:
            status, content_type, content_length = request_metadata(url, "GET", timeout, byte_range="bytes=0-0")
    except Exception as exc:
        error = str(exc)
    return {
        "source_image_url": url,
        "status": status,
        "content_type": content_type,
        "content_length": content_length,
        "error": error,
        "checked_at": timestamp(),
    }


def browser_retryable_check(row: dict[str, object]) -> bool:
    return (
        row.get("status") == 403
        and str(row.get("content_type") or "").casefold().startswith("text/html")
    )


def check_image_urls_browser(
    urls: list[str],
    *,
    profile_dir: Path,
    wait_seconds: float,
    batch_size: int,
) -> Iterable[dict[str, object]]:
    if not urls:
        return
    driver = create_visible_driver(profile_dir)
    try:
        fetch_verified_html(driver, urls[0], wait_seconds=wait_seconds)
        size = max(1, batch_size)
        for start in range(0, len(urls), size):
            batch_urls = urls[start : start + size]
            results = browser_check_batch(driver, batch_urls)
            if results and all(browser_retryable_check(row) for row in results):
                fetch_verified_html(driver, batch_urls[0], wait_seconds=wait_seconds)
                results = browser_check_batch(driver, batch_urls)
            for row in results:
                yield row
    finally:
        close_visible_driver(driver)


def browser_check_batch(
    driver: CdpChrome,
    urls: list[str],
) -> list[dict[str, object]]:
    encoded_urls = json.dumps(urls, ensure_ascii=True)
    expression = f"""
    (async () => {{
      const urls = {encoded_urls};
      return await Promise.all(urls.map(async (url) => {{
        try {{
          const response = await fetch(url, {{
            method: 'HEAD', credentials: 'include', cache: 'no-store'
          }});
          const rawLength = response.headers.get('content-length');
          return {{
            source_image_url: url,
            status: response.status,
            content_type: response.headers.get('content-type'),
            content_length: rawLength && /^\\d+$/.test(rawLength) ? Number(rawLength) : null,
            error: null
          }};
        }} catch (error) {{
          return {{
            source_image_url: url,
            status: null,
            content_type: null,
            content_length: null,
            error: String(error)
          }};
        }}
      }}));
    }})()
    """
    payload = driver.evaluate(expression, await_promise=True)
    if not isinstance(payload, list):
        raise RuntimeError("Chrome did not return an image URL check list")
    checked_at = timestamp()
    rows = []
    for row in payload:
        if not isinstance(row, dict):
            raise RuntimeError("Chrome returned an invalid image URL check row")
        rows.append({**row, "checked_at": checked_at})
    return rows


def request_metadata(
    url: str,
    method: str,
    timeout: float,
    byte_range: str | None = None,
) -> tuple[int | None, str | None, int | None]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "image/*,*/*;q=0.8",
    }
    if byte_range:
        headers["Range"] = byte_range
    request = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if method == "GET":
                response.read(1)
            return (
                getattr(response, "status", None),
                response.headers.get("Content-Type"),
                content_length_from_headers(response.headers),
            )
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            exc.headers.get("Content-Type") if exc.headers else None,
            content_length_from_headers(exc.headers) if exc.headers else None,
        )
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise RuntimeError(str(getattr(exc, "reason", exc))) from exc


def image_url_row(check: dict[str, object]) -> dict[str, object]:
    url = str(check["source_image_url"])
    meta = parse_gearsets_image_url(url)
    if meta is None:
        raise ValueError(f"Invalid gearsets image URL: {url}")
    status = check["status"]
    content_type = str(check["content_type"] or "")
    accessible = isinstance(status, int) and 200 <= status < 400 and (
        content_type.startswith("image/") or urlparse(url).netloc == GEARSETS_IMAGE_HOST
    )
    return {
        "source_image_url": url,
        "source_page_url": f"https://ffxiv.eorzeacollection.com/gearset/{meta.gearset_slug}",
        "accessible": accessible,
        "checked_at": check["checked_at"],
        "status": status,
        "content_type": check["content_type"],
        "content_length": check["content_length"],
        "error": check["error"],
        "image_kind": "gearset_static",
        "slot": meta.slot,
        "gearset_slug": meta.gearset_slug,
        "model_gender": meta.model_gender,
        "race": meta.race,
        "image_variant": meta.dye_state,
        "dye": True if meta.dye_state == "dyed" else "unknown",
        "source_type": meta.source_type,
        "source_path": meta.source_path,
        "collection_slug": meta.collection_slug,
        "role_slug": meta.role_slug,
        "family_slug": meta.family_slug,
        "image_part": meta.image_part,
    }


def build_manifest_rows(rows: Iterable[dict[str, object]], candidate_urls: set[str]) -> list[dict[str, object]]:
    checked_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("source_image_url") in candidate_urls
        and parse_gearsets_image_url(str(row.get("source_image_url")))
    ]
    dyed_keys = set()
    for row in checked_rows:
        if row.get("accessible") and row.get("image_variant") == "dyed":
            key = image_dye_key(str(row["source_image_url"]))
            if key is not None:
                dyed_keys.add(key)

    manifest_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in sorted(checked_rows, key=lambda value: str(value.get("source_image_url"))):
        url = str(row.get("source_image_url"))
        if url in seen or not row.get("accessible") or row.get("image_variant") != "normal":
            continue
        seen.add(url)
        manifest_row = dict(row)
        manifest_row["dye"] = True if image_dye_key(url) in dyed_keys else "unknown"
        manifest_rows.append(manifest_row)
    return manifest_rows


def image_dye_key(url: str) -> tuple[str, str, str, str, str] | None:
    meta = parse_gearsets_image_url(url)
    if meta is None:
        return None
    return (meta.source_path, meta.gearset_slug, meta.race, meta.model_gender, meta.image_part)


def content_length_from_headers(headers: object) -> int | None:
    content_range = headers.get("Content-Range")  # type: ignore[attr-defined]
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[-1]
        if total.isdigit():
            return int(total)
    content_length = headers.get("Content-Length")  # type: ignore[attr-defined]
    if content_length and content_length.isdigit():
        return int(content_length)
    return None


def print_progress(
    current: int,
    total: int,
    newly_checked: int,
    accessible: int,
    unavailable: int,
    started_at: float,
    final: bool,
) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = newly_checked / elapsed if newly_checked > 0 else 0.0
    remaining = max(total - current, 0)
    eta = format_duration(remaining / rate) if rate > 0 else "unknown"
    percent = (current / total * 100.0) if total else 100.0
    line = (
        f"Progress: {current}/{total} ({percent:5.1f}%) "
        f"new={newly_checked} ok={accessible} missing={unavailable} "
        f"rate={rate:6.1f}/s eta={eta}"
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


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


if __name__ == "__main__":
    raise SystemExit(main())
