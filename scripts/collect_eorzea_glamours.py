from __future__ import annotations

import argparse
import io
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import cloudscraper
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from ffxiv_prism.collectors.eorzea_collection.glamour_parser import (
    GLAMOUR_BASE_URL,
    enrich_equipment_from_sqlite,
    extract_glamour_id,
    parse_glamour_page,
)
from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import (
    CACHE_DIR,
    EORZEA_GLAMOUR_HTML_DIR,
    EORZEA_GLAMOUR_IMAGE_DIR,
    EORZEA_GLAMOUR_METADATA_DIR,
    PROCESSED_DIR,
    portable_path,
)

DEFAULT_OUTPUT = EORZEA_GLAMOUR_METADATA_DIR / "glamour_examples_raw.jsonl"
DEFAULT_SUMMARY = EORZEA_GLAMOUR_METADATA_DIR / "glamour_examples_summary.json"
DEFAULT_PROFILE_DIR = CACHE_DIR / "eorzea_glamour_browser"
DEFAULT_DB = PROCESSED_DIR / "eorzea.sqlite"
CORE_SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "Referer": f"{GLAMOUR_BASE_URL}/",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a bounded set of labeled Eorzea Collection glamour pages.")
    parser.add_argument("urls", nargs="*", help="Public glamour page URLs or numeric glamour ids.")
    parser.add_argument("--url-file", type=Path, help="Text file with one URL or glamour id per line.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--fetch-mode", choices=("http", "browser"), default="http")
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--wait-seconds", type=int, default=90)
    parser.add_argument("--headless", action="store_true", help="Only applies to --fetch-mode browser.")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument(
        "--require-all-core-matched",
        action="store_true",
        help="Reject a glamour when any non-empty core slot lacks an exact local piece match.",
    )
    parser.add_argument("--refresh", action="store_true", help="Refetch cached HTML and images.")
    parser.add_argument("--max-pages", type=int, default=10, help="Hard safety bound for one run.")
    args = parser.parse_args()
    if args.require_all_core_matched and not args.db.is_file():
        parser.error(f"--require-all-core-matched requires an existing database: {args.db}")

    urls = normalize_urls(args.urls, args.url_file)
    if not urls:
        parser.error("Provide at least one glamour URL/id or --url-file")
    if len(urls) > args.max_pages:
        parser.error(f"Requested {len(urls)} pages, exceeding --max-pages={args.max_pages}")

    existing = {int(row["glamour_id"]): row for row in read_jsonl(args.output) if row.get("glamour_id")}
    driver = None
    session = None
    collected = []
    errors = []
    try:
        for index, url in enumerate(urls):
            glamour_id = extract_glamour_id(url)
            cache_path = EORZEA_GLAMOUR_HTML_DIR / f"{glamour_id}.html"
            try:
                fetch_method = "cache"
                if args.refresh or not cache_path.exists():
                    if args.fetch_mode == "http":
                        if session is None:
                            session = create_http_session()
                        html, resolved_url = fetch_page_http(session, url)
                    else:
                        if driver is None:
                            driver = create_driver(args.profile_dir, args.headless)
                        html, resolved_url = fetch_page_browser(driver, url, args.wait_seconds)
                    fetch_method = args.fetch_mode
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(html, encoding="utf-8")
                else:
                    html = cache_path.read_text(encoding="utf-8")
                    resolved_url = url

                record = parse_glamour_page(html, resolved_url)
                validate_record(record)
                record["html_cache_path"] = portable_path(cache_path)
                record["fetch_method"] = fetch_method
                record["collected_at"] = utc_now()
                if args.db.exists():
                    enrich_equipment_from_sqlite(record, args.db)
                if args.require_all_core_matched:
                    require_all_core_items_matched(record)
                if args.no_images:
                    record["images"] = []
                else:
                    if session is None:
                        session = create_http_session()
                    record["images"] = collect_images(record, args.refresh, session)
                existing[int(record["glamour_id"])] = record
                collected.append(record)
                print_record_summary(record)
            except Exception as exc:
                errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
                print(f"ERROR {url}: {type(exc).__name__}: {exc}", flush=True)
            if index + 1 < len(urls):
                time.sleep(max(0.0, args.delay))
    finally:
        if driver is not None:
            driver.quit()
        if session is not None:
            session.close()

    rows = [existing[key] for key in sorted(existing)]
    write_jsonl(args.output, rows)
    summary = {
        "output": portable_path(args.output),
        "records_total": len(rows),
        "requested": len(urls),
        "collected_this_run": len(collected),
        "errors": errors,
        "image_count_total": sum(len(row.get("images", [])) for row in rows),
        "core_slot_labels": sum(
            1
            for row in rows
            for item in row.get("equipment", [])
            if not item.get("is_empty")
        ),
        "finished_at": utc_now(),
    }
    write_json(args.summary, summary)
    print(f"records_total={len(rows)} collected={len(collected)} errors={len(errors)}")
    return 1 if errors else 0


def create_http_session() -> cloudscraper.CloudScraper:
    session = cloudscraper.create_scraper()
    session.trust_env = False
    session.headers.update(BROWSER_HEADERS)
    try:
        session.get(f"{GLAMOUR_BASE_URL}/", timeout=30)
    except Exception:
        pass
    return session


def fetch_page_http(session: cloudscraper.CloudScraper, url: str) -> tuple[str, str]:
    response = session.get(url, timeout=45)
    response.raise_for_status()
    html = response.text
    if not page_has_glamour_content(html):
        raise ValueError("Response did not contain a glamour image gallery and five-slot equipment section")
    return html, response.url


def create_driver(profile_dir: Path, headless: bool) -> webdriver.Chrome:
    profile_dir.mkdir(parents=True, exist_ok=True)
    options = Options()
    options.add_argument(f"--user-data-dir={profile_dir.resolve()}")
    options.add_argument("--window-size=1440,1200")
    if headless:
        options.add_argument("--headless=new")
    return webdriver.Chrome(options=options)


def fetch_page_browser(driver: webdriver.Chrome, url: str, wait_seconds: int) -> tuple[str, str]:
    driver.get(url)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        html = driver.page_source
        if page_has_glamour_content(html):
            return html, driver.current_url
        time.sleep(1.0)
    raise TimeoutError("Glamour content did not load; complete any visible browser verification and retry")


def page_has_glamour_content(html: str) -> bool:
    return (
        "gear-icon-box-slot-name" in html
        and "glamours.eorzeacollection.com" in html
        and "HEAD" in html
        and "BODY" in html
    )


def validate_record(record: dict) -> None:
    glamour_id = int(record["glamour_id"])
    expected_prefix = f"https://glamours.eorzeacollection.com/{glamour_id}/"
    invalid_images = [url for url in record.get("image_urls", []) if not url.startswith(expected_prefix)]
    if invalid_images:
        raise ValueError(f"Image URLs outside glamour {glamour_id}: {invalid_images}")
    if not record.get("image_urls"):
        raise ValueError("No player glamour screenshots found")
    slots = [item.get("slot") for item in record.get("equipment", [])]
    if slots != list(CORE_SLOTS):
        raise ValueError(f"Expected exactly five core slots, got {slots}")


def is_exact_piece_match(item: dict) -> bool:
    match_type = item.get("piece_match_type")
    if match_type is None:
        return bool(item.get("matched_piece_ids"))
    return match_type == "exact_item"


def unmatched_core_items(record: dict) -> list[dict]:
    return [
        item
        for item in record.get("equipment", [])
        if item.get("is_core_slot")
        and not item.get("is_empty")
        and not is_exact_piece_match(item)
    ]


def require_all_core_items_matched(record: dict) -> None:
    unmatched = unmatched_core_items(record)
    if not unmatched:
        return
    details = ", ".join(
        f"{item.get('slot')}={item.get('item_name') or 'unknown'}"
        for item in unmatched
    )
    raise ValueError(f"Non-exact non-empty core items: {details}")

def collect_images(record: dict, refresh: bool, session: cloudscraper.CloudScraper) -> list[dict]:
    results = []
    glamour_id = int(record["glamour_id"])
    expected_prefix = f"https://glamours.eorzeacollection.com/{glamour_id}/"
    output_dir = EORZEA_GLAMOUR_IMAGE_DIR / str(glamour_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {
        Path(urlparse(url).path).name
        for url in record.get("image_urls", [])
    }
    for stale_path in output_dir.iterdir():
        if stale_path.is_file() and stale_path.name not in expected_names:
            stale_path.unlink()
    for position, url in enumerate(record.get("image_urls", [])):
        if not url.startswith(expected_prefix):
            raise ValueError(f"Refusing non-gallery image URL: {url}")
        filename = Path(urlparse(url).path).name or f"image-{position}.png"
        output_path = output_dir / filename
        if refresh or not output_path.exists():
            response = session.get(url, timeout=60)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                raise ValueError(f"Expected image response for {url}, got {content_type!r}")
            image_info = inspect_image_bytes(response.content, url)
            output_path.write_bytes(response.content)
        else:
            image_info = inspect_image_bytes(output_path.read_bytes(), str(output_path))
        results.append(
            {
                "position": position,
                "source_image_url": url,
                "local_path": portable_path(output_path),
                "byte_size": output_path.stat().st_size,
                **image_info,
            }
        )
    return results


def inspect_image_bytes(content: bytes, source: str) -> dict:
    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            image_format = (image.format or "").lower()
            image.verify()
    except Exception as exc:
        raise ValueError(f"Invalid image body from {source}: {exc}") from exc
    if width < 200 or height < 200:
        raise ValueError(f"Refusing likely icon from {source}: {width}x{height}")
    return {"width": width, "height": height, "image_format": image_format}


def print_record_summary(record: dict) -> None:
    print(
        f"\nglamour={record['glamour_id']} title={record.get('title')!r} "
        f"model={record.get('model')!r} submitted={record.get('submitted_at')!r}",
        flush=True,
    )
    if record.get("tags"):
        print(f"  tags: {', '.join(record['tags'])}", flush=True)
    for item in record.get("equipment", []):
        if item.get("is_empty"):
            detail = "empty"
        else:
            detail = item.get("item_name") or "unknown"
            if item.get("dyes"):
                detail += f" | dyes: {', '.join(item['dyes'])}"
            if item.get("matched_piece_ids"):
                detail += f" | piece_ids: {item['matched_piece_ids']}"
            else:
                detail += " | piece_ids: unmatched"
            if item.get("site_item_id"):
                detail += f" | site_item_id: {item['site_item_id']}"
            if item.get("occupied_slots") and len(item["occupied_slots"]) > 1:
                detail += f" | occupies: {', '.join(item['occupied_slots'])}"
        print(f"  {item['slot']}: {detail}", flush=True)
    for image in record.get("images", []):
        print(
            f"  image[{image['position']}]: {image['width']}x{image['height']} "
            f"{image['local_path']}",
            flush=True,
        )


def normalize_urls(values: list[str], url_file: Path | None) -> list[str]:
    raw = list(values)
    if url_file:
        raw.extend(
            line.strip()
            for line in url_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    urls = []
    seen = set()
    for value in raw:
        if value.isdigit():
            url = f"{GLAMOUR_BASE_URL}/glamour/{value}"
        else:
            glamour_id = extract_glamour_id(value)
            if glamour_id is None:
                raise ValueError(f"Not a glamour URL or id: {value}")
            url = f"{GLAMOUR_BASE_URL}/glamour/{glamour_id}"
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
