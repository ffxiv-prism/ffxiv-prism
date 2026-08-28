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
    args = parser.parse_args()

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
    if not candidate_urls and cached_check_rows:
        candidate_urls = candidate_urls_from_checks(cached_check_rows)
        if candidate_urls:
            print(
                "No list-page image seeds were available; rebuilding the manifest "
                f"from {len(candidate_urls)} cached URL checks.",
                file=sys.stderr,
                flush=True,
            )
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
    checked_rows_by_url: dict[str, dict[str, object]] = {}
    if use_cached_checks:
        for row in cached_check_rows:
            url = row.get("source_image_url")
            if not url or url not in candidate_url_set:
                continue
            checked_rows_by_url[str(url)] = row
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(check_image_url, url, args.timeout): url for url in pending_urls}
        for future in concurrent.futures.as_completed(futures):
            row = image_url_row(future.result())
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

    if batch:
        append_jsonl(checks_path, batch)

    elapsed_seconds = time.monotonic() - started_at
    checked_rows_by_url.update(new_rows_by_url)
    manifest_rows = build_manifest_rows(checked_rows_by_url.values(), candidate_url_set)
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
