"""Collect alternate HEAD appearances for every known normal head render.

Requires the optional collection dependency: pip install '.[collection]'.
Uses the same image/URL manifests and filenames as the ordinary collector.
HTTP errors are recorded separately from absent (404/410) appearances.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import json
from pathlib import Path
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ffxiv_prism.collectors.eorzea_collection.collector import EorzeaCollectionCollector, timestamp
from ffxiv_prism.collectors.eorzea_collection.image_url import parse_gearsets_image_url
from ffxiv_prism.collectors.eorzea_collection.models import ImageCandidate
from ffxiv_prism.jsonl import append_jsonl, read_jsonl, write_jsonl_atomic
from ffxiv_prism.paths import EORZEA_METADATA_DIR
from scripts.build_eorzea_image_url_manifest_from_list_cache import image_url_row
from scripts.download_eorzea_images_from_url_manifest import select_manifest_rows_to_append


def alternate_urls(rows):
    return sorted({str(r["source_image_url"]).removesuffix("head.png") + "visor.png"
                   for r in rows if (m := parse_gearsets_image_url(str(r.get("source_image_url", ""))))
                   and m.image_part == "head" and m.dye_state == "normal"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")
    from curl_cffi import requests

    manifest_path = EORZEA_METADATA_DIR / "image_url_manifest_raw.jsonl"
    image_path = EORZEA_METADATA_DIR / "image_manifest_raw.jsonl"
    checks_path = EORZEA_METADATA_DIR / "image_url_checks_raw.jsonl"
    urls = alternate_urls(read_jsonl(manifest_path))
    previous = {r["source_image_url"]: r for r in read_jsonl(checks_path)}
    collector = EorzeaCollectionCollector(respect_robots=False)

    def collect(url):
        meta = parse_gearsets_image_url(url)
        fields = {k: v for k, v in meta.to_dict().items() if k in ImageCandidate.__dataclass_fields__}
        candidate = ImageCandidate(source_image_url=url,
            source_page_url=f"https://ffxiv.eorzeacollection.com/gearset/{meta.gearset_slug}",
            image_kind="gearset_static", **fields)
        target = collector.image_target(candidate)
        if target.is_file() and collector.image_is_readable(target):
            return previous.get(url), collector.downloaded_image_record(candidate, target, True).to_dict()
        old = previous.get(url, {})
        if not args.refresh and old.get("transport") == "curl-chrome" and old.get("status") in (404, 410):
            return old, None
        for attempt in range(3):
            check = dict(source_image_url=url, checked_at=timestamp(), status=None,
                         content_type=None, content_length=None, error=None)
            try:
                response = requests.get(url, impersonate="chrome", timeout=args.timeout)
                check.update(status=response.status_code, content_type=response.headers.get("Content-Type"),
                             content_length=len(response.content))
                record = None
                if response.status_code == 200:
                    if not str(check["content_type"]).startswith("image/"):
                        raise ValueError("Server returned a non-image response")
                    with Image.open(BytesIO(response.content)) as image:
                        image.load()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".partial")
                    temporary.write_bytes(response.content)
                    temporary.replace(target)
                    record = collector.downloaded_image_record(candidate, target, False).to_dict()
                elif response.status_code not in (404, 410):
                    raise RuntimeError(f"HTTP {response.status_code}")
                row = image_url_row(check)
                row["transport"] = "curl-chrome"
                return row, record
            except Exception as exc:
                check["error"] = str(exc)
                if attempt < 2:
                    time.sleep(attempt + 1)
        row = image_url_row(check)
        row.update(accessible=False, transport="curl-chrome")
        return row, None

    checks, images = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect, url) for url in urls]
        for n, future in enumerate(as_completed(futures), 1):
            check, record = future.result()
            if check:
                checks.append(check)
                append_jsonl(checks_path, [check])
            if record:
                images.append(record)
                # Persist successful downloads even if a later request is interrupted.
                append_jsonl(image_path, select_manifest_rows_to_append(read_jsonl(image_path), [record]))
            if n % 100 == 0 or n == len(urls):
                print(f"checked={n}/{len(urls)} images={len(images)} errors={sum(bool(r.get('error')) for r in checks)}", flush=True)
    merged = {r["source_image_url"]: r for r in read_jsonl(manifest_path)}
    merged.update({r["source_image_url"]: r for r in checks if r.get("accessible")})
    write_jsonl_atomic(manifest_path, [merged[k] for k in sorted(merged)])
    report = dict(finished_at=timestamp(), candidates=len(urls), images=len(images),
                  downloaded=sum(not r["from_cache"] for r in images),
                  gearsets=len({r["gearset_slug"] for r in images}),
                  absent=sum(r.get("status") in (404, 410) for r in checks),
                  errors=[r for r in checks if r.get("error")])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({k: v if k != "errors" else len(v) for k,v in report.items()}), flush=True)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
