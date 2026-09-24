# 鍔熻兘锛氱紪鎺?Eorzea Collection 椤甸潰閲囬泦銆佸浘鐗囦笅杞姐€佸厓鏁版嵁钀界洏鍜屽彉浣撹窡韪€?
from __future__ import annotations

import hashlib
import json
import time

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from ffxiv_prism.collectors.eorzea_collection.http_client import PoliteHttpClient
from ffxiv_prism.collectors.eorzea_collection.models import (
    DownloadedImageRecord,
    GearsetRecord,
    ImageCandidate,
    ImageUrlRecord,
)
from ffxiv_prism.collectors.eorzea_collection.parser import (
    extract_gearset_links,
    extract_pagination_links,
    parse_gearset_detail,
)
from ffxiv_prism.collectors.eorzea_collection.url_utils import (
    BASE_URL,
    absolute_url,
    extension_from_url,
    extract_slug,
    short_hash,
)
from ffxiv_prism.jsonl import append_jsonl, write_json
from ffxiv_prism.paths import (
    portable_path,
    portable_payload,
    CACHE_DIR,
    EORZEA_HTML_DIR,
    EORZEA_IMAGE_DIR,
    EORZEA_METADATA_DIR,
    ensure_project_dirs,
)


@dataclass(frozen=True)
class CollectionSummary:
    pages_fetched: int = 0
    pages_from_cache: int = 0
    gearsets_parsed: int = 0
    gear_pieces_parsed: int = 0
    images_downloaded: int = 0
    images_from_cache: int = 0
    image_urls_checked: int = 0
    image_urls_accessible: int = 0
    image_urls_unavailable: int = 0
    errors: int = 0

    def add(self, **changes: int) -> "CollectionSummary":
        payload = self.__dict__.copy()
        for key, value in changes.items():
            payload[key] = payload.get(key, 0) + value
        return CollectionSummary(**payload)


class EorzeaCollectionCollector:
    def __init__(
        self,
        delay_seconds: float = 2.0,
        timeout_seconds: float = 30.0,
        image_workers: int = 1,
        respect_robots: bool = True,
        require_robots: bool = False,
    ) -> None:
        ensure_project_dirs()
        self.http = PoliteHttpClient(
            cache_dir=CACHE_DIR / "http",
            delay_seconds=delay_seconds,
            timeout_seconds=timeout_seconds,
            respect_robots=respect_robots,
            require_robots=require_robots,
        )
        self.html_dir = EORZEA_HTML_DIR
        self.image_dir = EORZEA_IMAGE_DIR
        self.metadata_dir = EORZEA_METADATA_DIR
        self.image_workers = max(1, image_workers)

    def discover_gearsets(
        self,
        start_url: str = f"{BASE_URL}/gearsets",
        max_pages: int = 1,
        force_refresh: bool = False,
    ) -> tuple[list[str], CollectionSummary]:
        queue = [absolute_url(start_url)]
        visited: set[str] = set()
        gearset_urls: list[str] = []
        seen_gearsets: set[str] = set()
        summary = CollectionSummary()

        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            html_text, from_cache = self._fetch_html(
                url,
                "gearset-list",
                short_hash(url),
                force_refresh=force_refresh,
            )
            summary = summary.add(
                pages_fetched=0 if from_cache else 1,
                pages_from_cache=1 if from_cache else 0,
            )
            for gearset_url in extract_gearset_links(html_text, url):
                if gearset_url not in seen_gearsets:
                    gearset_urls.append(gearset_url)
                    seen_gearsets.add(gearset_url)
            for page_url in extract_pagination_links(html_text, url, "/gearsets"):
                if page_url not in visited and page_url not in queue:
                    queue.append(page_url)

        append_jsonl(
            self.metadata_dir / "gearset_discovery_raw.jsonl",
            [{"source_page_url": url, "discovered_at": timestamp()} for url in gearset_urls],
        )
        return gearset_urls, summary

    def collect_gearset(
        self,
        gearset_url: str,
        guess_missing_model_variants: bool = False,
        races: tuple[str, ...] | list[str] | None = None,
        genders: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[GearsetRecord, list[object], list[DownloadedImageRecord], CollectionSummary]:
        gearset_url = absolute_url(gearset_url)
        slug = extract_slug(gearset_url, "gearset")
        summary = CollectionSummary()
        html_text, from_cache = self._fetch_html(gearset_url, "gearset", slug)
        summary = summary.add(
            pages_fetched=0 if from_cache else 1,
            pages_from_cache=1 if from_cache else 0,
        )
        gearset = parse_gearset_detail(
            html_text,
            gearset_url,
            expand_missing_model_variants=guess_missing_model_variants,
            races=races,
            genders=genders,
        )
        summary = summary.add(gearsets_parsed=1, gear_pieces_parsed=len(gearset.pieces))

        downloaded_images = self.download_images(gearset.images)
        summary = summary.add(
            images_downloaded=sum(1 for image in downloaded_images if not image.from_cache),
            images_from_cache=sum(1 for image in downloaded_images if image.from_cache),
        )
        append_jsonl(self.metadata_dir / "gearset_details_raw.jsonl", [gearset.to_dict()])
        append_jsonl(self.metadata_dir / "image_manifest_raw.jsonl", [image.to_dict() for image in downloaded_images])

        write_json(
            self.metadata_dir / "latest_run_summary.json",
            {
                "finished_at": timestamp(),
                "gearset_url": gearset_url,
                "summary": summary.__dict__,
            },
        )
        return gearset, [], downloaded_images, summary

    def collect_gearset_image_urls(
        self,
        gearset_url: str,
        verify: bool = True,
        guess_missing_model_variants: bool = False,
        races: tuple[str, ...] | list[str] | None = None,
        genders: tuple[str, ...] | list[str] | None = None,
        parts: tuple[str, ...] | list[str] | None = None,
        dye_states: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[GearsetRecord, list[ImageUrlRecord], CollectionSummary]:
        gearset_url = absolute_url(gearset_url)
        slug = extract_slug(gearset_url, "gearset")
        summary = CollectionSummary()
        html_text, from_cache = self._fetch_html(gearset_url, "gearset", slug)
        summary = summary.add(
            pages_fetched=0 if from_cache else 1,
            pages_from_cache=1 if from_cache else 0,
        )
        gearset = parse_gearset_detail(
            html_text,
            gearset_url,
            expand_missing_model_variants=guess_missing_model_variants,
            races=races,
            genders=genders,
            parts=parts,
            dye_states=dye_states,
        )
        summary = summary.add(gearsets_parsed=1, gear_pieces_parsed=len(gearset.pieces))

        url_records = self.verify_image_urls(gearset.images, verify=verify)
        accessible_records = [record for record in url_records if record.accessible]
        manifest_records = build_url_manifest_records(accessible_records)
        summary = summary.add(
            image_urls_checked=len(url_records),
            image_urls_accessible=len(manifest_records),
            image_urls_unavailable=len(url_records) - len(accessible_records),
        )
        append_jsonl(self.metadata_dir / "gearset_details_raw.jsonl", [gearset.to_dict()])
        append_jsonl(self.metadata_dir / "image_url_checks_raw.jsonl", [record.to_dict() for record in url_records])
        append_jsonl(
            self.metadata_dir / "image_url_manifest_raw.jsonl",
            [record.to_dict() for record in manifest_records],
        )
        write_json(
            self.metadata_dir / "latest_url_manifest_summary.json",
            {
                "finished_at": timestamp(),
                "gearset_url": gearset_url,
                "summary": summary.__dict__,
            },
        )
        return gearset, manifest_records, summary

    def download_images(
        self,
        image_candidates: list[ImageCandidate],
        progress: Callable[[int, int, DownloadedImageRecord | None], None] | None = None,
    ) -> list[DownloadedImageRecord]:
        if self.image_workers > 1:
            return self._download_images_parallel(image_candidates, progress=progress)

        records: list[DownloadedImageRecord] = []
        unique_candidates: list[ImageCandidate] = []
        seen: set[str] = set()
        for candidate in image_candidates:
            if candidate.source_image_url in seen:
                continue
            unique_candidates.append(candidate)
            seen.add(candidate.source_image_url)

        total = len(unique_candidates)
        for index, candidate in enumerate(unique_candidates, start=1):
            record = self._download_image_record(candidate)
            if record:
                records.append(record)
            if progress:
                progress(index, total, record)
        return records

    def verify_image_urls(self, image_candidates: list[ImageCandidate], verify: bool = True) -> list[ImageUrlRecord]:
        records: list[ImageUrlRecord] = []
        seen: set[str] = set()
        for candidate in image_candidates:
            if candidate.source_image_url in seen:
                continue
            seen.add(candidate.source_image_url)
            records.append(self._image_url_record(candidate, verify=verify))
        return records

    def _download_images_parallel(
        self,
        image_candidates: list[ImageCandidate],
        progress: Callable[[int, int, DownloadedImageRecord | None], None] | None = None,
    ) -> list[DownloadedImageRecord]:
        unique_candidates: list[ImageCandidate] = []
        seen: set[str] = set()
        for candidate in image_candidates:
            if candidate.source_image_url in seen:
                continue
            unique_candidates.append(candidate)
            seen.add(candidate.source_image_url)

        records: list[DownloadedImageRecord] = []
        completed = 0
        total = len(unique_candidates)
        with ThreadPoolExecutor(max_workers=self.image_workers) as executor:
            future_to_candidate = {
                executor.submit(self._download_image_record, candidate): candidate for candidate in unique_candidates
            }
            for future in as_completed(future_to_candidate):
                record = future.result()
                if record:
                    records.append(record)
                completed += 1
                if progress:
                    progress(completed, total, record)
        records.sort(key=lambda record: record.source_image_url)
        return records

    def _download_image_record(self, candidate: ImageCandidate) -> DownloadedImageRecord | None:
        try:
            local_path, from_cache = self._download_image(candidate)
            return self.downloaded_image_record(candidate, local_path, from_cache)
        except Exception as exc:
            if "HTTP 404" not in str(exc):
                self._log_error("download_image", candidate.source_image_url, exc)
            return None

    def _image_url_record(self, candidate: ImageCandidate, verify: bool) -> ImageUrlRecord:
        if not verify:
            return self._image_url_record_from_candidate(candidate, accessible=True, status=None)
        try:
            result = self.http.check_url(candidate.source_image_url)
            content_type = result.content_type or ""
            looks_like_image = content_type.startswith("image/") or candidate.source_image_url.endswith(".png")
            accessible = result.status is not None and 200 <= result.status < 400 and looks_like_image
            return self._image_url_record_from_candidate(
                candidate,
                accessible=accessible,
                status=result.status,
                content_type=result.content_type,
                content_length=result.content_length,
            )
        except Exception as exc:
            self._log_error("check_image_url", candidate.source_image_url, exc)
            return self._image_url_record_from_candidate(candidate, accessible=False, error=str(exc))

    def _image_url_record_from_candidate(
        self,
        candidate: ImageCandidate,
        accessible: bool,
        status: int | None,
        content_type: str | None = None,
        content_length: int | None = None,
        error: str | None = None,
    ) -> ImageUrlRecord:
        return ImageUrlRecord(
            source_image_url=candidate.source_image_url,
            source_page_url=candidate.source_page_url,
            accessible=accessible,
            checked_at=timestamp(),
            status=status,
            content_type=content_type,
            content_length=content_length,
            error=error,
            image_kind=candidate.image_kind,
            slot=candidate.slot,
            gearset_slug=candidate.gearset_slug,
            model_gender=candidate.model_gender,
            race=candidate.race,
            image_variant=candidate.dye_state,
            dye=True if candidate.dye_state == "dyed" else "unknown",
            source_type=candidate.source_type,
            source_path=candidate.source_path,
            collection_slug=candidate.collection_slug,
            role_slug=candidate.role_slug,
            family_slug=candidate.family_slug,
            image_part=candidate.image_part,
        )

    def _fetch_html(
        self,
        url: str,
        category: str,
        slug: str,
        *,
        force_refresh: bool = False,
    ) -> tuple[str, bool]:
        target = self.html_dir / category / f"{slug}.html"
        text, result = self.http.fetch_text(
            url,
            target,
            force_refresh=force_refresh,
        )
        return text, result.from_cache

    def _download_image(self, candidate: ImageCandidate) -> tuple[Path, bool]:
        target = self.image_target(candidate)
        result = self.http.fetch_bytes(candidate.source_image_url, target, force_refresh=target.is_file() and not self.image_is_readable(target))
        return result.local_path, result.from_cache

    @staticmethod
    def image_is_readable(path: Path) -> bool:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
            return True
        except (OSError, ValueError, SyntaxError):
            return False

    def image_target(self, candidate: ImageCandidate) -> Path:
        group = candidate.gearset_slug or "misc"
        suffix = extension_from_url(candidate.source_image_url, ".img")
        semantic_name = self._semantic_image_filename(candidate, suffix)
        return self.image_dir / candidate.image_kind / group / semantic_name

    def downloaded_image_record(
        self,
        candidate: ImageCandidate,
        local_path: Path,
        from_cache: bool,
    ) -> DownloadedImageRecord:
        if not self.image_is_readable(local_path):
            raise ValueError(f"Downloaded image cannot be decoded: {local_path}")
        return DownloadedImageRecord(
            source_image_url=candidate.source_image_url,
            source_page_url=candidate.source_page_url,
            local_path=portable_path(local_path) or str(local_path),
            sha256=sha256_file(local_path),
            byte_size=local_path.stat().st_size,
            from_cache=from_cache,
            image_kind=candidate.image_kind,
            slot=candidate.slot,
            gearset_slug=candidate.gearset_slug,
            model_gender=candidate.model_gender,
            race=candidate.race,
            dye_state=candidate.dye_state,
            source_type=candidate.source_type,
            source_path=candidate.source_path,
            collection_slug=candidate.collection_slug,
            role_slug=candidate.role_slug,
            family_slug=candidate.family_slug,
            image_part=candidate.image_part,
        )

    def _semantic_image_filename(self, candidate: ImageCandidate, suffix: str) -> str:
        pieces = [
            candidate.source_type,
            candidate.collection_slug,
            candidate.gearset_slug,
            candidate.race,
            candidate.model_gender,
            candidate.dye_state,
            candidate.image_part,
        ]
        if all(pieces):
            return "-".join(str(piece) for piece in pieces) + suffix
        return f"{short_hash(candidate.source_image_url, 24)}{suffix}"

    def _log_error(self, stage: str, url: str, exc: Exception) -> None:
        append_jsonl(
            self.metadata_dir / "errors.jsonl",
            [
                {
                    "stage": stage,
                    "url": url,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "created_at": timestamp(),
                }
            ],
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_url_manifest_records(records: list[ImageUrlRecord]) -> list[ImageUrlRecord]:
    dyed_keys = {
        image_url_record_dye_key(record)
        for record in records
        if record.accessible and record.image_variant == "dyed"
    }
    manifest_records: list[ImageUrlRecord] = []
    for record in records:
        if not record.accessible or record.image_variant != "normal":
            continue
        manifest_records.append(
            replace(
                record,
                dye=True if image_url_record_dye_key(record) in dyed_keys else "unknown",
            )
        )
    return manifest_records


def image_url_record_dye_key(record: ImageUrlRecord) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    return (record.source_path, record.gearset_slug, record.race, record.model_gender, record.image_part)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def print_summary(summary: CollectionSummary, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"summary": summary.__dict__}
    if extra:
        payload.update(extra)
    print(json.dumps(portable_payload(payload), ensure_ascii=False, indent=2, sort_keys=True))
