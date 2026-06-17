# 功能：编排 Eorzea Collection 页面采集、图片下载、元数据落盘和变体跟踪。

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ffxiv_prism.collectors.eorzea_collection.http_client import PoliteHttpClient
from ffxiv_prism.collectors.eorzea_collection.models import (
    DownloadedImageRecord,
    GearPieceRecord,
    GearsetRecord,
    GlamourVariantRecord,
    ImageCandidate,
)
from ffxiv_prism.collectors.eorzea_collection.parser import (
    extract_gearset_links,
    extract_glamour_links,
    extract_pagination_links,
    parse_gearset_detail,
    parse_glamour_detail,
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
    glamour_variants_parsed: int = 0
    images_downloaded: int = 0
    images_from_cache: int = 0
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
            html_text, from_cache = self._fetch_html(url, "gearset-list", short_hash(url))
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
        include_piece_glamours: bool = False,
        max_glamours_per_piece: int = 0,
        guess_missing_model_variants: bool = False,
        races: tuple[str, ...] | list[str] | None = None,
        genders: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[GearsetRecord, list[GlamourVariantRecord], list[DownloadedImageRecord], CollectionSummary]:
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

        glamour_variants: list[GlamourVariantRecord] = []
        if include_piece_glamours and max_glamours_per_piece > 0:
            variants, variant_images, variant_summary = self.collect_piece_glamour_variants(
                gearset.pieces,
                max_glamours_per_piece=max_glamours_per_piece,
            )
            glamour_variants.extend(variants)
            downloaded_images.extend(variant_images)
            summary = summary.add(**variant_summary.__dict__)

        write_json(
            self.metadata_dir / "latest_run_summary.json",
            {
                "finished_at": timestamp(),
                "gearset_url": gearset_url,
                "summary": summary.__dict__,
            },
        )
        return gearset, glamour_variants, downloaded_images, summary

    def collect_piece_glamour_variants(
        self,
        pieces: list[GearPieceRecord],
        max_glamours_per_piece: int,
    ) -> tuple[list[GlamourVariantRecord], list[DownloadedImageRecord], CollectionSummary]:
        variants: list[GlamourVariantRecord] = []
        downloaded_images: list[DownloadedImageRecord] = []
        summary = CollectionSummary()

        for piece in pieces:
            if not piece.glamours_using_piece_url:
                continue
            list_slug = f"piece-{short_hash(piece.glamours_using_piece_url)}"
            try:
                html_text, from_cache = self._fetch_html(piece.glamours_using_piece_url, "piece-glamours", list_slug)
                summary = summary.add(
                    pages_fetched=0 if from_cache else 1,
                    pages_from_cache=1 if from_cache else 0,
                )
                glamour_urls = extract_glamour_links(html_text, piece.glamours_using_piece_url)
                append_jsonl(
                    self.metadata_dir / "gear_piece_glamour_links_raw.jsonl",
                    [
                        {
                            "piece_name": piece.name,
                            "piece_slot": piece.slot,
                            "piece_url": piece.source_item_url,
                            "glamour_url": glamour_url,
                            "source_list_url": piece.glamours_using_piece_url,
                            "discovered_at": timestamp(),
                        }
                        for glamour_url in glamour_urls
                    ],
                )
                for glamour_url in glamour_urls[:max_glamours_per_piece]:
                    variant, images, variant_summary = self.collect_glamour_variant(
                        glamour_url,
                        related_piece=piece,
                    )
                    variants.append(variant)
                    downloaded_images.extend(images)
                    summary = summary.add(**variant_summary.__dict__)
            except Exception as exc:
                self._log_error("piece_glamour_variants", piece.glamours_using_piece_url, exc)
                summary = summary.add(errors=1)

        if variants:
            append_jsonl(self.metadata_dir / "glamour_variants_raw.jsonl", [variant.to_dict() for variant in variants])
        if downloaded_images:
            append_jsonl(self.metadata_dir / "image_manifest_raw.jsonl", [image.to_dict() for image in downloaded_images])
        return variants, downloaded_images, summary

    def collect_glamour_variant(
        self,
        glamour_url: str,
        related_piece: GearPieceRecord | None = None,
    ) -> tuple[GlamourVariantRecord, list[DownloadedImageRecord], CollectionSummary]:
        glamour_url = absolute_url(glamour_url)
        slug = extract_slug(glamour_url, "glamour")
        summary = CollectionSummary()
        html_text, from_cache = self._fetch_html(glamour_url, "glamour", slug)
        summary = summary.add(
            pages_fetched=0 if from_cache else 1,
            pages_from_cache=1 if from_cache else 0,
        )
        variant = parse_glamour_detail(
            html_text,
            glamour_url,
            related_piece_name=related_piece.name if related_piece else None,
            related_piece_slot=related_piece.slot if related_piece else None,
            related_piece_url=related_piece.source_item_url if related_piece else None,
        )
        images = self.download_images(variant.images)
        summary = summary.add(
            glamour_variants_parsed=1,
            images_downloaded=sum(1 for image in images if not image.from_cache),
            images_from_cache=sum(1 for image in images if image.from_cache),
        )
        return variant, images, summary

    def download_images(self, image_candidates: list[ImageCandidate]) -> list[DownloadedImageRecord]:
        if self.image_workers > 1:
            return self._download_images_parallel(image_candidates)

        records: list[DownloadedImageRecord] = []
        seen: set[str] = set()
        for candidate in image_candidates:
            if candidate.source_image_url in seen:
                continue
            seen.add(candidate.source_image_url)
            record = self._download_image_record(candidate)
            if record:
                records.append(record)
        return records

    def _download_images_parallel(self, image_candidates: list[ImageCandidate]) -> list[DownloadedImageRecord]:
        unique_candidates: list[ImageCandidate] = []
        seen: set[str] = set()
        for candidate in image_candidates:
            if candidate.source_image_url in seen:
                continue
            unique_candidates.append(candidate)
            seen.add(candidate.source_image_url)

        records: list[DownloadedImageRecord] = []
        with ThreadPoolExecutor(max_workers=self.image_workers) as executor:
            future_to_candidate = {
                executor.submit(self._download_image_record, candidate): candidate for candidate in unique_candidates
            }
            for future in as_completed(future_to_candidate):
                record = future.result()
                if record:
                    records.append(record)
        records.sort(key=lambda record: record.source_image_url)
        return records

    def _download_image_record(self, candidate: ImageCandidate) -> DownloadedImageRecord | None:
        try:
            local_path, from_cache = self._download_image(candidate)
            digest = sha256_file(local_path)
            return DownloadedImageRecord(
                source_image_url=candidate.source_image_url,
                source_page_url=candidate.source_page_url,
                local_path=str(local_path),
                sha256=digest,
                byte_size=local_path.stat().st_size,
                from_cache=from_cache,
                image_kind=candidate.image_kind,
                slot=candidate.slot,
                gearset_slug=candidate.gearset_slug,
                glamour_slug=candidate.glamour_slug,
                piece_name=candidate.piece_name,
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
        except Exception as exc:
            if "HTTP 404" not in str(exc):
                self._log_error("download_image", candidate.source_image_url, exc)
            return None

    def _fetch_html(self, url: str, category: str, slug: str) -> tuple[str, bool]:
        target = self.html_dir / category / f"{slug}.html"
        text, result = self.http.fetch_text(url, target)
        return text, result.from_cache

    def _download_image(self, candidate: ImageCandidate) -> tuple[Path, bool]:
        group = candidate.gearset_slug or candidate.glamour_slug or "misc"
        suffix = extension_from_url(candidate.source_image_url, ".img")
        semantic_name = self._semantic_image_filename(candidate, suffix)
        target = self.image_dir / candidate.image_kind / group / semantic_name
        result = self.http.fetch_bytes(candidate.source_image_url, target)
        return result.local_path, result.from_cache

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


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def print_summary(summary: CollectionSummary, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"summary": summary.__dict__}
    if extra:
        payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
