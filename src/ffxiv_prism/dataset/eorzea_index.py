# 功能：把 Eorzea Collection 原始图片清单整理成后续检索和打标签可用的数据集索引。

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ffxiv_prism.jsonl import append_jsonl, read_jsonl, write_json
from ffxiv_prism.paths import EORZEA_METADATA_DIR, PROCESSED_DIR, ensure_project_dirs


def build_eorzea_image_index(
    manifest_path: Path | None = None,
    output_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deduplicated image index and return its coverage summary."""

    ensure_project_dirs()
    manifest_path = manifest_path or EORZEA_METADATA_DIR / "image_manifest_raw.jsonl"
    output_path = output_path or PROCESSED_DIR / "eorzea_images_index.jsonl"
    summary_path = summary_path or PROCESSED_DIR / "eorzea_dataset_summary.json"

    rows = read_jsonl(manifest_path)
    deduped = dedupe_by_hash_and_url(rows)

    if output_path.exists():
        output_path.unlink()
    append_jsonl(output_path, deduped)

    summary = summarize_image_index(deduped)
    write_json(summary_path, summary)
    return summary


def dedupe_by_hash_and_url(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for row in rows:
        key = (row.get("sha256"), row.get("source_image_url"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalize_image_row(row))
    return deduped


def normalize_image_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": row.get("sha256") or row.get("source_image_url"),
        "local_path": row.get("local_path"),
        "source": "eorzea_collection",
        "source_page_url": row.get("source_page_url"),
        "source_image_url": row.get("source_image_url"),
        "sha256": row.get("sha256"),
        "byte_size": row.get("byte_size"),
        "image_kind": row.get("image_kind"),
        "slot": row.get("slot"),
        "gearset_slug": row.get("gearset_slug"),
        "model_gender": row.get("model_gender"),
        "race": row.get("race"),
        "dye_state": row.get("dye_state"),
        "source_type": row.get("source_type"),
        "source_path": row.get("source_path"),
        "collection_slug": row.get("collection_slug"),
        "role_slug": row.get("role_slug"),
        "family_slug": row.get("family_slug"),
        "image_part": row.get("image_part"),
    }


def summarize_image_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind = Counter(row.get("image_kind") or "unknown" for row in rows)
    by_slot = Counter(row.get("slot") or "unknown" for row in rows)
    by_part = Counter(row.get("image_part") or "unknown" for row in rows)
    by_role = Counter(row.get("role_slug") or "unknown" for row in rows)
    by_gender = Counter(row.get("model_gender") or "unknown" for row in rows)
    by_race = Counter(row.get("race") or "unknown" for row in rows)
    by_slot_gender_race: dict[str, int] = defaultdict(int)

    for row in rows:
        key = "|".join(
            [
                row.get("slot") or "unknown",
                row.get("model_gender") or "unknown",
                row.get("race") or "unknown",
            ]
        )
        by_slot_gender_race[key] += 1

    return {
        "image_count": len(rows),
        "by_kind": dict(sorted(by_kind.items())),
        "by_slot": dict(sorted(by_slot.items())),
        "by_part": dict(sorted(by_part.items())),
        "by_role": dict(sorted(by_role.items())),
        "by_gender": dict(sorted(by_gender.items())),
        "by_race": dict(sorted(by_race.items())),
        "by_slot_gender_race": dict(sorted(by_slot_gender_race.items())),
    }


def summary_to_text(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
