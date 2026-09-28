"""Image-level eligibility and update evidence, independent of report metadata."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from collections import Counter
from pathlib import Path
from typing import Any

from ffxiv_prism.jsonl import read_jsonl, write_jsonl_atomic
from ffxiv_prism.collectors.eorzea_collection.image_url import is_full_preview_image
from ffxiv_prism.paths import resolve_project_path

SLOTS = {"HEAD", "BODY", "HANDS", "LEGS", "FEET"}
SAMPLE_PATH = Path("data/processed/equipment_image_samples.jsonl")
META_PATH = Path("data/raw/eorzea_collection/metadata")


def sample_rows(database: Path) -> list[dict[str, Any]]:
    """Require decodable images and source equipment labels; names/dyes are optional.

    Duplicates are collapsed only when both bytes and equipment labels agree.
    Training still requires independent quality review and leakage-safe splitting.
    """
    with closing(sqlite3.connect(database)) as conn:
        conn.row_factory = sqlite3.Row
        images = [dict(row) for row in conn.execute("SELECT * FROM images ORDER BY image_row_id")]
        links: dict[int, list[dict[str, Any]]] = {}
        for row in conn.execute("SELECT image_row_id, piece_id, link_type FROM gear_piece_images ORDER BY piece_id"):
            links.setdefault(row["image_row_id"], []).append(dict(row))
    seen = set()
    result = []
    for row in images:
        labels = links.get(row["image_row_id"], [])
        exact = [label for label in labels if label["link_type"] == "exact_slot"]
        labels = exact or labels
        piece_ids = sorted({label["piece_id"] for label in labels})
        path = resolve_project_path(row.get("local_path"))
        status = "eligible"
        if row.get("qa_status") != "ready" or path is None or not path.is_file():
            status = "quality_rejected"
        elif row.get("slot") not in SLOTS or row.get("dye_state") == "dyed":
            status = "unsupported_view"
        elif not piece_ids:
            status = "pending_labels"
        key = (row.get("sha256") or row["source_image_url"], row.get("slot"), tuple(piece_ids))
        if status == "eligible":
            if key in seen:
                status = "duplicate"
            else:
                seen.add(key)
        result.append({
            "source_image_url": row["source_image_url"], "local_path": row["local_path"],
            "sha256": row.get("sha256"), "slot": row.get("slot"),
            "piece_ids": piece_ids, "eligibility_status": status,
            "training_status": "pending_review_and_split" if status == "eligible" else "not_eligible",
        })
    return result


def build_sample_manifest(root: Path) -> dict[str, Any]:
    rows = sample_rows(root / "data/processed/eorzea.sqlite")
    write_jsonl_atomic(root / SAMPLE_PATH, rows)
    return {"images": len(rows), "by_status": dict(Counter(row["eligibility_status"] for row in rows)),
            "training_admitted_images": None, "training_status": "not_evaluated"}


def image_snapshot(root: Path, *, include_full_previews: bool = False) -> dict[str, list[str]]:
    discovered_rows = [row for row in read_jsonl(root / META_PATH / "image_url_manifest_raw.jsonl")
                       if row.get("source_image_url") and row.get("accessible", True)
                       and (row.get("image_variant") or row.get("dye_state") or "normal") == "normal"]
    discovered = {row["source_image_url"] for row in discovered_rows}
    download_required = {row["source_image_url"] for row in discovered_rows
                         if include_full_previews or not is_full_preview_image(
                             row["source_image_url"], row.get("image_part"))}
    downloaded = set()
    for row in read_jsonl(root / META_PATH / "image_manifest_raw.jsonl"):
        path = resolve_project_path(row.get("local_path"))
        if row.get("source_image_url") and path and path.is_file():
            downloaded.add(row["source_image_url"])
    database = root / "data/processed/eorzea.sqlite"
    samples = sample_rows(database) if database.is_file() else []
    states = {state: {r["source_image_url"] for r in samples if r["eligibility_status"] == state}
              for state in ("eligible", "pending_labels", "duplicate", "quality_rejected", "unsupported_view")}
    qa_ready = set()
    if database.is_file():
        with closing(sqlite3.connect(database)) as conn:
            for url, path in conn.execute("SELECT source_image_url, local_path FROM images WHERE qa_status = 'ready'"):
                resolved = resolve_project_path(path)
                if resolved and resolved.is_file():
                    qa_ready.add(url)
    return {key: sorted(value) for key, value in {"discovered": discovered, "download_required": download_required, "downloaded": downloaded,
            "qa_ready": qa_ready, **states}.items()}


def update_result(before: dict, after: dict, *, candidate: dict, download: dict,
                  warnings: int, failed: bool, planned: bool) -> dict[str, Any]:
    new = lambda key: len(set(after.get(key, [])) - set(before.get(key, [])))
    indexed = int(candidate.get("new_indexed_images") or 0) if candidate.get("activated") else 0
    required = after.get("download_required", after.get("discovered", []))
    missing = len(set(required) - set(after.get("downloaded", [])))
    if planned:
        outcome, label = "planned", "更新计划（未执行）"
    elif failed:
        outcome, label = "failed", "图片更新未完成，请查看失败阶段"
    elif indexed:
        outcome, label = "images_indexed", f"新增 {indexed} 张图片已进入检索"
    elif new("eligible"):
        outcome, label = "images_pending_index", "新增合格图片尚未进入检索索引"
    else:
        outcome, label = "no_new_samples", "无新增有效检索样本（可能仅更新元数据）"
    if not planned and not failed and (warnings or missing or download.get("download_failures")):
        label += "；部分下载或补充资料待处理"
    return {
        "outcome": outcome, "message": label,
        "new_discovered_images": new("discovered"),
        "new_local_images": new("downloaded"),
        "images_downloaded": download.get("images_downloaded"),
        "download_failures": download.get("download_failures"),
        "missing_download_images": missing,
        "new_qa_ready_images": new("qa_ready"), "new_eligible_images": new("eligible"),
        "new_indexed_images": indexed,
        "pending_label_images": len(after.get("pending_labels", [])),
        "quality_rejected_images": len(after.get("quality_rejected", [])),
        "duplicate_images": len(after.get("duplicate", [])),
        "training_candidate_images": len(after.get("eligible", [])),
        "training_admitted_images": None, "training_status": "not_evaluated",
        "metadata_warnings": warnings,
    }
