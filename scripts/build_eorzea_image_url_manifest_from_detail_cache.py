# 功能：从已缓存 gearset 详情页中真实暴露的静态图片 URL 重建 URL 表，不生成盲测候选。
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.collector import timestamp  # noqa: E402
from ffxiv_prism.collectors.eorzea_collection.image_url import parse_gearsets_image_url  # noqa: E402
from ffxiv_prism.collectors.eorzea_collection.parser import parse_html  # noqa: E402
from ffxiv_prism.jsonl import append_jsonl  # noqa: E402
from ffxiv_prism.paths import EORZEA_HTML_DIR, EORZEA_METADATA_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build image URL manifest from cached gearset detail HTML.")
    parser.add_argument(
        "--detail-cache-dir",
        type=Path,
        default=EORZEA_HTML_DIR / "gearset",
        help="Directory containing cached gearset detail HTML pages.",
    )
    parser.add_argument(
        "--checks-output",
        type=Path,
        default=EORZEA_METADATA_DIR / "image_url_checks_raw.jsonl",
        help="Output JSONL for source-backed URL rows.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=EORZEA_METADATA_DIR / "image_url_manifest_raw.jsonl",
        help="Output JSONL for normal image URLs used by downloads.",
    )
    args = parser.parse_args()

    rows = discover_detail_image_rows(args.detail_cache_dir)
    manifest_rows = build_manifest_rows(rows)

    args.checks_output.parent.mkdir(parents=True, exist_ok=True)
    args.checks_output.write_text("", encoding="utf-8")
    args.manifest_output.write_text("", encoding="utf-8")
    append_jsonl(args.checks_output, rows)
    append_jsonl(args.manifest_output, manifest_rows)

    print(
        json.dumps(
            {
                "detail_page_count": len(list(args.detail_cache_dir.glob("*.html"))),
                "checks_url_count": len(rows),
                "manifest_url_count": len(manifest_rows),
                "dye_true_count": sum(1 for row in manifest_rows if row.get("dye") is True),
                "dye_unknown_count": sum(1 for row in manifest_rows if row.get("dye") == "unknown"),
                "checks_output": str(args.checks_output),
                "manifest_output": str(args.manifest_output),
                "finished_at": timestamp(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def discover_detail_image_rows(cache_dir: Path) -> list[dict[str, object]]:
    rows_by_url: dict[str, dict[str, object]] = {}
    checked_at = timestamp()
    for path in sorted(cache_dir.glob("*.html")):
        slug = path.stem
        page_url = f"https://ffxiv.eorzeacollection.com/gearset/{slug}"
        parsed = parse_html(path.read_text(encoding="utf-8", errors="replace"), page_url)
        for image in parsed.images:
            for url in image.urls:
                meta = parse_gearsets_image_url(url)
                if meta is None or url in rows_by_url:
                    continue
                rows_by_url[url] = {
                    "source_image_url": url,
                    "source_page_url": page_url,
                    "accessible": True,
                    "checked_at": checked_at,
                    "status": None,
                    "content_type": None,
                    "content_length": None,
                    "error": None,
                    "verification_method": "source_html_reference",
                    "image_kind": "gearset_static",
                    "slot": meta.slot,
                    "gearset_slug": meta.gearset_slug,
                    "glamour_slug": None,
                    "piece_name": None,
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
    return sorted(rows_by_url.values(), key=lambda row: str(row["source_image_url"]))


def build_manifest_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    dyed_keys = {
        image_dye_key(row)
        for row in rows
        if row.get("accessible") and row.get("image_variant") == "dyed"
    }
    manifest_rows: list[dict[str, object]] = []
    for row in rows:
        if not row.get("accessible") or row.get("image_variant") != "normal":
            continue
        manifest_row = dict(row)
        manifest_row["dye"] = True if image_dye_key(row) in dyed_keys else "unknown"
        manifest_rows.append(manifest_row)
    return manifest_rows


def image_dye_key(row: dict[str, object]) -> tuple[object, object, object, object, object]:
    return (
        row.get("source_path"),
        row.get("gearset_slug"),
        row.get("race"),
        row.get("model_gender"),
        row.get("image_part"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
