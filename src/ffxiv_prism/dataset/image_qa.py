from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image

from ffxiv_prism.jsonl import (
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from ffxiv_prism.paths import PROCESSED_DIR, ensure_project_dirs, portable_path, resolve_project_path


DEFAULT_INDEX_PATH = PROCESSED_DIR / "eorzea_images_index.jsonl"
DEFAULT_QA_INDEX_PATH = PROCESSED_DIR / "eorzea_images_qa_index.jsonl"
DEFAULT_QA_SUMMARY_PATH = PROCESSED_DIR / "eorzea_image_qa_summary.json"
DEFAULT_MISSING_MANIFEST_PATH = PROCESSED_DIR / "eorzea_missing_images_manifest.jsonl"


def build_image_qa_index(
    index_path: Path | None = None,
    output_path: Path | None = None,
    summary_path: Path | None = None,
    missing_manifest_path: Path | None = None,
    min_width: int = 64,
    min_height: int = 64,
) -> dict[str, Any]:
    """Validate local image files referenced by the processed image index."""

    ensure_project_dirs()
    index_path = index_path or DEFAULT_INDEX_PATH
    output_path = output_path or DEFAULT_QA_INDEX_PATH
    summary_path = summary_path or DEFAULT_QA_SUMMARY_PATH
    missing_manifest_path = missing_manifest_path or DEFAULT_MISSING_MANIFEST_PATH

    rows = read_jsonl(index_path)
    sha_counts = Counter(row.get("sha256") for row in rows if row.get("sha256"))
    path_counts = Counter(normalized_local_path(row) for row in rows if row.get("local_path"))

    qa_rows = [
        annotate_image_row(
            row,
            sha_duplicate_count=sha_counts.get(row.get("sha256"), 0),
            path_reference_count=path_counts.get(normalized_local_path(row), 0),
            min_width=min_width,
            min_height=min_height,
        )
        for row in rows
    ]

    write_jsonl_atomic(output_path, qa_rows)
    write_jsonl_atomic(
        missing_manifest_path,
        [row for row in qa_rows if row["qa_status"] == "missing"],
    )

    summary = summarize_qa_rows(qa_rows)
    summary.update(
        {
            "index": portable_path(index_path),
            "output": portable_path(output_path),
            "missing_manifest": portable_path(missing_manifest_path),
            "min_width": min_width,
            "min_height": min_height,
        }
    )
    write_json_atomic(summary_path, summary)
    return summary


def normalized_local_path(row: dict[str, Any]) -> str | None:
    return portable_path(resolve_project_path(row.get("local_path")))


def annotate_image_row(
    row: dict[str, Any],
    sha_duplicate_count: int = 0,
    path_reference_count: int = 0,
    min_width: int = 64,
    min_height: int = 64,
) -> dict[str, Any]:
    qa = dict(row)
    local_path = row.get("local_path")
    path = resolve_project_path(local_path)
    if path:
        qa["local_path"] = portable_path(path)
    flags: list[str] = []

    qa["file_exists"] = False
    qa["readable_image"] = False
    qa["actual_byte_size"] = None
    qa["width"] = None
    qa["height"] = None
    qa["byte_size_matches_manifest"] = None
    qa["sha_duplicate_count"] = sha_duplicate_count
    qa["path_reference_count"] = path_reference_count

    if not path:
        flags.append("missing_local_path")
        qa["qa_status"] = "missing"
        qa["quality_flags"] = flags
        return qa

    if not path.exists():
        flags.append("missing_file")
        qa["qa_status"] = "missing"
        qa["quality_flags"] = flags
        return qa

    qa["file_exists"] = True
    actual_byte_size = path.stat().st_size
    qa["actual_byte_size"] = actual_byte_size

    expected_byte_size = row.get("byte_size")
    if isinstance(expected_byte_size, int):
        qa["byte_size_matches_manifest"] = actual_byte_size == expected_byte_size
        if actual_byte_size != expected_byte_size:
            flags.append("byte_size_mismatch")

    try:
        width, height, image_format = read_image_dimensions(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (ImageHeaderError, OSError, ValueError, SyntaxError) as exc:
        flags.append("invalid_image")
        qa["image_format"] = None
        qa["image_error"] = str(exc)
        qa["qa_status"] = "invalid"
        qa["quality_flags"] = flags
        return qa

    qa["readable_image"] = True
    qa["width"] = width
    qa["height"] = height
    qa["image_format"] = image_format
    qa["image_error"] = None

    if width < min_width or height < min_height:
        flags.append("too_small")
    if sha_duplicate_count > 1:
        flags.append("duplicate_sha256")
    if path_reference_count > 1:
        flags.append("duplicate_local_path")

    qa["qa_status"] = "ready" if not any(flag in flags for flag in ("too_small",)) else "flagged"
    qa["quality_flags"] = flags
    return qa


def read_image_dimensions(path: Path) -> tuple[int, int, str]:
    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return read_png_dimensions(header)
        if header.startswith(b"\xff\xd8"):
            handle.seek(0)
            return read_jpeg_dimensions(handle)
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            handle.seek(0)
            return read_webp_dimensions(handle)
        if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return read_gif_dimensions(header)
    raise ImageHeaderError("unsupported or unrecognized image header")


def read_png_dimensions(header: bytes) -> tuple[int, int, str]:
    if len(header) < 24:
        raise ImageHeaderError("truncated PNG header")
    width, height = struct.unpack(">II", header[16:24])
    return width, height, "png"


def read_gif_dimensions(header: bytes) -> tuple[int, int, str]:
    if len(header) < 10:
        raise ImageHeaderError("truncated GIF header")
    width, height = struct.unpack("<HH", header[6:10])
    return width, height, "gif"


def read_jpeg_dimensions(handle: BinaryIO) -> tuple[int, int, str]:
    if handle.read(2) != b"\xff\xd8":
        raise ImageHeaderError("missing JPEG SOI marker")

    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while True:
        marker_start = handle.read(1)
        if not marker_start:
            break
        if marker_start != b"\xff":
            continue

        marker = handle.read(1)
        while marker == b"\xff":
            marker = handle.read(1)
        if not marker:
            break

        marker_id = marker[0]
        if marker_id in (0xD8, 0xD9):
            continue

        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            break
        segment_length = struct.unpack(">H", length_bytes)[0]
        if segment_length < 2:
            raise ImageHeaderError("invalid JPEG segment length")

        if marker_id in sof_markers:
            payload = handle.read(5)
            if len(payload) != 5:
                break
            height, width = struct.unpack(">HH", payload[1:5])
            return width, height, "jpeg"

        handle.seek(segment_length - 2, 1)

    raise ImageHeaderError("JPEG dimensions not found")


def read_webp_dimensions(handle: BinaryIO) -> tuple[int, int, str]:
    header = handle.read(30)
    if len(header) < 30:
        raise ImageHeaderError("truncated WebP header")

    chunk_type = header[12:16]
    if chunk_type == b"VP8X":
        width = int.from_bytes(header[24:27], "little") + 1
        height = int.from_bytes(header[27:30], "little") + 1
        return width, height, "webp"

    if chunk_type == b"VP8 ":
        width = struct.unpack("<H", header[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", header[28:30])[0] & 0x3FFF
        return width, height, "webp"

    if chunk_type == b"VP8L":
        signature = header[20]
        if signature != 0x2F:
            raise ImageHeaderError("invalid WebP lossless signature")
        b0, b1, b2, b3 = header[21:25]
        width = 1 + (((b1 & 0x3F) << 8) | b0)
        height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
        return width, height, "webp"

    raise ImageHeaderError(f"unsupported WebP chunk {chunk_type!r}")


def summarize_qa_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(row.get("qa_status") or "unknown" for row in rows)
    by_flag: Counter[str] = Counter()
    by_format = Counter(row.get("image_format") or "unknown" for row in rows)
    missing_examples: list[dict[str, Any]] = []
    invalid_examples: list[dict[str, Any]] = []

    for row in rows:
        by_flag.update(row.get("quality_flags") or [])
        if row.get("qa_status") == "missing" and len(missing_examples) < 20:
            missing_examples.append(example_row(row))
        if row.get("qa_status") == "invalid" and len(invalid_examples) < 20:
            invalid_examples.append(example_row(row))

    ready_rows = [row for row in rows if row.get("qa_status") == "ready"]
    return {
        "image_rows": len(rows),
        "ready_rows": len(ready_rows),
        "file_exists_rows": sum(1 for row in rows if row.get("file_exists")),
        "readable_image_rows": sum(1 for row in rows if row.get("readable_image")),
        "missing_rows": by_status.get("missing", 0),
        "invalid_rows": by_status.get("invalid", 0),
        "flagged_rows": by_status.get("flagged", 0),
        "duplicate_sha256_rows": by_flag.get("duplicate_sha256", 0),
        "duplicate_local_path_rows": by_flag.get("duplicate_local_path", 0),
        "byte_size_mismatch_rows": by_flag.get("byte_size_mismatch", 0),
        "too_small_rows": by_flag.get("too_small", 0),
        "by_status": dict(sorted(by_status.items())),
        "by_flag": dict(sorted(by_flag.items())),
        "by_format": dict(sorted(by_format.items())),
        "missing_examples": missing_examples,
        "invalid_examples": invalid_examples,
    }


def example_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "gearset_slug": row.get("gearset_slug"),
        "image_part": row.get("image_part"),
        "dye_state": row.get("dye_state"),
        "local_path": portable_path(resolve_project_path(row.get("local_path"))),
        "source_image_url": row.get("source_image_url"),
    }


def summary_to_text(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)


class ImageHeaderError(ValueError):
    pass
