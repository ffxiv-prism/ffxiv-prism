from __future__ import annotations

import sys
import tempfile
import unittest
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.dataset.image_qa import annotate_image_row, read_image_dimensions
from ffxiv_prism.paths import PROJECT_ROOT as PACKAGE_PROJECT_ROOT, portable_path, resolve_project_path


class ImageQaTests(unittest.TestCase):
    def test_read_png_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            path.write_bytes(make_png(13, 17))

            self.assertEqual(read_image_dimensions(path), (13, 17, "png"))

    def test_read_webp_lossless_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.webp"
            path.write_bytes(make_webp_lossless_header(321, 654))

            self.assertEqual(read_image_dimensions(path), (321, 654, "webp"))

    def test_annotate_missing_image(self) -> None:
        row = {
            "local_path": str(Path("missing.png").resolve()),
            "source_image_url": "https://example.test/missing.png",
            "byte_size": 10,
            "sha256": "abc",
        }

        annotated = annotate_image_row(row)

        self.assertEqual(annotated["qa_status"], "missing")
        self.assertFalse(annotated["file_exists"])
        self.assertIn("missing_file", annotated["quality_flags"])

    def test_annotate_ready_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            data = make_png(130, 170)
            path.write_bytes(data)
            row = {
                "local_path": str(path),
                "source_image_url": "https://example.test/sample.png",
                "byte_size": len(data),
                "sha256": "abc",
            }

            annotated = annotate_image_row(row)

            self.assertEqual(annotated["qa_status"], "ready")
            self.assertTrue(annotated["readable_image"])
            self.assertEqual(annotated["width"], 130)
            self.assertEqual(annotated["height"], 170)

    def test_resolves_historical_project_absolute_path(self) -> None:
        historical = r"C:\Pycproject\ffxiv-prism\data\processed\sample.png"

        self.assertEqual(resolve_project_path(historical), PACKAGE_PROJECT_ROOT / "data" / "processed" / "sample.png")
        self.assertEqual(portable_path(historical), "data/processed/sample.png")

def make_png(width: int, height: int) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_payload = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    ihdr = make_png_chunk(b"IHDR", ihdr_payload)
    iend = make_png_chunk(b"IEND", b"")
    return signature + ihdr + iend


def make_png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + kind + payload + checksum.to_bytes(4, "big")


def make_webp_lossless_header(width: int, height: int) -> bytes:
    width_bits = width - 1
    height_bits = height - 1
    b0 = width_bits & 0xFF
    b1 = ((width_bits >> 8) & 0x3F) | ((height_bits & 0x03) << 6)
    b2 = (height_bits >> 2) & 0xFF
    b3 = (height_bits >> 10) & 0x0F
    payload = bytes([0x2F, b0, b1, b2, b3])
    return b"RIFF" + (len(payload) + 17).to_bytes(4, "little") + b"WEBPVP8L" + (len(payload) + 5).to_bytes(4, "little") + payload + b"\x00" * 5


if __name__ == "__main__":
    unittest.main()


