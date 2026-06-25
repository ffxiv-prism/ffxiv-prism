from __future__ import annotations

import importlib.util
import io
import unittest
from pathlib import Path

from PIL import Image

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_eorzea_glamours.py"
SPEC = importlib.util.spec_from_file_location("collect_eorzea_glamours", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GlamourCollectorTests(unittest.TestCase):
    def test_inspect_image_accepts_screenshot(self) -> None:
        payload = png_bytes((340, 570))
        info = MODULE.inspect_image_bytes(payload, "test.png")
        self.assertEqual(info, {"width": 340, "height": 570, "image_format": "png"})

    def test_inspect_image_rejects_icon(self) -> None:
        with self.assertRaisesRegex(ValueError, "likely icon"):
            MODULE.inspect_image_bytes(png_bytes((48, 48)), "icon.png")

    def test_validate_record_rejects_other_glamour_images(self) -> None:
        record = {
            "glamour_id": 10,
            "image_urls": ["https://glamours.eorzeacollection.com/11/icon.png"],
            "equipment": [{"slot": slot} for slot in MODULE.CORE_SLOTS],
        }
        with self.assertRaisesRegex(ValueError, "outside glamour"):
            MODULE.validate_record(record)

    def test_require_all_core_items_matched_rejects_unmatched_nonempty_slot(self) -> None:
        record = {
            "equipment": [
                {
                    "slot": "BODY",
                    "is_core_slot": True,
                    "is_empty": False,
                    "item_name": "Missing Coat",
                    "matched_piece_ids": [99],
                    "piece_match_type": "appearance_equivalent",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "BODY=Missing Coat"):
            MODULE.require_all_core_items_matched(record)

    def test_require_all_core_items_matched_allows_empty_slots(self) -> None:
        record = {
            "equipment": [
                {
                    "slot": "HEAD",
                    "is_core_slot": True,
                    "is_empty": True,
                    "item_name": None,
                    "matched_piece_ids": [],
                },
                {
                    "slot": "BODY",
                    "is_core_slot": True,
                    "is_empty": False,
                    "item_name": "Known Coat",
                    "matched_piece_ids": [42],
                },
            ]
        }
        MODULE.require_all_core_items_matched(record)

def png_bytes(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, "PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
