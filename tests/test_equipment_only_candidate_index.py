from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from scripts.build_equipment_only_candidate_index import (
    build_equipment_only_crop,
    build_template_difference_crop,
)


class EquipmentOnlyCandidateIndexTests(unittest.TestCase):
    def test_removes_non_equipment_pixels(self) -> None:
        image = Image.new("RGB", (20, 20), (240, 200, 180))
        segmentation = np.full((20, 20), 11, dtype=np.uint8)
        segmentation[5:15, 6:14] = 4

        output, stats = build_equipment_only_crop(
            image,
            segmentation,
            slot="BODY",
            normalized_box=(0.0, 0.0, 1.0, 1.0),
            background=127,
        )

        pixels = np.asarray(output)
        self.assertFalse(stats["used_fallback"])
        self.assertTrue(np.all(pixels[0, 0] == 127))
        self.assertFalse(np.all(pixels[10, 10] == 127))
        output.close()
        image.close()

    def test_falls_back_when_parser_finds_no_equipment(self) -> None:
        image = Image.new("RGB", (20, 20), (10, 20, 30))
        segmentation = np.zeros((20, 20), dtype=np.uint8)

        output, stats = build_equipment_only_crop(
            image,
            segmentation,
            slot="FEET",
            normalized_box=(0.0, 0.0, 1.0, 1.0),
        )

        self.assertTrue(stats["used_fallback"])
        self.assertEqual(output.getpixel((0, 0)), (10, 20, 30))
        output.close()
        image.close()

    def test_template_difference_removes_unchanged_mannequin(self) -> None:
        template = Image.new("RGB", (20, 20), (190, 190, 190))
        image = template.copy()
        image.paste((20, 40, 160), (7, 5, 14, 16))

        output, stats = build_template_difference_crop(
            image,
            template,
            normalized_box=(0.0, 0.0, 1.0, 1.0),
            background=127,
            diff_threshold=24,
        )

        pixels = np.asarray(output)
        self.assertFalse(stats["used_fallback"])
        self.assertTrue(np.all(pixels[0, 0] == 127))
        self.assertFalse(np.all(pixels[10, 10] == 127))
        output.close()
        image.close()
        template.close()


if __name__ == "__main__":
    unittest.main()
