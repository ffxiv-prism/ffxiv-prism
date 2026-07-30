from __future__ import annotations

import unittest

from ffxiv_prism.vision.outfit_regions import CropRegion
from scripts.build_pose_slot_visibility_manifest import summarize_slot_regions


class PoseSlotVisibilityManifestTests(unittest.TestCase):
    def test_hands_require_both_visible_regions_for_clear(self) -> None:
        result = summarize_slot_regions(
            "HANDS",
            [
                CropRegion(
                    "HANDS",
                    "hands_left",
                    (0, 0, 40, 80),
                    0.9,
                    "visible",
                ),
                CropRegion(
                    "HANDS",
                    "hands_right",
                    (50, 0, 90, 80),
                    0.8,
                    "partially_visible",
                ),
            ],
        )

        self.assertEqual(result["decision"], "partial")

    def test_small_or_missing_regions_are_unusable(self) -> None:
        small = summarize_slot_regions(
            "FEET",
            [
                CropRegion(
                    "FEET",
                    "feet_left",
                    (0, 0, 10, 40),
                    0.9,
                    "visible",
                ),
                CropRegion(
                    "FEET",
                    "feet_right",
                    (20, 0, 30, 40),
                    0.9,
                    "visible",
                ),
            ],
        )
        missing = summarize_slot_regions(
            "HEAD",
            [],
        )

        self.assertEqual(small["decision"], "unusable")
        self.assertEqual(missing["decision"], "unusable")


if __name__ == "__main__":
    unittest.main()
