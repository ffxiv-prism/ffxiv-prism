from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from ffxiv_prism.matching.coverage_profiles import (
    _head_parsing_spatial_profile,
)
from ffxiv_prism.vision.human_parsing import (
    apply_human_mask,
    build_global_foreground_mask,
    composite_foreground,
    transparent_foreground,
)


class HumanParsingTests(unittest.TestCase):
    def test_head_parsing_profile_is_scale_normalized_and_unit_length(self) -> None:
        segmentation = np.zeros((12, 8), dtype=np.uint8)
        segmentation[2:10, 2:6] = 4
        segmentation[3:6, 2:6] = 2
        segmentation[6:8, 3:5] = 11

        profile = _head_parsing_spatial_profile(segmentation, 8)

        self.assertEqual(profile.shape, (8 * 8 * 3,))
        self.assertAlmostEqual(float(np.linalg.norm(profile)), 1.0, places=5)

    def test_person_mask_removes_background(self) -> None:
        image = Image.new("RGB", (4, 4), "white")
        segmentation = np.zeros((4, 4), dtype=np.uint8)
        segmentation[1:3, 1:3] = 4

        result = apply_human_mask(
            image,
            segmentation,
            mode="person",
            slot="BODY",
            dilation=1,
            feather=0,
        )

        self.assertEqual(result.image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.image.getpixel((1, 1)), (255, 255, 255))

    def test_slot_mask_falls_back_when_target_is_missing(self) -> None:
        image = Image.new("RGB", (4, 4), "white")
        segmentation = np.full((4, 4), 4, dtype=np.uint8)

        result = apply_human_mask(
            image,
            segmentation,
            mode="slot",
            slot="FEET",
            dilation=1,
            feather=0,
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.selected_fraction, 1.0)

    def test_global_mask_uses_original_coordinates_and_pose_envelope(self) -> None:
        image = Image.new("RGB", (20, 20), "white")
        segmentation = np.zeros((20, 20), dtype=np.uint8)
        segmentation[5:15, 8:12] = 4
        keypoints = [
            {"x": 10, "y": y, "score": 0.9}
            for y in range(3, 18)
        ]

        result = build_global_foreground_mask(
            image,
            segmentation,
            keypoints=keypoints,
            dilation_fraction=0,
            feather=0,
        )

        self.assertEqual(result.mask.size, image.size)
        self.assertEqual(result.keypoint_coverage, 1.0)
        self.assertTrue(result.used_pose_envelope)
        self.assertGreater(result.foreground_fraction, result.person_fraction)

    def test_global_mask_can_be_composited_or_kept_transparent(self) -> None:
        image = Image.new("RGB", (4, 4), "red")
        mask = Image.new("L", (4, 4), 0)
        mask.putpixel((1, 1), 255)

        composite = composite_foreground(image, mask, (127, 127, 127))
        rgba = transparent_foreground(image, mask)

        self.assertEqual(composite.getpixel((0, 0)), (127, 127, 127))
        self.assertEqual(composite.getpixel((1, 1)), (255, 0, 0))
        self.assertEqual(rgba.getpixel((0, 0))[3], 0)
        self.assertEqual(rgba.getpixel((1, 1))[3], 255)

    def test_global_mask_repairs_enclosed_parser_holes(self) -> None:
        image = Image.new("RGB", (12, 12), "white")
        segmentation = np.zeros((12, 12), dtype=np.uint8)
        segmentation[2:10, 2:10] = 4
        segmentation[4:8, 4:8] = 0

        result = build_global_foreground_mask(
            image,
            segmentation,
            dilation_fraction=0,
            feather=0,
        )

        self.assertEqual(result.mask.getpixel((5, 5)), 255)
        self.assertGreater(result.repaired_hole_fraction, 0.0)
