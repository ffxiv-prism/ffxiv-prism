from __future__ import annotations

import unittest

import numpy as np

from scripts.build_hands_pose_tube_crops import (
    arm_tube_mask,
    build_extent_profile_lookup,
    example_extent_class,
)


class BuildHandsPoseTubeCropsTests(unittest.TestCase):
    def test_mask_tracks_forearm_and_excludes_far_background(self) -> None:
        mask = arm_tube_mask(
            (100, 100),
            {
                "left_shoulder": (20.0, 50.0, 1.0),
                "left_elbow": (45.0, 50.0, 1.0),
                "left_wrist": (70.0, 50.0, 1.0),
            },
            sides=("left",),
            width_ratio=0.4,
            wrist_extension=0.4,
            upper_extension=0.2,
            minimum_score=0.2,
            feather=0.0,
        )
        self.assertIsNotNone(mask)
        array = np.asarray(mask)
        self.assertEqual(int(array[50, 70]), 255)
        self.assertEqual(int(array[50, 78]), 255)
        self.assertEqual(int(array[5, 5]), 0)
        mask.close()

    def test_short_extent_stops_before_elbow_and_sleeve(self) -> None:
        mask = arm_tube_mask(
            (100, 100),
            {
                "left_shoulder": (50.0, 10.0, 1.0),
                "left_elbow": (50.0, 45.0, 1.0),
                "left_wrist": (50.0, 80.0, 1.0),
            },
            sides=("left",),
            width_ratio=0.30,
            wrist_extension=0.30,
            upper_extension=0.25,
            minimum_score=0.2,
            feather=0.0,
            forearm_extent=0.20,
        )
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(mask.getpixel((50, 75)), 0)
        self.assertEqual(mask.getpixel((50, 55)), 0)
        self.assertEqual(mask.getpixel((50, 35)), 0)
        mask.close()

    def test_low_confidence_arm_returns_none(self) -> None:
        mask = arm_tube_mask(
            (20, 20),
            {
                "right_shoulder": (3.0, 3.0, 0.1),
                "right_elbow": (7.0, 7.0, 1.0),
                "right_wrist": (12.0, 12.0, 1.0),
            },
            sides=("right",),
            width_ratio=0.5,
            wrist_extension=0.4,
            upper_extension=0.2,
            minimum_score=0.2,
            feather=0.0,
        )
        self.assertIsNone(mask)

    def test_extent_profiles_are_piece_and_gender_specific(self) -> None:
        lookup = build_extent_profile_lookup(
            [
                {
                    "piece_id": 7,
                    "model_gender": "female",
                    "extent_class": "short",
                    "extent_source": "calibrated_name_prior",
                },
                {
                    "piece_id": 7,
                    "model_gender": "male",
                    "extent_class": "long",
                    "extent_source": "canonical_visual_clip",
                },
            ],
            allowed_sources={"calibrated_name_prior"},
        )
        self.assertEqual(lookup, {(7, "female"): "short"})
        example = {
            "model_gender": "female",
            "core_slot_labels": {"HANDS": {"piece_ids": [7]}},
        }
        self.assertEqual(example_extent_class(example, lookup), "short")

    def test_conflicting_extent_profiles_are_rejected(self) -> None:
        lookup = build_extent_profile_lookup(
            [
                {
                    "piece_id": 9,
                    "model_gender": "female",
                    "extent_class": "short",
                    "extent_source": "calibrated_name_prior",
                },
                {
                    "piece_id": 9,
                    "model_gender": "female",
                    "extent_class": "long",
                    "extent_source": "calibrated_name_prior",
                },
            ]
        )
        self.assertEqual(lookup, {})


if __name__ == "__main__":
    unittest.main()
