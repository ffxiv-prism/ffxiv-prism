from __future__ import annotations

import unittest

import numpy as np

from scripts.build_hands_pose_tube_crops import arm_tube_mask


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


if __name__ == "__main__":
    unittest.main()
