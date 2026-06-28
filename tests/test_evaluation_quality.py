from __future__ import annotations

import unittest

from ffxiv_prism.vision.evaluation_quality import assess_front_full_body


def pose_rows(points: dict[str, tuple[float, float]]) -> list[dict[str, float | str]]:
    return [
        {"name": name, "x": x, "y": y, "score": 0.9}
        for name, (x, y) in points.items()
    ]


FRONTAL = {
    "nose": (170, 60),
    "left_shoulder": (220, 140),
    "right_shoulder": (120, 140),
    "left_wrist": (245, 270),
    "right_wrist": (95, 270),
    "left_hip": (205, 290),
    "right_hip": (135, 290),
    "left_knee": (205, 405),
    "right_knee": (135, 405),
    "left_ankle": (205, 535),
    "right_ankle": (135, 535),
}


class EvaluationQualityTests(unittest.TestCase):
    def test_accepts_large_frontal_full_body_pose(self) -> None:
        result = assess_front_full_body(
            {
                "person_box": [70, 20, 270, 555],
                "keypoints": pose_rows(FRONTAL),
            },
            (340, 570),
        )

        self.assertTrue(result.passed)
        self.assertGreater(result.quality_score, 0.9)

    def test_rejects_side_view_and_missing_lower_body(self) -> None:
        points = dict(FRONTAL)
        points["left_shoulder"] = (178, 140)
        points["right_shoulder"] = (160, 144)
        points["left_ankle"] = (180, 330)
        points["right_ankle"] = (160, 335)
        result = assess_front_full_body(
            {
                "person_box": [120, 30, 220, 420],
                "keypoints": pose_rows(points),
            },
            (1040, 570),
        )

        self.assertFalse(result.passed)
        self.assertIn("likely_side_view", result.rejection_reasons)
        self.assertIn("not_full_body", result.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
