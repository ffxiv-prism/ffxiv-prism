from __future__ import annotations

import unittest

from ffxiv_prism.vision.evaluation_quality import (
    assess_front_full_body,
    filter_examples_by_quality,
    summarize_quality_split_coverage,
)


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
    def test_filters_examples_to_passed_quality_rows(self) -> None:
        examples = [
            {"example_id": "10:0"},
            {"example_id": "10:1"},
            {"example_id": "11:0"},
        ]
        quality_rows = [
            {"example_id": "10:0", "passed": True},
            {"example_id": "10:1", "passed": False},
        ]

        filtered = filter_examples_by_quality(examples, quality_rows)

        self.assertEqual(filtered, [{"example_id": "10:0"}])

    def test_summarizes_quality_split_coverage(self) -> None:
        examples = [
            {
                "example_id": "10:0",
                "glamour_id": 10,
                "core_slot_labels": {
                    "BODY": {"item_name": "Coat", "piece_ids": [1]},
                    "HEAD": {"item_name": "Hat", "piece_ids": [2]},
                },
            },
            {
                "example_id": "10:1",
                "glamour_id": 10,
                "core_slot_labels": {"BODY": {"item_name": "Coat", "piece_ids": [1]}},
            },
            {
                "example_id": "20:0",
                "glamour_id": 20,
                "core_slot_labels": {"BODY": {"item_name": "Robe"}},
            },
        ]
        quality_rows = [
            {"example_id": "10:0", "passed": True},
            {"example_id": "10:1", "passed": False, "rejection_reasons": ["back_view"]},
            {"example_id": "20:0", "passed": True},
        ]

        summary = summarize_quality_split_coverage(
            examples,
            quality_rows,
            {"10": "development_test", "20": "train"},
            target_groups=2,
            target_cases=4,
            slots=("HEAD", "BODY"),
        )

        self.assertEqual(summary["accepted_image_count"], 2)
        self.assertTrue(summary["case_policy"]["require_piece_ids"])
        self.assertEqual(summary["rejected_reason_counts"], {"back_view": 1})
        self.assertEqual(summary["splits"]["development_test"]["image_count"], 1)
        self.assertEqual(summary["splits"]["development_test"]["glamour_group_count"], 1)
        self.assertEqual(summary["splits"]["development_test"]["by_slot"]["BODY"], 1)
        self.assertEqual(summary["splits"]["development_test"]["slot_case_count"], 2)
        self.assertEqual(summary["splits"]["train"]["slot_case_count"], 0)
        self.assertFalse(summary["sealed_target"]["ready"])
        self.assertEqual(summary["sealed_target"]["group_gap"], 1)
        self.assertEqual(summary["sealed_target"]["case_gap"], 2)

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
