from __future__ import annotations

import unittest

from scripts.build_slot_quality_recovery_queue import (
    build_recovery_rows,
    recoverable_for_slot,
)


class BuildSlotQualityRecoveryQueueTests(unittest.TestCase):
    def test_feet_crop_can_recover_upper_slots_only(self) -> None:
        reasons = ["feet_likely_cropped"]
        self.assertTrue(recoverable_for_slot(reasons, "HEAD"))
        self.assertTrue(recoverable_for_slot(reasons, "BODY"))
        self.assertTrue(recoverable_for_slot(reasons, "HANDS"))
        self.assertFalse(recoverable_for_slot(reasons, "LEGS"))
        self.assertFalse(recoverable_for_slot(reasons, "FEET"))

    def test_head_crop_can_recover_lower_slots(self) -> None:
        reasons = ["not_full_body", "head_likely_cropped"]
        self.assertFalse(recoverable_for_slot(reasons, "HEAD"))
        self.assertTrue(recoverable_for_slot(reasons, "BODY"))
        self.assertTrue(recoverable_for_slot(reasons, "HANDS"))
        self.assertTrue(recoverable_for_slot(reasons, "LEGS"))
        self.assertTrue(recoverable_for_slot(reasons, "FEET"))

    def test_visual_reject_is_never_recovered(self) -> None:
        self.assertFalse(
            recoverable_for_slot(
                ["feet_likely_cropped", "visual_review_reject"],
                "HEAD",
            )
        )

    def test_generic_pose_failure_is_not_recovered(self) -> None:
        self.assertFalse(recoverable_for_slot(["not_full_body"], "HEAD"))
        self.assertFalse(
            recoverable_for_slot(["feet_likely_cropped", "likely_side_view"], "HEAD")
        )

    def test_manual_conversion_allows_pose_failures_for_human_review(self) -> None:
        self.assertTrue(
            recoverable_for_slot(
                ["low_required_keypoint_confidence", "likely_side_view"],
                "HEAD",
                policy="manual_conversion",
            )
        )
        self.assertFalse(
            recoverable_for_slot(
                ["head_likely_cropped"],
                "HEAD",
                policy="manual_conversion",
            )
        )

    def test_review_row_includes_item_name(self) -> None:
        examples = [
            {
                "example_id": "1:0",
                "glamour_id": "1",
                "local_path": "one.png",
                "core_slot_labels": {
                    "HEAD": {"piece_ids": [7], "item_name": "Example Hat"}
                },
            },
            {
                "example_id": "2:0",
                "glamour_id": "2",
                "local_path": "two.png",
                "core_slot_labels": {
                    "HEAD": {"piece_ids": [7], "item_name": "Example Hat"}
                },
            },
        ]
        quality = [
            {"example_id": "1:0", "passed": True},
            {
                "example_id": "2:0",
                "passed": False,
                "quality_score": 0.95,
                "rejection_reasons": ["likely_side_view"],
            },
        ]

        rows = build_recovery_rows(
            examples,
            quality,
            min_quality_score=0.9,
            policy="manual_conversion",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["slot_details"]["HEAD"]["item_name"],
            "Example Hat",
        )


if __name__ == "__main__":
    unittest.main()
