from __future__ import annotations

import unittest

from scripts.audit_slot_quality_folds import (
    audit_crop_coverage,
    audit_crop_resolution,
    fold_gate,
    summarize_numeric,
    validate_eligibility,
)


class AuditSlotQualityFoldsTests(unittest.TestCase):
    def test_eligibility_rejects_unknown_slots_and_orphans(self) -> None:
        examples = [{"example_id": "1:0"}]
        rows = [
            {"example_id": "1:0", "eligible_slots": ["HEAD", "UNKNOWN"]},
            {"example_id": "2:0", "eligible_slots": ["BODY"]},
        ]

        errors = validate_eligibility(examples, rows)

        self.assertEqual(errors["invalid_slot_ids"], ["1:0"])
        self.assertEqual(errors["orphan_ids"], ["2:0"])

    def test_fold_gate_requires_every_quantity_and_leakage_check(self) -> None:
        audit = {
            "fold_count": 3,
            "identity_leak_count": 0,
            "exact_content_leak_count": 0,
            "near_content_leak_count": 0,
            "assignment_rotation_issue_count": 0,
            "folds": [
                {
                    "missing_assignment_count": 0,
                    "train_quantity_ok": True,
                    "validation_quantity_ok": True,
                }
                for _ in range(3)
            ],
        }
        self.assertTrue(fold_gate(audit))
        audit["folds"][1]["train_quantity_ok"] = False
        self.assertFalse(fold_gate(audit))

    def test_crop_gate_requires_positive_visibility_for_each_slot(self) -> None:
        eligibility = [
            {"example_id": "1:0", "eligible_slots": ["HEAD", "BODY"]}
        ]
        records = {
            "1:0": [
                {"slot": "HEAD", "visibility": "visible"},
                {"slot": "BODY", "visibility": "pose_uncertain"},
            ]
        }

        result = audit_crop_coverage(eligibility, records)

        self.assertEqual(result["covered_image_slot_pairs"], 1)
        self.assertEqual(result["missing"], [{"example_id": "1:0", "slot": "BODY"}])

    def test_numeric_summary_interpolates_percentiles(self) -> None:
        summary = summarize_numeric([10, 20, 30, 40])

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["median"], 25.0)
        self.assertEqual(summary["p25"], 17.5)

    def test_crop_resolution_uses_per_image_primary_crop_average(self) -> None:
        eligibility = [
            {"example_id": "1:0", "eligible_slots": ["HANDS"]},
            {"example_id": "2:0", "eligible_slots": ["HANDS"]},
        ]
        records = {
            "1:0": [
                {
                    "slot": "HANDS",
                    "role": "primary",
                    "visibility": "visible",
                    "width": 40,
                    "height": 80,
                },
                {
                    "slot": "HANDS",
                    "role": "primary",
                    "visibility": "partially_visible",
                    "width": 60,
                    "height": 100,
                },
            ],
            "2:0": [
                {
                    "slot": "HANDS",
                    "role": "auxiliary",
                    "visibility": "visible",
                    "width": 100,
                    "height": 100,
                }
            ],
        }

        result = audit_crop_resolution(eligibility, records)["slots"]["HANDS"]

        self.assertEqual(result["eligible_image_count"], 2)
        self.assertEqual(result["measured_image_count"], 1)
        self.assertEqual(result["missing_primary"], ["2:0"])
        self.assertEqual(result["primary_short_side_pixels"]["median"], 50.0)
        self.assertEqual(result["primary_crop_pixels"]["median"], 4600.0)


if __name__ == "__main__":
    unittest.main()
