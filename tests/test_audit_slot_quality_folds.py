from __future__ import annotations

import unittest

from scripts.audit_slot_quality_folds import (
    audit_crop_coverage,
    fold_gate,
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


if __name__ == "__main__":
    unittest.main()
