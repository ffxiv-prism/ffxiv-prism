from __future__ import annotations

import unittest

from scripts.audit_glamour_training_readiness import (
    audit_pose_slot_visibility,
    audit_slot_visibility,
    build_identity_roots,
    hamming_distance,
    identity_for,
    label_piece_ids,
    profile_slot,
    Thresholds,
    wilson_half_width,
    wilson_interval,
)


class TrainingReadinessAuditTests(unittest.TestCase):
    def test_identity_roots_close_overlapping_piece_groups(self) -> None:
        examples = [
            {"core_slot_labels": {"HANDS": {"piece_ids": [1, 2]}}},
            {"core_slot_labels": {"HANDS": {"piece_ids": [2, 3]}}},
        ]
        roots = build_identity_roots(examples, "HANDS")

        self.assertEqual(identity_for(examples[0], "HANDS", roots), roots[1])
        self.assertEqual(identity_for(examples[1], "HANDS", roots), roots[1])
        self.assertEqual(roots[1], roots[3])

    def test_label_piece_ids_filters_invalid_values(self) -> None:
        example = {
            "core_slot_labels": {
                "HEAD": {"piece_ids": [3, "2", 3, 0, None, "bad"]}
            }
        }

        self.assertEqual(label_piece_ids(example, "HEAD"), (2, 3))

    def test_hamming_distance_counts_changed_bits(self) -> None:
        self.assertEqual(hamming_distance(0b1010, 0b0011), 2)

    def test_wilson_precision_improves_with_sample_size(self) -> None:
        small = wilson_half_width(100)
        large = wilson_half_width(1000)

        self.assertIsNotNone(small)
        self.assertIsNotNone(large)
        self.assertGreater(small, large)
        self.assertLess(large, 0.03)

    def test_wilson_interval_reports_observed_usable_uncertainty(self) -> None:
        interval = wilson_interval(98, 100)

        self.assertIsNotNone(interval)
        self.assertGreater(interval[0], 0.92)
        self.assertLess(interval[0], 0.94)

    def test_visibility_audit_counts_only_labelled_slot_rows(self) -> None:
        examples = [
            {
                "example_id": "1:0",
                "core_slot_labels": {"HANDS": {"piece_ids": [10]}},
            },
            {
                "example_id": "2:0",
                "core_slot_labels": {"BODY": {"piece_ids": [20]}},
            },
        ]
        reviews = [
            {
                "example_id": "1:0",
                "slot": "HANDS",
                "visibility": "clear",
                "usability": "usable",
                "issue": "none",
                "review_source": "manual_independent_glamour_audit",
            },
            {
                "example_id": "2:0",
                "slot": "HANDS",
                "visibility": "unusable",
                "usability": "unusable",
                "issue": "wrong_slot",
            },
        ]

        result = audit_slot_visibility(examples, reviews)

        self.assertEqual(result["HANDS"]["labelled_images"], 1)
        self.assertEqual(result["HANDS"]["reviewed_images"], 1)
        self.assertEqual(result["HANDS"]["visibility_counts"], {"clear": 1})
        self.assertEqual(result["HANDS"]["independent_sample_images"], 1)
        self.assertEqual(result["HANDS"]["independent_sample_usable"], 1)
        self.assertEqual(result["BODY"]["reviewed_images"], 0)

    def test_pose_visibility_audit_reports_missing_and_decisions(self) -> None:
        examples = [
            {
                "example_id": "1:0",
                "core_slot_labels": {"HANDS": {"piece_ids": [10]}},
            },
            {
                "example_id": "2:0",
                "core_slot_labels": {"HANDS": {"piece_ids": [11]}},
            },
        ]
        pose_rows = [
            {
                "example_id": "1:0",
                "error": None,
                "slot_visibility": {"HANDS": {"decision": "clear"}},
            }
        ]

        result = audit_pose_slot_visibility(examples, pose_rows)["HANDS"]

        self.assertEqual(result["reviewed_images"], 1)
        self.assertEqual(result["missing_images"], 1)
        self.assertEqual(result["decision_counts"], {"clear": 1})

    def test_slot_profile_counts_independent_glamours_per_identity(self) -> None:
        examples = [
            {
                "example_id": "1:0",
                "glamour_id": "1",
                "core_slot_labels": {"HANDS": {"piece_ids": [10]}},
            },
            {
                "example_id": "1:1",
                "glamour_id": "1",
                "core_slot_labels": {"HANDS": {"piece_ids": [10]}},
            },
            {
                "example_id": "2:0",
                "glamour_id": "2",
                "core_slot_labels": {"HANDS": {"piece_ids": [10]}},
            },
            {
                "example_id": "3:0",
                "glamour_id": "3",
                "core_slot_labels": {"HANDS": {"piece_ids": [20]}},
            },
        ]
        roots = build_identity_roots(examples, "HANDS")

        profile, _ = profile_slot(
            "HANDS",
            examples,
            {10, 20},
            {10, 20},
            [],
            roots,
            Thresholds(),
        )

        self.assertEqual(profile["appearance_identity_count"], 2)
        self.assertEqual(profile["multi_glamour_identity_count"], 1)
        self.assertEqual(profile["single_glamour_identity_count"], 1)
        self.assertEqual(profile["multi_glamour_identity_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
