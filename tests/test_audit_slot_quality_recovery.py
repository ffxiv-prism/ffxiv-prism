from __future__ import annotations

import unittest

from scripts.audit_slot_quality_recovery import (
    build_eligibility_rows,
    summarize_slot_coverage,
    validate_reviews,
)


def example(example_id: str, glamour_id: str, piece_id: int) -> dict:
    return {
        "example_id": example_id,
        "glamour_id": glamour_id,
        "model_race": "Roegadyn",
        "core_slot_labels": {
            "HEAD": {"piece_ids": [piece_id], "item_name": "Hat"}
        },
    }


class AuditSlotQualityRecoveryTests(unittest.TestCase):
    def test_review_validation_is_exact_and_fail_closed(self) -> None:
        queue = [
            {
                "example_id": "2:0",
                "recoverable_slots": ["HEAD", "BODY"],
            }
        ]
        reviews = [
            {
                "example_id": "2:0",
                "global_decision": "accept",
                "slot_decisions": {"HEAD": "usable"},
            }
        ]

        errors, accepted = validate_reviews(queue, reviews)

        self.assertEqual(errors["invalid_slot_key_ids"], ["2:0"])
        self.assertEqual(accepted, {})

    def test_rejected_image_cannot_expose_usable_slot(self) -> None:
        queue = [{"example_id": "2:0", "recoverable_slots": ["HEAD"]}]
        reviews = [
            {
                "example_id": "2:0",
                "global_decision": "reject",
                "slot_decisions": {"HEAD": "usable"},
            }
        ]

        errors, accepted = validate_reviews(queue, reviews)

        self.assertEqual(errors["rejected_with_usable_slot_ids"], ["2:0"])
        self.assertEqual(accepted, {})

    def test_recovery_converts_identity_to_independent_glamours(self) -> None:
        examples = [example("1:0", "1", 7), example("2:0", "2", 7)]
        quality = [
            {"example_id": "1:0", "passed": True},
            {"example_id": "2:0", "passed": False},
        ]
        baseline = build_eligibility_rows(examples, quality, {})
        recovered = build_eligibility_rows(
            examples,
            quality,
            {"2:0": {"HEAD"}},
        )

        before = summarize_slot_coverage(examples, baseline)["HEAD"]
        after = summarize_slot_coverage(examples, recovered)["HEAD"]

        self.assertEqual(before["multi_glamour_identity_count"], 0)
        self.assertEqual(after["multi_glamour_identity_count"], 1)
        self.assertEqual(after["multi_glamour_identity_rate"], 1.0)

    def test_known_unusable_slot_and_rejected_image_are_removed_everywhere(self) -> None:
        examples = [
            example("1:0", "1", 7),
            example("2:0", "2", 8),
        ]
        quality = [
            {"example_id": "1:0", "passed": True},
            {"example_id": "2:0", "passed": True},
        ]
        reviews = {
            "1:0": {
                "decision": "accept",
                "slot_usability": {"HEAD": "unusable"},
            },
            "2:0": {"decision": "reject"},
        }

        rows = build_eligibility_rows(examples, quality, {}, reviews)

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
