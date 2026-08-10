from __future__ import annotations

import unittest

from scripts.build_training_visibility_reviews import merge_manual_reviews


class BuildTrainingVisibilityReviewsTests(unittest.TestCase):
    def test_merges_slots_and_promotes_non_game_image_to_global_reject(self) -> None:
        output, summary = merge_manual_reviews(
            [
                {
                    "example_id": "1:0",
                    "slot": "feet",
                    "usability": "unusable",
                    "issue": "non_game_image",
                    "review_source": "manual_a",
                },
                {
                    "example_id": "1:0",
                    "slot": "body",
                    "usability": "usable",
                    "issue": "none",
                    "review_source": "manual_b",
                },
            ]
        )

        self.assertEqual(output[0]["decision"], "reject")
        self.assertEqual(
            output[0]["slot_usability"],
            {"BODY": "usable", "FEET": "unusable"},
        )
        self.assertEqual(summary["globally_rejected_images"], 1)

    def test_repeated_identical_review_is_counted(self) -> None:
        row = {
            "example_id": "1:0",
            "slot": "FEET",
            "usability": "usable",
            "issue": "none",
            "review_source": "manual",
        }

        _, summary = merge_manual_reviews([row, row])

        self.assertEqual(summary["duplicate_rows"], 1)
        self.assertEqual(summary["unique_image_slot_reviews"], 1)

    def test_conflicting_review_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting reviews"):
            merge_manual_reviews(
                [
                    {
                        "example_id": "1:0",
                        "slot": "FEET",
                        "usability": "usable",
                    },
                    {
                        "example_id": "1:0",
                        "slot": "FEET",
                        "usability": "unusable",
                    },
                ]
            )

    def test_source_review_rejects_whole_image(self) -> None:
        output, summary = merge_manual_reviews(
            [],
            [
                {
                    "example_id": "2:0",
                    "decision": "reject",
                    "issue": "reference_art",
                    "review_source": "manual_source_domain_triage",
                }
            ],
        )

        self.assertEqual(output[0]["decision"], "reject")
        self.assertEqual(output[0]["slot_usability"], {})
        self.assertEqual(summary["source_decision_counts"], {"reject": 1})


if __name__ == "__main__":
    unittest.main()
