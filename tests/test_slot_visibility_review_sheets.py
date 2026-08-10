from __future__ import annotations

import unittest

from scripts.build_slot_visibility_review_sheets import (
    deterministic_example,
    select_preferred_examples,
)


class SlotVisibilityReviewSheetTests(unittest.TestCase):
    def test_deterministic_example_is_order_independent(self) -> None:
        rows = [
            {"example_id": "10:2", "image_position": 2},
            {"example_id": "10:0", "image_position": 0},
            {"example_id": "10:1", "image_position": 1},
        ]

        selected = deterministic_example(rows, 20260730, "10")
        reversed_selected = deterministic_example(
            list(reversed(rows)),
            20260730,
            "10",
        )

        self.assertEqual(selected["example_id"], reversed_selected["example_id"])

    def test_deterministic_example_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty glamour"):
            deterministic_example([], 1, "10")

    def test_select_preferred_examples_preserves_queue_order(self) -> None:
        by_glamour = {
            "10": [
                {"example_id": "10:0"},
                {"example_id": "10:1"},
            ],
            "20": [{"example_id": "20:0"}],
        }

        selected = select_preferred_examples(
            by_glamour,
            [
                {"example_id": "20:0"},
                {"example_id": "10:1"},
                {"example_id": "10:0"},
            ],
        )

        self.assertEqual(list(selected), ["20", "10"])
        self.assertEqual(
            [row["example_id"] for row in selected["10"]],
            ["10:1", "10:0"],
        )


if __name__ == "__main__":
    unittest.main()
