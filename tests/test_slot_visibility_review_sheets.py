from __future__ import annotations

import unittest

from scripts.build_slot_visibility_review_sheets import deterministic_example


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


if __name__ == "__main__":
    unittest.main()
