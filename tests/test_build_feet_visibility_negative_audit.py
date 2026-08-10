from __future__ import annotations

import unittest

from scripts.build_feet_visibility_negative_audit import (
    select_rule_negative_sample,
    stable_sample_key,
)


class BuildFeetVisibilityNegativeAuditTests(unittest.TestCase):
    def test_selects_only_rule_negative_distinct_glamours(self) -> None:
        rows = [
            {"example_id": "a:0", "glamour_id": "a", "triage_for_review": False},
            {"example_id": "a:1", "glamour_id": "a", "triage_for_review": False},
            {"example_id": "b:0", "glamour_id": "b", "triage_for_review": True},
            {"example_id": "c:0", "glamour_id": "c", "triage_for_review": False},
            {"example_id": "d:0", "glamour_id": "d", "triage_for_review": False},
        ]

        selected = select_rule_negative_sample(
            rows,
            sample_size=2,
            seed="test",
            excluded_example_ids={"d:0"},
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row["glamour_id"] for row in selected}), 2)
        self.assertNotIn("b:0", {row["example_id"] for row in selected})
        self.assertNotIn("d:0", {row["example_id"] for row in selected})

    def test_selection_is_order_independent(self) -> None:
        rows = [
            {"example_id": f"{index}:0", "glamour_id": str(index), "triage_for_review": False}
            for index in range(5)
        ]

        first = select_rule_negative_sample(rows, sample_size=3, seed="test")
        second = select_rule_negative_sample(
            list(reversed(rows)), sample_size=3, seed="test"
        )

        self.assertEqual(
            [row["example_id"] for row in first],
            [row["example_id"] for row in second],
        )

    def test_stable_key_changes_with_seed(self) -> None:
        row = {"example_id": "1:0"}
        self.assertNotEqual(stable_sample_key(row, "a"), stable_sample_key(row, "b"))

    def test_rejects_insufficient_glamours(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested 2"):
            select_rule_negative_sample(
                [
                    {
                        "example_id": "a:0",
                        "glamour_id": "a",
                        "triage_for_review": False,
                    }
                ],
                sample_size=2,
                seed="test",
            )


if __name__ == "__main__":
    unittest.main()
