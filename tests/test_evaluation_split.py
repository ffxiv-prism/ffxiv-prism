from __future__ import annotations

import unittest

from ffxiv_prism.vision.evaluation_split import (
    development_split_map,
    stable_new_group_role,
)


class EvaluationSplitTests(unittest.TestCase):
    def test_new_group_assignment_is_stable(self) -> None:
        self.assertEqual(
            stable_new_group_role("12345"),
            stable_new_group_role(12345),
        )

    def test_blind_groups_are_excluded_from_development_map(self) -> None:
        mapped = development_split_map(
            ["1", "2", "3", "4"],
            {
                "1": "train",
                "2": "validation",
                "3": "development_test",
                "4": "blind_test",
            },
        )
        self.assertEqual(
            mapped,
            {"1": "train", "2": "validation", "3": "test"},
        )


if __name__ == "__main__":
    unittest.main()
