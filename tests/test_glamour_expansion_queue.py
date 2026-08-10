from __future__ import annotations

import unittest

from ffxiv_prism.vision.evaluation_split import stable_new_group_role
from scripts.build_glamour_expansion_queue import filter_candidate_roles


class GlamourExpansionQueueTests(unittest.TestCase):
    def test_filters_candidates_to_requested_stable_role(self) -> None:
        glamour_ids = range(1000, 1100)
        expected = [
            value
            for value in glamour_ids
            if stable_new_group_role(value) == "train"
        ]

        self.assertEqual(
            filter_candidate_roles(glamour_ids, {"train"}),
            expected,
        )
        self.assertGreater(len(expected), 0)
        self.assertLess(len(expected), len(glamour_ids))


if __name__ == "__main__":
    unittest.main()
