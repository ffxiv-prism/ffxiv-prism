from __future__ import annotations

import unittest

from scripts.evaluate_multiview_group_fusion import (
    fuse_group_cases,
    summarize_group_cases,
)


class MultiviewGroupFusionTests(unittest.TestCase):
    def test_reciprocal_rank_votes_promote_consistent_candidate(self) -> None:
        cases = [
            {
                "status": "eligible",
                "glamour_id": 10,
                "example_id": "10:0",
                "slot": "BODY",
                "expected_name": "Correct Coat",
                "expected_appearance_key": "correct coat",
                "eligible_expected_piece_ids": [7],
                "results": [
                    {"piece_id": 11, "rank": 1, "appearance_key": "first distractor"},
                    {"piece_id": 7, "rank": 2, "appearance_key": "correct coat"},
                ],
            },
            {
                "status": "eligible",
                "glamour_id": 10,
                "example_id": "10:1",
                "slot": "BODY",
                "expected_name": "Correct Coat",
                "expected_appearance_key": "correct coat",
                "eligible_expected_piece_ids": [7],
                "results": [
                    {"piece_id": 12, "rank": 1, "appearance_key": "second distractor"},
                    {"piece_id": 7, "rank": 2, "appearance_key": "correct coat"},
                ],
            },
        ]

        fused = fuse_group_cases(cases, slots={"BODY"}, top_k=10)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["view_count"], 2)
        self.assertEqual(fused[0]["results"][0]["piece_id"], 7)
        self.assertTrue(fused[0]["appearance_hit_at_1"])
        self.assertEqual(
            summarize_group_cases(fused, 10)["appearance_recall_at_1"],
            1.0,
        )

    def test_ignores_ineligible_and_unselected_slots(self) -> None:
        cases = [
            {
                "status": "no_usable_crop",
                "glamour_id": 10,
                "example_id": "10:0",
                "slot": "BODY",
            },
            {
                "status": "eligible",
                "glamour_id": 10,
                "example_id": "10:0",
                "slot": "HEAD",
                "eligible_expected_piece_ids": [1],
                "results": [{"piece_id": 1, "rank": 1}],
            },
        ]

        self.assertEqual(fuse_group_cases(cases, slots={"BODY"}), [])


if __name__ == "__main__":
    unittest.main()
