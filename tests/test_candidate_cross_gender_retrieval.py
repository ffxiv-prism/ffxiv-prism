from __future__ import annotations

import unittest

from scripts.diagnose_candidate_cross_gender_retrieval import (
    rank_cross_view_result,
    summarize_cases,
)


class CandidateCrossGenderRetrievalTests(unittest.TestCase):
    def test_rank_cross_view_result_deduplicates_piece_rows(self) -> None:
        rows = [
            {"pieces": [{"piece_id": 1, "name_en": "Wrong"}]},
            {"pieces": [{"piece_id": 1, "name_en": "Wrong"}]},
            {"pieces": [{"piece_id": 2, "name_en": "Expected"}]},
        ]
        rank = rank_cross_view_result(rows, {2}, {"expected"})
        self.assertEqual(rank, 2)

    def test_rank_cross_view_result_accepts_shared_appearance_name(self) -> None:
        rows = [
            {"pieces": [{"piece_id": 7, "name_en": "Shared Gloves"}]},
        ]
        rank = rank_cross_view_result(rows, {99}, {"shared gloves"})
        self.assertEqual(rank, 1)

    def test_summary_excludes_missing_opposite_gender_truth(self) -> None:
        cases = [
            {"status": "eligible", "rank": 2},
            {"status": "eligible", "rank": None},
            {"status": "truth_not_in_opposite_gender_index", "rank": None},
        ]
        summary = summarize_cases(cases, (1, 10))
        self.assertEqual(summary["eligible_case_count"], 2)
        self.assertEqual(summary["hits"], {"1": 0, "10": 1})


if __name__ == "__main__":
    unittest.main()
