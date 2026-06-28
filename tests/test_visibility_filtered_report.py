from __future__ import annotations

import unittest

from scripts.evaluate_visibility_filtered_report import summarize_cases


class VisibilityFilteredReportTests(unittest.TestCase):
    def test_summarizes_only_supplied_visible_cases(self) -> None:
        result = summarize_cases(
            [
                {
                    "slot": "BODY",
                    "appearance_hit_at_1": True,
                    "appearance_hit_at_k": True,
                    "appearance_rank": 1,
                },
                {
                    "slot": "BODY",
                    "appearance_hit_at_1": False,
                    "appearance_hit_at_k": True,
                    "appearance_rank": 4,
                },
            ],
            10,
        )

        self.assertEqual(result["eligible_case_count"], 2)
        self.assertEqual(result["appearance_recall_at_1"], 0.5)
        self.assertEqual(result["appearance_recall_at_k"], 1.0)
        self.assertEqual(result["by_slot"]["BODY"]["appearance_mrr"], 0.625)


if __name__ == "__main__":
    unittest.main()
