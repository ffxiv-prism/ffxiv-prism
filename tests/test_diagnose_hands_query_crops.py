from __future__ import annotations

import unittest

from scripts.diagnose_hands_query_crops import (
    oracle_summary,
    ranks_by_id,
    select_policy_records,
    summarize_fixed_population,
)


class DiagnoseHandsQueryCropsTests(unittest.TestCase):
    def test_select_policy_records_keeps_all_example_keys(self) -> None:
        records = {
            "a": [
                {"name": "hands_left", "path": "left.jpg"},
                {"name": "hands_right", "path": "right.jpg"},
            ],
            "b": [{"name": "hands_right", "path": "b-right.jpg"}],
        }
        selected = select_policy_records(records, frozenset({"hands_left"}))
        self.assertEqual(selected["a"], [{"name": "hands_left", "path": "left.jpg"}])
        self.assertEqual(selected["b"], [])

    def test_fixed_population_counts_missing_crop_as_miss(self) -> None:
        cases = [
            {"example_id": "a", "status": "eligible", "rank": 3},
            {"example_id": "b", "status": "no_usable_crop", "rank": None},
        ]
        ranks = ranks_by_id(cases, frozenset({"a", "b"}))
        summary = summarize_fixed_population(ranks, (1, 10))
        self.assertEqual(summary["eligible_case_count"], 2)
        self.assertEqual(summary["hits"], {"1": 0, "10": 1})
        self.assertEqual(summary["missing_truth_rank_count"], 1)

    def test_oracle_uses_best_rank_per_example_without_changing_denominator(self) -> None:
        trials = [
            {"trial_id": "left", "ranks": {"a": 2, "b": None, "c": 20}},
            {"trial_id": "right", "ranks": {"a": 5, "b": 7, "c": 4}},
        ]
        oracle = oracle_summary(trials, frozenset({"a", "b", "c"}), (1, 10))
        self.assertEqual(oracle["best_rank_by_example"], {"a": 2, "b": 7, "c": 4})
        self.assertEqual(oracle["summary"]["hits"]["10"], 3)
        self.assertEqual(oracle["winning_trials_by_example"]["a"], ["left"])


if __name__ == "__main__":
    unittest.main()
