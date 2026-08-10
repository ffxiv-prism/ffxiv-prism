from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_compact_rank_gate import (
    apply_top1_promotion,
    apply_top1_promotion_to_cases,
    find_domain_regressions,
    promotion_gate_decision,
)


class CompactRankGateTests(unittest.TestCase):
    def test_gate_accepts_supported_promotion(self) -> None:
        accepted, diagnostic = promotion_gate_decision(
            np.asarray([10, 20, 30]),
            np.asarray([20, 10, 30]),
            np.asarray([20, 30, 10]),
            max_base_rank=2,
            max_last_branch_rank=1,
            min_last_branch_rank_ratio=2.0,
        )

        self.assertTrue(accepted)
        self.assertEqual(diagnostic["candidate_base_rank"], 2)

    def test_gate_rejects_deep_baseline_candidate(self) -> None:
        accepted, diagnostic = promotion_gate_decision(
            np.asarray([10, 30, 40, 20]),
            np.asarray([20, 10, 30, 40]),
            np.asarray([20, 10, 30, 40]),
            max_base_rank=3,
            max_last_branch_rank=1,
            min_last_branch_rank_ratio=1.0,
        )

        self.assertFalse(accepted)
        self.assertEqual(diagnostic["reason"], "insufficient_agreement")

    def test_gate_can_require_old_winner_to_remain_in_last_branch_shortlist(self) -> None:
        accepted, diagnostic = promotion_gate_decision(
            np.asarray([10, 20, 30]),
            np.asarray([20, 10, 30]),
            np.asarray([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 10]),
            max_base_rank=2,
            max_last_branch_rank=1,
            min_last_branch_rank_ratio=2.0,
            max_base_last_branch_rank=10,
        )

        self.assertFalse(accepted)
        self.assertEqual(diagnostic["base_last_branch_rank"], 11)

    def test_domain_regression_reports_raw_hit_loss(self) -> None:
        baseline = [
            {
                "example_id": "1",
                "status": "eligible",
                "model_gender": "female",
                "model_race": "Elezen",
                "rank": 1,
            }
        ]
        candidate = [{**baseline[0], "rank": 2}]

        regressions = find_domain_regressions(baseline, candidate, [1, 10])

        self.assertEqual(len(regressions), 2)
        self.assertTrue(all(row["cutoff"] == 1 for row in regressions))

    def test_top1_promotion_preserves_the_baseline_tail(self) -> None:
        promoted = apply_top1_promotion(
            np.asarray([10, 20, 30, 40]),
            np.asarray([30, 50, 10, 20]),
            True,
        )

        self.assertEqual(promoted.tolist(), [30, 10, 20, 40])

    def test_top1_case_update_matches_move_to_front_rank_changes(self) -> None:
        baseline = [
            {"example_id": "correct-new", "rank": 4},
            {"example_id": "old-winner", "rank": 1},
            {"example_id": "before-truth", "rank": 5},
        ]
        candidate = [
            {"example_id": "correct-new", "rank": 1},
            {"example_id": "old-winner", "rank": 2},
            {"example_id": "before-truth", "rank": 6},
        ]
        updated = apply_top1_promotion_to_cases(
            baseline,
            candidate,
            {case["example_id"]: True for case in baseline},
            {
                "correct-new": {"candidate_base_rank": 4},
                "old-winner": {"candidate_base_rank": 2},
                "before-truth": {"candidate_base_rank": 2},
            },
        )

        self.assertEqual([case["rank"] for case in updated], [1, 2, 5])


if __name__ == "__main__":
    unittest.main()
