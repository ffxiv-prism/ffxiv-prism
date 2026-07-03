from __future__ import annotations

import unittest

from ffxiv_prism.vision.evaluation_domains import (
    build_domain_balance_weights,
    summarize_domain_evaluation,
)


class EvaluationDomainTests(unittest.TestCase):
    def test_balancing_upweights_rare_race_gender_groups(self) -> None:
        examples = [
            {
                "example_id": "1:0",
                "glamour_id": 1,
                "model_race": "Hyur",
                "model_gender": "female",
            },
            {
                "example_id": "2:0",
                "glamour_id": 2,
                "model_race": "Hyur",
                "model_gender": "female",
            },
            {
                "example_id": "3:0",
                "glamour_id": 3,
                "model_race": "Hrothgar",
                "model_gender": "male",
            },
        ]

        weights, report = build_domain_balance_weights(examples)

        self.assertGreater(weights["3:0"], weights["1:0"])
        self.assertEqual(
            report["domain_group_counts"],
            {"hrothgar|male": 1, "hyur|female": 2},
        )

    def test_reports_macro_and_undercovered_domains(self) -> None:
        examples = [
            {
                "example_id": "1:0",
                "model_race": "Hyur",
                "model_gender": "female",
            },
            {
                "example_id": "2:0",
                "model_race": "Viera",
                "model_gender": "male",
            },
        ]
        cases = [
            {
                "example_id": "1:0",
                "glamour_id": 1,
                "status": "eligible",
                "appearance_hit_at_1": True,
                "appearance_hit_at_k": True,
                "appearance_rank": 1,
            },
            {
                "example_id": "2:0",
                "glamour_id": 2,
                "status": "eligible",
                "appearance_hit_at_1": False,
                "appearance_hit_at_k": True,
                "appearance_rank": 3,
            },
        ]

        summary = summarize_domain_evaluation(
            cases,
            examples,
            min_groups_per_domain=1,
            min_cases_per_domain=2,
        )

        self.assertEqual(summary["overall"]["appearance_recall_at_1"], 0.5)
        self.assertEqual(
            summary["dimensions"]["model_race"][
                "macro_appearance_recall_at_1"
            ],
            0.5,
        )
        self.assertEqual(
            summary["dimensions"]["model_race"]["undercovered_domains"],
            ["hyur", "viera"],
        )
        self.assertFalse(summary["ready"])


if __name__ == "__main__":
    unittest.main()
