from __future__ import annotations

import unittest

from scripts.summarize_frozen_rank_gate import domain_hit_table


class SummarizeFrozenRankGateTests(unittest.TestCase):
    def test_domain_hit_table_keeps_raw_counts(self) -> None:
        baseline = [
            {
                "example_id": "1",
                "status": "eligible",
                "model_gender": "female",
                "model_race": "Hyur",
                "rank": 2,
            }
        ]
        candidate = [{**baseline[0], "rank": 1}]

        table = domain_hit_table(baseline, candidate, [1, 10])

        self.assertEqual(
            table["model_gender"]["female"]["baseline_hits"]["1"], 0
        )
        self.assertEqual(
            table["model_race"]["Hyur"]["candidate_hits"]["1"], 1
        )


if __name__ == "__main__":
    unittest.main()
