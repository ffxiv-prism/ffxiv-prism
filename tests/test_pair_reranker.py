from __future__ import annotations

import unittest

import torch

from scripts.train_glamour_pair_reranker import (
    PairReranker,
    aggregate_candidate_views,
    aggregate_crop_scores,
    combine_group_weights,
)


class PairRerankerTests(unittest.TestCase):
    def test_reranker_starts_as_scaled_cosine(self) -> None:
        model = PairReranker(8, hidden_dimension=16, slot_dimension=4)
        queries = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        candidates = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)

        scores = model(queries, candidates, "BODY")
        expected = 10.0 * (queries * candidates).sum(dim=-1)

        self.assertTrue(torch.allclose(scores, expected, atol=1e-6))

    def test_per_slot_heads_also_start_as_scaled_cosine(self) -> None:
        model = PairReranker(
            8,
            hidden_dimension=16,
            slot_dimension=4,
            per_slot_heads=True,
        )
        queries = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        candidates = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        expected = 10.0 * (queries * candidates).sum(dim=-1)

        for slot in ("HEAD", "BODY", "HANDS", "LEGS", "FEET"):
            self.assertTrue(
                torch.allclose(
                    model(queries, candidates, slot),
                    expected,
                    atol=1e-6,
                )
            )

    def test_max_aggregation_keeps_strongest_query_evidence(self) -> None:
        scores = torch.tensor([[1.0, 4.0], [3.0, 2.0]])

        aggregated = aggregate_crop_scores(scores, "max")

        self.assertTrue(torch.equal(aggregated, torch.tensor([3.0, 4.0])))

    def test_logmeanexp_matches_single_query(self) -> None:
        scores = torch.tensor([[1.5, -2.0]])

        aggregated = aggregate_crop_scores(scores, "logmeanexp")

        self.assertTrue(torch.allclose(aggregated, scores[0]))

    def test_rejects_unknown_aggregation(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_crop_scores(torch.ones(1, 2), "median")

    def test_top2_mean_uses_distinct_candidate_views(self) -> None:
        views = {
            "upper_body": {"score": 3.0, "piece_id": 1},
            "torso_core": {"score": 1.0, "piece_id": 1},
        }

        result = aggregate_candidate_views(views, "top2_mean")

        self.assertEqual(result["score"], 2.0)
        self.assertEqual(result["best_view_score"], 3.0)
        self.assertEqual(result["candidate_view_count"], 2)

    def test_group_weights_multiply_replay_and_domain_factors(self) -> None:
        combined = combine_group_weights(
            {"old": 1.2, "rare-old": 1.5},
            {"old": 1.5, "new": 1.0, "rare-old": 1.5},
        )

        self.assertAlmostEqual(combined["old"], 1.8)
        self.assertAlmostEqual(combined["rare-old"], 2.25)
        self.assertAlmostEqual(combined["new"], 1.0)


if __name__ == "__main__":
    unittest.main()
