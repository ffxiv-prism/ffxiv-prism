from __future__ import annotations

import unittest

import torch

from scripts.train_glamour_pair_reranker import PairReranker


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


if __name__ == "__main__":
    unittest.main()
