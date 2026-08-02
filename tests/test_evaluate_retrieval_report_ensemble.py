from __future__ import annotations

import unittest

from scripts.evaluate_retrieval_report_ensemble import (
    fuse_primary_pool,
    fuse_union_pool,
)


class EvaluateRetrievalReportEnsembleTests(unittest.TestCase):
    def test_zero_secondary_weight_preserves_primary_order(self) -> None:
        primary = [{"piece_id": 1}, {"piece_id": 2}, {"piece_id": 3}]
        secondary = [{"piece_id": 3}, {"piece_id": 2}, {"piece_id": 1}]
        fused = fuse_primary_pool(
            primary,
            secondary,
            secondary_weight=0.0,
            rrf_k=1,
            pool_depth=3,
        )
        self.assertEqual([row["piece_id"] for row in fused], [1, 2, 3])

    def test_secondary_can_promote_within_primary_pool(self) -> None:
        primary = [{"piece_id": 1}, {"piece_id": 2}, {"piece_id": 3}]
        secondary = [{"piece_id": 3}, {"piece_id": 2}, {"piece_id": 1}]
        fused = fuse_primary_pool(
            primary,
            secondary,
            secondary_weight=2.0,
            rrf_k=1,
            pool_depth=3,
        )
        self.assertEqual(fused[0]["piece_id"], 3)

    def test_candidates_outside_pool_are_not_inserted(self) -> None:
        primary = [{"piece_id": 1}, {"piece_id": 2}, {"piece_id": 3}]
        secondary = [{"piece_id": 9}, {"piece_id": 3}]
        fused = fuse_primary_pool(
            primary,
            secondary,
            secondary_weight=3.0,
            rrf_k=1,
            pool_depth=2,
        )
        self.assertEqual({row["piece_id"] for row in fused}, {1, 2, 3})

    def test_union_pool_retains_new_secondary_candidate(self) -> None:
        primary = [{"piece_id": 1}, {"piece_id": 2}]
        secondary = [{"piece_id": 3}, {"piece_id": 2}]
        fused = fuse_union_pool(
            primary,
            secondary,
            secondary_weight=1.0,
            rrf_k=1,
            pool_depth=2,
        )
        self.assertEqual({row["piece_id"] for row in fused}, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
