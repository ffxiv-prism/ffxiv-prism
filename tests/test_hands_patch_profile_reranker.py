from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_hands_patch_profile_reranker import rerank_piece_ids


class HandsPatchProfileRerankerTests(unittest.TestCase):
    def test_patch_score_only_reorders_fixed_leading_pool(self) -> None:
        baseline = np.asarray((10, 20, 30, 40), dtype=np.int64)

        reranked = rerank_piece_ids(
            baseline,
            {10: 0.1, 20: 0.9, 30: 0.2},
            rerank_k=3,
            blend=1.0,
        )

        np.testing.assert_array_equal(reranked, np.asarray((20, 30, 10, 40)))

    def test_zero_blend_preserves_baseline(self) -> None:
        baseline = np.asarray((10, 20, 30), dtype=np.int64)

        reranked = rerank_piece_ids(
            baseline,
            {10: 0.1, 20: 0.9, 30: 0.2},
            rerank_k=3,
            blend=0.0,
        )

        np.testing.assert_array_equal(reranked, baseline)


if __name__ == "__main__":
    unittest.main()
