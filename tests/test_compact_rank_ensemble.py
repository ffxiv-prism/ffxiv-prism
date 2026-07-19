from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_compact_rank_ensemble import (
    _fuse_all_examples_max,
    _fuse_crops_then_branches,
    constrain_piece_ids_to_pool,
    stable_weighted_rrf,
    stable_weighted_rrf_with_scores,
)


class CompactRankEnsembleTests(unittest.TestCase):
    def test_stable_rrf_preserves_first_branch_order_when_scores_tie(self) -> None:
        result = stable_weighted_rrf(
            [np.asarray([1, 2]), np.asarray([2, 1])],
            [1.0, 1.0],
            rrf_k=1,
            top_k=2,
        )
        self.assertEqual(result.tolist(), [1, 2])

    def test_stable_rrf_applies_branch_weights(self) -> None:
        result = stable_weighted_rrf(
            [np.asarray([1, 2]), np.asarray([2, 1])],
            [1.0, 2.0],
            rrf_k=1,
            top_k=2,
        )
        self.assertEqual(result.tolist(), [2, 1])

    def test_final_pool_restores_candidates_missing_from_reranker(self) -> None:
        result = constrain_piece_ids_to_pool(
            np.asarray([1, 2, 3]),
            np.asarray([4, 2, 1]),
        )
        self.assertEqual(result.tolist(), [2, 1, 3])

    def test_stable_rrf_returns_scores_aligned_with_ranked_ids(self) -> None:
        piece_ids, scores = stable_weighted_rrf_with_scores(
            [np.asarray([1, 2]), np.asarray([2, 1])],
            [1.0, 2.0],
            rrf_k=1,
            top_k=2,
        )
        self.assertEqual(piece_ids.tolist(), [2, 1])
        np.testing.assert_allclose(scores, [4.0 / 3.0, 7.0 / 6.0])

    def test_max_crop_fusion_uses_weighted_scores_and_stable_ties(self) -> None:
        result = _fuse_all_examples_max(
            {
                "sample": [
                    {
                        "path": "crop-a",
                        "role": "primary",
                        "visibility": "visible",
                        "confidence": 1.0,
                    },
                    {
                        "path": "crop-b",
                        "role": "auxiliary",
                        "visibility": "visible",
                        "confidence": 1.0,
                    },
                ]
            },
            {"crop-a": 0, "crop-b": 1},
            [np.asarray([1, 2]), np.asarray([2, 1])],
            [np.asarray([0.5, 0.25]), np.asarray([0.6, 0.1])],
            candidate_k=2,
        )
        self.assertEqual(result["sample"].tolist(), [1, 2])

    def test_crop_then_branch_preserves_single_branch_raw_max_order(self) -> None:
        records = {
            "sample": [
                {
                    "path": "crop-a",
                    "role": "primary",
                    "visibility": "visible",
                    "confidence": 1.0,
                },
                {
                    "path": "crop-b",
                    "role": "primary",
                    "visibility": "visible",
                    "confidence": 1.0,
                },
            ]
        }
        result = _fuse_crops_then_branches(
            [np.asarray([[1, 2], [2, 1]])],
            [np.asarray([[0.6, 0.5], [0.9, 0.2]])],
            [1.0],
            records_by_example=records,
            path_index={"crop-a": 0, "crop-b": 1},
            branch_rrf_k=1,
            candidate_k=2,
        )
        self.assertEqual(result["sample"].tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
