from __future__ import annotations

import unittest

import numpy as np
import torch

from ffxiv_prism.matching.patch_profiles import (
    dino_patch_grid,
    pool_patch_grid,
    symmetric_chamfer_score_matrix,
)


class PatchProfileTests(unittest.TestCase):
    def test_dinov3_register_tokens_are_excluded(self) -> None:
        tokens = torch.arange(2 * 14 * 3, dtype=torch.float32).reshape(1, 28, 3)

        patches = dino_patch_grid(tokens, num_register_tokens=2)

        self.assertEqual(tuple(patches.shape), (1, 5, 5, 3))
        torch.testing.assert_close(patches[0, 0, 0], tokens[0, 3])

    def test_patch_pooling_returns_normalized_compact_grid(self) -> None:
        patches = torch.randn(2, 4, 4, 8)
        projection = torch.randn(8, 3)

        profiles = pool_patch_grid(
            patches,
            output_grid_size=2,
            projection=projection,
        )

        self.assertEqual(tuple(profiles.shape), (2, 4, 3))
        torch.testing.assert_close(
            torch.linalg.vector_norm(profiles, dim=-1),
            torch.ones(2, 4),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_symmetric_chamfer_is_patch_order_invariant(self) -> None:
        query = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        candidates = np.asarray(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[-1.0, 0.0], [0.0, -1.0]],
            ],
            dtype=np.float32,
        )

        scores = symmetric_chamfer_score_matrix(query, candidates)

        self.assertAlmostEqual(float(scores[0, 0]), 1.0, places=6)
        self.assertLess(float(scores[0, 1]), float(scores[0, 0]))


if __name__ == "__main__":
    unittest.main()
