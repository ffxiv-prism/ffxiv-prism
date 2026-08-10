from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_hands_crop_role_weights import (
    fuse_examples_with_role_multipliers,
)


class HandsCropRoleWeightsTests(unittest.TestCase):
    def test_zero_context_multiplier_removes_context_vote(self) -> None:
        records = {
            "example": [
                {
                    "path": "primary.jpg",
                    "role": "primary",
                    "visibility": "visible",
                    "confidence": 1.0,
                },
                {
                    "path": "context.jpg",
                    "role": "context",
                    "visibility": "visible",
                    "confidence": 1.0,
                },
            ]
        }
        rankings = [np.asarray([10, 20]), np.asarray([20, 10])]
        result = fuse_examples_with_role_multipliers(
            records,
            {"primary.jpg": 0, "context.jpg": 1},
            rankings,
            role_multipliers={"primary": 1.0, "context": 0.0},
            crop_rrf_k=1,
            candidate_k=2,
        )
        self.assertEqual(result["example"].tolist(), [10, 20])


if __name__ == "__main__":
    unittest.main()
