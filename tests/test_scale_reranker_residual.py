from __future__ import annotations

import unittest

import torch

from scripts.scale_reranker_residual import scale_checkpoint_payload


class ScaleRerankerResidualTest(unittest.TestCase):
    def test_scales_shared_residual_output_only(self) -> None:
        payload = {
            "per_slot_heads": False,
            "state_dict": {
                "base_scale": torch.tensor(10.0),
                "residual.0.weight": torch.tensor([3.0]),
                "residual.5.weight": torch.tensor([2.0]),
                "residual.5.bias": torch.tensor([4.0]),
            },
        }

        scaled = scale_checkpoint_payload(payload, 0.5)

        self.assertEqual(scaled["state_dict"]["base_scale"].item(), 10.0)
        self.assertEqual(
            scaled["state_dict"]["residual.0.weight"].item(),
            3.0,
        )
        self.assertEqual(
            scaled["state_dict"]["residual.5.weight"].item(),
            1.0,
        )
        self.assertEqual(
            scaled["state_dict"]["residual.5.bias"].item(),
            2.0,
        )
        self.assertEqual(
            payload["state_dict"]["residual.5.weight"].item(),
            2.0,
        )

    def test_rejects_negative_scale(self) -> None:
        with self.assertRaises(ValueError):
            scale_checkpoint_payload({"state_dict": {}}, -0.1)


if __name__ == "__main__":
    unittest.main()
