from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_hands_occupied_slot_penalty import (
    gate_occupied_top1_promotion,
    rerank_occupied_slot_pieces,
)


class HandsOccupiedSlotPenaltyTests(unittest.TestCase):
    def test_zero_multiplier_stably_moves_occupied_pieces_to_tail(self) -> None:
        result = rerank_occupied_slot_pieces(
            np.asarray([20, 10, 30, 40]),
            {10, 40},
            occupied_multiplier=0.0,
            rrf_k=1,
        )
        self.assertEqual(result.tolist(), [10, 40, 20, 30])

    def test_gate_promotes_nearby_exact_piece_over_occupied_top1(self) -> None:
        result, accepted = gate_occupied_top1_promotion(
            np.asarray([20, 10, 30]),
            np.asarray([10, 30, 20]),
            {10, 30},
            max_base_rank=2,
        )
        self.assertTrue(accepted)
        self.assertEqual(result.tolist(), [10, 20, 30])


if __name__ == "__main__":
    unittest.main()
