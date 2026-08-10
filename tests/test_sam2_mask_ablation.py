from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_sam2_mask_ablation import (
    inset_prompt_box,
    select_sam_mask,
)


class Sam2MaskAblationTests(unittest.TestCase):
    def test_inset_prompt_box_stays_inside_image(self) -> None:
        self.assertEqual(inset_prompt_box((100, 200)), [4.0, 8.0, 96.0, 192.0])

    def test_empty_high_score_mask_is_rejected(self) -> None:
        masks = np.zeros((3, 10, 10), dtype=bool)
        masks[1, 2:8, 2:8] = True
        masks[2, :, :] = True
        scores = np.asarray([0.99, 0.80, 0.70], dtype=np.float32)
        self.assertEqual(select_sam_mask(masks, scores, "BODY"), 1)

    def test_small_hand_mask_can_win(self) -> None:
        masks = np.zeros((2, 20, 20), dtype=bool)
        masks[0, 8:12, 8:12] = True
        masks[1, 2:18, 2:18] = True
        scores = np.asarray([0.85, 0.80], dtype=np.float32)
        self.assertEqual(select_sam_mask(masks, scores, "HANDS"), 0)
