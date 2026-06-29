from __future__ import annotations

import unittest

from ffxiv_prism.vision.rank_fusion import (
    appearance_union_recall,
    fuse_case_results_rrf,
)


def case(results: list[dict], expected: str = "target") -> dict:
    return {
        "status": "eligible",
        "example_id": "example:0",
        "slot": "BODY",
        "eligible_expected_piece_ids": [9],
        "expected_appearance_key": expected,
        "results": results,
    }


class RankFusionTests(unittest.TestCase):
    def test_weighted_rrf_preserves_primary_top_one(self) -> None:
        primary = case(
            [
                {"piece_id": 1, "appearance_key": "other"},
                {"piece_id": 9, "appearance_key": "target"},
            ]
        )
        secondary = case(
            [
                {"piece_id": 2, "appearance_key": "another"},
                {"piece_id": 9, "appearance_key": "target"},
            ]
        )
        fused = fuse_case_results_rrf([primary], [secondary], primary_weight=0.6)[0]
        self.assertEqual(fused["results"][0]["piece_id"], 9)
        self.assertTrue(fused["appearance_hit_at_1"])

    def test_union_recall_counts_secondary_only_hit(self) -> None:
        primary = case([{"piece_id": 1, "appearance_key": "other"}])
        secondary = case([{"piece_id": 9, "appearance_key": "target"}])
        self.assertEqual(appearance_union_recall([primary], [secondary]), 1.0)
