from __future__ import annotations

import unittest

from ffxiv_prism.matching.footwear_consensus import (
    FOOTWEAR_CONSENSUS_CROPS,
    apply_footwear_consensus,
    footwear_consensus_records,
)


class FootwearConsensusTests(unittest.TestCase):
    def candidates(self):
        return [
            dict(piece_id=i, rank=i, score=1 / i, link_type="exact_slot")
            for i in range(1, 31)
        ]

    def votes(self, *ids):
        return [[dict(piece_id=i, link_type="exact_slot")] for i in ids]

    def test_agreement_moves_existing_appearance_and_preserves_recall_and_scores(self):
        original = self.candidates()
        revised, diagnostic = apply_footwear_consensus(
            original, self.votes(40, 4, 41), {4: (40, 41), 40: (4, 41), 41: (4, 40)}
        )
        self.assertTrue(diagnostic["ranking_changed"])
        self.assertEqual(revised[0]["piece_id"], 4)
        self.assertEqual(revised[0]["score"], 0.25)
        for k in (5, 10, 20, 30):
            self.assertEqual(
                {r["piece_id"] for r in original[:k]},
                {r["piece_id"] for r in revised[:k]},
            )
        self.assertEqual(
            [r["piece_id"] for r in revised[1:]],
            [r["piece_id"] for r in original if r["piece_id"] != 4],
        )
        self.assertEqual([r["rank"] for r in revised], list(range(1, 31)))
        self.assertNotIn("footwear_consensus_gate", original[3])

    def test_disagreement_missing_votes_and_outside_shortlist_are_no_ops(self):
        original = self.candidates()
        for votes in (
            self.votes(4, 4, 3),
            self.votes(4, 4),
            [[], [], []],
            self.votes(6, 6, 6),
            self.votes(1, 1, 1),
        ):
            with self.subTest(votes=votes):
                result, diagnostic = apply_footwear_consensus(original, votes, {})
                self.assertEqual(result, original)
                self.assertFalse(diagnostic["ranking_changed"])

    def test_occupied_slot_support_cannot_promote(self):
        votes = self.votes(4, 4, 4)
        votes[1][0]["link_type"] = "occupied_slot"
        result, diagnostic = apply_footwear_consensus(self.candidates(), votes, {})
        self.assertFalse(diagnostic["ranking_changed"])
        self.assertEqual(result, self.candidates())

    def test_clipped_or_uncertain_whole_feet_disable_consensus(self):
        records = [
            dict(name=name, visibility="visible", confidence=0.9, tags=[])
            for name in FOOTWEAR_CONSENSUS_CROPS
        ]
        self.assertEqual(len(footwear_consensus_records(records)), 3)
        for change in (
            {"visibility": "partially_visible"},
            {"confidence": 0.79},
            {"confidence": float("nan")},
            {"tags": ["source_edge_truncation_risk"]},
        ):
            modified = [dict(r) for r in records]
            modified[0].update(change)
            self.assertEqual(footwear_consensus_records(modified), [])
        self.assertEqual(footwear_consensus_records(records[:-1]), [])


if __name__ == "__main__":
    unittest.main()
