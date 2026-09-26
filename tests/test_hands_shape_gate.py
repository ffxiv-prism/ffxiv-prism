from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ffxiv_prism.matching.hands_morphology import (
    hands_shape_decision, rerank_candidates_with_hand_shape,
)


def profile(mitten: float) -> tuple[float, ...]:
    return (mitten, 1 - mitten, 0, 0, 0)


class HandsShapeGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {"rank": i, "piece_id": i, "score": 1 / i, "link_type": "exact_slot"}
            for i in range(1, 12)
        ]
        self.query = [profile(.98), profile(.84)]
        self.profiles = {i: profile(.05) for i in range(1, 12)}
        self.profiles[10] = profile(.99)

    def test_recovers_deep_candidate_without_names_or_identity_prior(self) -> None:
        result, diagnostic = rerank_candidates_with_hand_shape(
            self.candidates, self.query, self.profiles,
        )
        self.assertEqual(result[0]["piece_id"], 10)
        self.assertEqual(result[0]["hand_shape_gate"]["previous_rank"], 10)
        self.assertEqual(result[0]["score"], self.candidates[9]["score"])
        self.assertEqual([r["piece_id"] for r in result[1:]], list(range(1, 10)) + [11])
        self.assertTrue(diagnostic["ranking_changed"])
        self.assertNotIn("hand_shape_gate", self.candidates[9])

    def test_uncertain_disagreeing_or_single_hand_leaves_ranking_unchanged(self) -> None:
        for query in ([profile(.98)], [profile(.98), profile(.69)],
                      [profile(.05), profile(.10)], []):
            with self.subTest(query=query):
                result, diagnostic = rerank_candidates_with_hand_shape(
                    self.candidates, query, self.profiles,
                )
                self.assertEqual(result, self.candidates)
                self.assertFalse(diagnostic["active"])

    def test_incomplete_previews_are_not_silently_penalized(self) -> None:
        del self.profiles[5]
        result, diagnostic = rerank_candidates_with_hand_shape(
            self.candidates, self.query, self.profiles,
        )
        self.assertEqual(result, self.candidates)
        self.assertEqual(diagnostic["reason"], "candidate_preview_unavailable")

    def test_plausible_baseline_is_preserved(self) -> None:
        self.profiles[1] = profile(.3)
        result, diagnostic = rerank_candidates_with_hand_shape(
            self.candidates, self.query, self.profiles,
        )
        self.assertEqual(result, self.candidates)
        self.assertEqual(diagnostic["reason"], "baseline_shape_not_contradicted")

    def test_occupied_outfits_are_not_promoted_and_promotions_are_bounded(self) -> None:
        self.profiles = {i: profile(.9 + i / 1000) for i in range(1, 12)}
        self.profiles[1] = profile(.05)
        self.candidates[10]["link_type"] = "occupied_slot"
        result, _ = rerank_candidates_with_hand_shape(self.candidates, self.query, self.profiles)
        self.assertEqual(sum("hand_shape_gate" in row for row in result), 5)
        self.assertEqual(result[5]["piece_id"], 1)
        self.assertEqual(result[-1]["piece_id"], 11)
        self.assertEqual([row["rank"] for row in result], list(range(1, 12)))

    def test_invalid_probabilities_rejected(self) -> None:
        for values in ((float("nan"), 0, 0, 0, 0), (0, 0, 0, 0, 0), (1, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                hands_shape_decision([values, values])

    def test_runtime_skips_classifier_when_complete_hands_unavailable(self) -> None:
        from ffxiv_prism.cli.match import _rerank_hands_shape
        with patch("ffxiv_prism.cli.match._classify_style_paths") as classifier:
            result, diagnostic = _rerank_hands_shape(
                self.candidates, [], "model", device=None, batch_size=1,
            )
        classifier.assert_not_called()
        self.assertEqual(result, self.candidates)
        self.assertFalse(diagnostic["active"])

    def test_runtime_only_classifies_candidates_after_bilateral_confirmation(self) -> None:
        from ffxiv_prism.cli.match import _rerank_hands_shape
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.jpg"
            path.touch()
            records = [{"name": "hands_palm_" + side, "visibility": "visible", "path": path}
                       for side in ("left", "right")]
            with patch("ffxiv_prism.cli.match._classify_style_paths",
                       return_value=[profile(.1), profile(.1)]) as classifier:
                result, diagnostic = _rerank_hands_shape(
                    self.candidates, records, "model", device=None, batch_size=1,
                )
            self.assertEqual(classifier.call_count, 1)
            self.assertEqual(result, self.candidates)
            self.assertFalse(diagnostic["active"])

    def test_runtime_classifies_unique_previews_and_attaches_gate_evidence(self) -> None:
        from ffxiv_prism.cli.match import _rerank_hands_shape
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.jpg", Path(directory) / "second.jpg"
            first.touch()
            second.touch()
            records = [{"name": "hands_palm_" + side, "visibility": "visible", "path": first}
                       for side in ("left", "right")]
            candidates = [dict(row, local_preview=second if row["piece_id"] == 10 else first)
                          for row in self.candidates]
            with patch("ffxiv_prism.cli.match._classify_style_paths", side_effect=[
                self.query, [profile(.05), profile(.99)],
            ]) as classifier:
                result, diagnostic = _rerank_hands_shape(
                    candidates, records, "model", device=None, batch_size=1,
                )
            self.assertEqual(classifier.call_count, 2)
            self.assertEqual(classifier.call_args_list[1].args[2], [first, second])
            self.assertEqual(result[0]["piece_id"], 10)
            self.assertTrue(diagnostic["ranking_changed"])

    def test_runtime_missing_preview_preserves_entire_pool(self) -> None:
        from ffxiv_prism.cli.match import _rerank_hands_shape
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.jpg"
            path.touch()
            records = [{"name": "hands_palm_" + side, "visibility": "visible", "path": path}
                       for side in ("left", "right")]
            with patch("ffxiv_prism.cli.match._classify_style_paths",
                       return_value=self.query) as classifier:
                result, diagnostic = _rerank_hands_shape(
                    self.candidates, records, "model", device=None, batch_size=1,
                )
            self.assertEqual(classifier.call_count, 1)
            self.assertEqual(result, self.candidates)
            self.assertEqual(diagnostic["reason"], "candidate_preview_unavailable")


if __name__ == "__main__":
    unittest.main()
