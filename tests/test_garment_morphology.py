from __future__ import annotations

import unittest

from ffxiv_prism.matching.garment_morphology import (
    BODY_STYLE_CLASSES,
    LEGS_STYLE_CLASSES,
    StylePrediction,
    apply_style_crop_multipliers,
    rerank_candidates_with_style,
    style_policy,
)


class GarmentMorphologyTests(unittest.TestCase):
    def test_body_policy_prioritizes_waist_and_shoulder_style_crops(self) -> None:
        prediction = StylePrediction.from_probabilities(
            "BODY",
            (0.80, 0.10, 0.06, 0.04),
        )
        policy = style_policy(prediction, minimum_confidence=0.75)
        records = [
            {"name": "body_torso", "role": "primary"},
            {"name": "body_upper_arm_overlap", "role": "auxiliary"},
            {"name": "body_context", "role": "context"},
        ]

        adjusted = apply_style_crop_multipliers(records, policy)

        self.assertTrue(policy.active)
        self.assertEqual(
            [row["style_weight_multiplier"] for row in adjusted],
            [1.0, 0.55, 0.25],
        )
        self.assertEqual(
            policy.candidate_crop_routes["body_torso"],
            ("torso_core",),
        )
        self.assertNotIn("style_weight_multiplier", records[0])

    def test_body_style_reranker_promotes_crop_top_candidate(self) -> None:
        prediction = StylePrediction.from_probabilities(
            "BODY",
            (0.82, 0.08, 0.06, 0.04),
        )
        candidates = [
            {"rank": 1, "piece_id": 1, "score": 0.95, "name_en": "Long coat"},
            {"rank": 2, "piece_id": 2, "score": 0.90, "name_en": "Crop top"},
            {"rank": 3, "piece_id": 3, "score": 0.85, "name_en": "Regular top"},
        ]
        profiles = {
            1: (0.04, 0.04, 0.12, 0.80),
            2: (0.78, 0.10, 0.08, 0.04),
            3: (0.08, 0.08, 0.76, 0.08),
        }

        reranked = rerank_candidates_with_style(
            candidates,
            prediction,
            profiles,
            rerank_k=3,
            blend=0.60,
            minimum_confidence=0.75,
        )

        self.assertEqual([row["piece_id"] for row in reranked], [2, 1, 3])
        self.assertGreater(reranked[0]["style_score"], reranked[1]["style_score"])
        self.assertEqual(reranked[0]["retrieval_score"], 0.90)
        self.assertEqual([row["rank"] for row in reranked], [1, 2, 3])

    def test_legs_style_can_outweigh_unmatched_lower_leg_coverage(self) -> None:
        prediction = StylePrediction.from_probabilities(
            "LEGS",
            (0.72, 0.12, 0.06, 0.06, 0.04),
        )
        policy = style_policy(prediction, minimum_confidence=0.70)
        records = [
            {"name": "legs_waist_thighs"},
            {"name": "legs_left"},
            {"name": "legs_right"},
            {"name": "legs_context"},
            {"name": "legs_boot_overlap"},
        ]
        adjusted = apply_style_crop_multipliers(records, policy)

        self.assertEqual(
            [row["style_weight_multiplier"] for row in adjusted],
            [1.25, 0.60, 0.60, 0.40, 0.20],
        )
        self.assertEqual(
            policy.candidate_crop_routes["legs_context"],
            ("full",),
        )

        candidates = [
            {
                "rank": 1,
                "piece_id": 10,
                "score": 0.94,
                "name_en": "Long bottoms with calf coverage",
            },
            {
                "rank": 2,
                "piece_id": 11,
                "score": 0.88,
                "name_en": "Hot shorts",
            },
        ]
        reranked = rerank_candidates_with_style(
            candidates,
            prediction,
            {
                10: (0.02, 0.03, 0.02, 0.03, 0.90),
                11: (0.80, 0.08, 0.04, 0.04, 0.04),
            },
            rerank_k=2,
            blend=0.60,
            minimum_confidence=0.70,
        )

        self.assertEqual(reranked[0]["piece_id"], 11)

    def test_composite_asymmetric_leg_style_uses_target_mass_and_soft_features(
        self,
    ) -> None:
        prediction = StylePrediction.from_probabilities(
            "LEGS",
            (0.34, 0.01, 0.20, 0.33, 0.12),
        )
        policy = style_policy(prediction, minimum_confidence=0.50)
        candidates = [
            {"rank": 1, "piece_id": 1, "score": 0.95},
            {"rank": 2, "piece_id": 2, "score": 0.90},
        ]

        reranked = rerank_candidates_with_style(
            candidates,
            prediction,
            {
                1: (0.55, 0.05, 0.05, 0.05, 0.30),
                2: (0.01, 0.01, 0.02, 0.94, 0.02),
            },
            rerank_k=2,
            blend=0.80,
            minimum_confidence=0.50,
        )

        self.assertTrue(policy.active)
        self.assertLess(prediction.confidence, 0.50)
        self.assertEqual(reranked[0]["piece_id"], 2)
        self.assertGreater(reranked[0]["style_score"], reranked[1]["style_score"])

    def test_low_confidence_and_missing_profiles_are_true_noops(self) -> None:
        candidates = [
            {"rank": 7, "piece_id": 1, "score": 0.90},
            {"rank": 8, "piece_id": 2, "score": 0.80},
        ]
        low_confidence = StylePrediction.from_probabilities(
            "BODY",
            (0.40, 0.30, 0.20, 0.10),
        )
        low_result = rerank_candidates_with_style(
            candidates,
            low_confidence,
            {},
            rerank_k=2,
            blend=0.50,
            minimum_confidence=0.75,
        )
        self.assertEqual(low_result, candidates)
        self.assertIsNot(low_result[0], candidates[0])

        confident = StylePrediction.from_probabilities(
            "BODY",
            (0.80, 0.10, 0.06, 0.04),
        )
        incomplete_result = rerank_candidates_with_style(
            candidates,
            confident,
            {1: (0.70, 0.10, 0.10, 0.10)},
            rerank_k=2,
            blend=0.50,
            minimum_confidence=0.75,
        )
        self.assertEqual(incomplete_result, candidates)

        non_target = StylePrediction.from_probabilities(
            "BODY",
            (0.05, 0.05, 0.80, 0.10),
        )
        non_target_result = rerank_candidates_with_style(
            candidates,
            non_target,
            {
                1: (0.70, 0.10, 0.10, 0.10),
                2: (0.10, 0.10, 0.70, 0.10),
            },
            rerank_k=2,
            blend=0.50,
            minimum_confidence=0.75,
        )
        self.assertEqual(non_target_result, candidates)

    def test_rerank_preserves_tail_and_writes_score_diagnostics(self) -> None:
        prediction = StylePrediction.from_probabilities(
            "LEGS",
            dict(
                zip(
                    LEGS_STYLE_CLASSES,
                    (0.65, 0.15, 0.08, 0.10, 0.02),
                    strict=True,
                )
            ),
        )
        candidates = [
            {"rank": 1, "piece_id": 1, "score": 0.99},
            {"rank": 2, "piece_id": 2, "score": 0.95},
            {"rank": 3, "piece_id": 3, "score": 0.90},
            {"rank": 4, "piece_id": 4, "score": 0.85, "tail_marker": "keep"},
        ]
        profiles = {
            1: (0.01, 0.02, 0.03, 0.04, 0.90),
            2: (0.65, 0.15, 0.08, 0.07, 0.05),
            3: (0.08, 0.10, 0.62, 0.10, 0.10),
        }

        reranked = rerank_candidates_with_style(
            candidates,
            prediction,
            profiles,
            rerank_k=3,
            blend=0.60,
            minimum_confidence=0.65,
        )

        self.assertEqual(reranked[-1]["piece_id"], 4)
        self.assertEqual(reranked[-1]["score"], 0.85)
        self.assertEqual(reranked[-1]["tail_marker"], "keep")
        self.assertNotIn("style_score", reranked[-1])
        self.assertEqual([row["rank"] for row in reranked], [1, 2, 3, 4])
        for row in reranked[:3]:
            self.assertIn("retrieval_score", row)
            self.assertIn("retrieval_percentile", row)
            self.assertIn("style_score", row)
            self.assertIn("style_percentile", row)
            self.assertGreaterEqual(row["score"], 0.0)
            self.assertLessEqual(row["score"], 1.0)

    def test_prediction_validates_probability_taxonomy(self) -> None:
        prediction = StylePrediction.from_probabilities(
            "body",
            dict(zip(BODY_STYLE_CLASSES, (0.1, 0.2, 0.3, 0.4), strict=True)),
        )
        self.assertEqual(prediction.slot, "BODY")
        self.assertEqual(prediction.predicted_class, "long_top")
        self.assertEqual(prediction.confidence, 0.4)

        with self.assertRaises(ValueError):
            StylePrediction.from_probabilities("BODY", (0.5, 0.2, 0.1, 0.1))


if __name__ == "__main__":
    unittest.main()
