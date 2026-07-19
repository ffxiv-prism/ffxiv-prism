from __future__ import annotations

import unittest

import numpy as np

from ffxiv_prism.matching.head_morphology import (
    classification_summary,
    fit_balanced_ridge_classifier,
    head_morphology_class,
    predict_ridge_probabilities,
    rerank_candidates_by_enclosing_probability,
    rerank_candidates_by_covered_probability,
    select_confidence_threshold,
)


class HeadMorphologyTests(unittest.TestCase):
    def test_name_taxonomy_separates_covering_face_and_open_items(self) -> None:
        self.assertEqual(head_morphology_class("Virtu Orison Hood"), "enclosing")
        self.assertEqual(head_morphology_class("Edenmorn Eye Mask"), "face")
        self.assertEqual(head_morphology_class("Thavnairian Headdress"), "open")

    def test_balanced_ridge_classifier_learns_separable_classes(self) -> None:
        features = np.asarray(
            [[-2.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, 1.0], [2.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        labels = [0, 0, 1, 1, 2, 2]
        weights = fit_balanced_ridge_classifier(
            features, labels, ridge=0.01, class_count=3
        )
        probabilities = predict_ridge_probabilities(features, weights)
        summary = classification_summary(labels, probabilities)

        self.assertEqual(probabilities.shape, (6, 3))
        self.assertGreaterEqual(summary["accuracy"], 5 / 6)

    def test_high_confidence_open_prediction_stably_promotes_open_items(self) -> None:
        candidates = [
            {"rank": 1, "piece_id": 1, "name_en": "Orison Hood", "score": 1.0},
            {"rank": 2, "piece_id": 2, "name_en": "Ribbon", "score": 0.9},
            {"rank": 3, "piece_id": 3, "name_en": "Circlet", "score": 0.8},
        ]

        reranked = rerank_candidates_by_enclosing_probability(
            candidates, 0.05, rerank_k=3, minimum_confidence=0.9
        )

        self.assertEqual([row["piece_id"] for row in reranked], [2, 3, 1])

    def test_selects_lowest_threshold_meeting_training_precision(self) -> None:
        probabilities = np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.45, 0.55], [0.4, 0.6]],
            dtype=np.float32,
        )
        selected = select_confidence_threshold(
            [0, 0, 0, 1], probabilities, target_precision=1.0, minimum_count=2
        )

        self.assertAlmostEqual(selected["threshold"], 0.6, places=5)
        self.assertEqual(selected["count"], 3)

    def test_covered_router_groups_masks_with_hoods(self) -> None:
        candidates = [
            {"rank": 1, "piece_id": 1, "name_en": "Ribbon", "score": 1.0},
            {"rank": 2, "piece_id": 2, "name_en": "Eye Mask", "score": 0.9},
            {"rank": 3, "piece_id": 3, "name_en": "Orison Hood", "score": 0.8},
        ]

        reranked = rerank_candidates_by_covered_probability(
            candidates, 0.8, rerank_k=3, minimum_confidence=0.7
        )

        self.assertEqual([row["piece_id"] for row in reranked], [2, 3, 1])


if __name__ == "__main__":
    unittest.main()
