from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_feet_visibility_gate import (
    confusion_metrics,
    semantic_feet_features,
    stratified_split,
    visibility_rule,
)


class FeetVisibilityGateTests(unittest.TestCase):
    def test_semantic_features_distinguish_floor_garment_from_shoes(self) -> None:
        segmentation = np.zeros((10, 6), dtype=np.uint8)
        segmentation[1:9, 1:5] = 7

        features = semantic_feet_features(segmentation)

        self.assertEqual(features["shoe_share_person"], 0.0)
        self.assertEqual(features["lower_garment_bottom_ratio"], 1.0)
        self.assertEqual(
            features["lower_garment_bottom_band_share"],
            1.0,
        )

    def test_visibility_rule_requires_weak_shoe_evidence(self) -> None:
        features = {
            "shoe_share_person": 0.2,
            "lower_garment_bottom_ratio": 1.0,
            "lower_garment_bottom_band_share": 1.0,
            "person_bottom_edge_gap_ratio": 0.0,
        }

        predicted = visibility_rule(
            features,
            min_lower_bottom_ratio=0.9,
            min_bottom_band_share=0.5,
            max_shoe_share=0.01,
            max_edge_gap_ratio=0.0,
        )

        self.assertFalse(predicted)

    def test_confusion_metrics_reports_false_negatives(self) -> None:
        metrics = confusion_metrics(
            [True, True, False, False],
            [True, False, True, False],
        )

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)

    def test_stratified_split_places_each_label_in_both_roles(self) -> None:
        rows = [
            {"example_id": f"usable-{index}", "unusable": False}
            for index in range(4)
        ] + [
            {"example_id": f"unusable-{index}", "unusable": True}
            for index in range(4)
        ]

        assignments = stratified_split(rows)

        for prefix in ("usable", "unusable"):
            roles = {
                assignments[f"{prefix}-{index}"] for index in range(4)
            }
            self.assertEqual(roles, {"development", "holdout"})


if __name__ == "__main__":
    unittest.main()
