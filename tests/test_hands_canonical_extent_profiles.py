from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from scripts.build_hands_canonical_extent_profiles import (
    clip_feature_tensor,
    extent_class_from_features,
    name_extent_prior,
    parser_extent_features,
    select_profile_rows,
    stratified_qa_profiles,
)


class HandsCanonicalExtentProfilesTests(unittest.TestCase):
    def test_clip_feature_wrapper_accepts_tensor_or_pooler_output(self) -> None:
        tensor = np.asarray([[1.0, 2.0]])
        self.assertIs(clip_feature_tensor(tensor), tensor)

        class Output:
            pooler_output = tensor

        self.assertIs(clip_feature_tensor(Output()), tensor)

    def test_parser_coverage_uses_non_skin_labels_near_arm(self) -> None:
        segmentation = np.zeros((80, 80), dtype=np.uint8)
        segmentation[10:70, 30:40] = 14
        segmentation[45:70, 40:50] = 4
        features, selected = parser_extent_features(
            segmentation,
            dilation=21,
            minimum_arm_fraction=0.01,
        )
        self.assertEqual(features["valid_arm_count"], 1)
        self.assertIsNotNone(features["parser_extent_score"])
        self.assertGreater(float(features["parser_extent_score"]), 0.1)
        self.assertTrue(selected[50, 45])
        self.assertFalse(selected[5, 5])

    def test_visual_clip_routes_ambiguous_gloves(self) -> None:
        extent, score, source = extent_class_from_features(
            "Unclassified Gloves", (0.05, 0.05, 0.90)
        )
        self.assertEqual((extent, source), ("long", "canonical_visual_clip"))
        self.assertGreater(score, 0.9)
        extent, _, source = extent_class_from_features(
            "Unclassified Gloves", (1.0, 0.0, 0.0)
        )
        self.assertEqual((extent, source), ("short", "canonical_visual_clip"))

    def test_calibrated_name_prior_repairs_known_visual_blind_spots(self) -> None:
        self.assertEqual(name_extent_prior("Critical Hit Tour False Nails"), "short")
        self.assertEqual(name_extent_prior("Spaekona's Wrist Torques"), None)
        self.assertEqual(name_extent_prior("Augmented Evoker's Ringbands"), None)
        self.assertEqual(name_extent_prior("Theophany Shortgloves"), "short")
        self.assertEqual(name_extent_prior("Valerian Fingerless Halfgloves"), "short")
        self.assertEqual(name_extent_prior("Street Handwear"), None)
        extent, _, source = extent_class_from_features(
            "Courtly Lover's Armguards", (1.0, 0.0, 0.0)
        )
        self.assertEqual((extent, source), ("long", "calibrated_name_prior"))

    def test_extent_thresholds_are_ordered(self) -> None:
        self.assertEqual(
            extent_class_from_features("Unknown", (0.71, 0.0, 0.29))[0],
            "short",
        )
        self.assertEqual(
            extent_class_from_features("Unknown", (0.40, 0.60, 0.0))[0],
            "medium",
        )
        self.assertEqual(
            extent_class_from_features("Unknown", (0.55, 0.0, 0.45))[0],
            "long",
        )

    def test_profile_selection_is_piece_and_gender_specific(self) -> None:
        rows = [
            {
                "slot": "HANDS",
                "candidate_crop": "both_hands",
                "model_gender": "female",
                "race": "hyur",
                "dye_state": "normal",
                "local_path": "female.jpg",
                "pieces": [{"piece_id": 7, "name_en": "Seven"}],
            },
            {
                "slot": "HANDS",
                "candidate_crop": "both_hands",
                "model_gender": "male",
                "race": "hyur",
                "dye_state": "normal",
                "local_path": "male.jpg",
                "pieces": [{"piece_id": 7, "name_en": "Seven"}],
            },
        ]
        selected = select_profile_rows(rows, allowed_piece_genders={(7, "female")})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["local_path"], "female.jpg")

    def test_qa_sample_round_robins_extent_and_source_strata(self) -> None:
        rows = []
        piece_id = 0
        for extent in ("short", "long"):
            for source in ("calibrated_name_prior", "canonical_visual_clip"):
                for _ in range(3):
                    rows.append(
                        {
                            "piece_id": piece_id,
                            "model_gender": "female",
                            "extent_class": extent,
                            "extent_source": source,
                        }
                    )
                    piece_id += 1
        selected = stratified_qa_profiles(rows, limit=8)
        counts = Counter(
            (row["extent_class"], row["extent_source"]) for row in selected
        )
        self.assertEqual(set(counts.values()), {2})


if __name__ == "__main__":
    unittest.main()
