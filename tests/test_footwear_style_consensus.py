import unittest
from unittest.mock import patch

from ffxiv_prism.matching.garment_morphology import (
    StylePrediction, footwear_style_consensus_policy, footwear_style_records,
)
from ffxiv_prism.matching.inference import current_best_configs, load_current_best_selection
from scripts.update_active_candidate_indexes import active_target_plan


class FootwearStyleConsensusTests(unittest.TestCase):
    def prediction(self, tall):
        return StylePrediction.from_probabilities("FEET", (1-tall, tall))

    def test_stockings_context_cannot_override_disagreeing_shoe(self):
        self.assertFalse(footwear_style_consensus_policy(self.prediction(.9), [self.prediction(.15)]).active)

    def test_missing_uncertain_and_disagreeing_views_abstain(self):
        for views in ([], [.55], [.95, .55], [.95, .1]):
            self.assertFalse(footwear_style_consensus_policy(self.prediction(.9), [self.prediction(p) for p in views]).active)

    def test_agreed_tall_footwear_keeps_reranking(self):
        self.assertTrue(footwear_style_consensus_policy(self.prediction(.9), [self.prediction(.95), self.prediction(.85)]).active)
        self.assertFalse(footwear_style_consensus_policy(self.prediction(.4), [self.prediction(.95)]).active)

    def test_only_complete_visible_shoe_views_corroborate(self):
        rows = [dict(name="feet_full_left", visibility="visible", tags=[]),
                dict(name="feet_full_right", visibility="visible", tags=["source_edge_truncation_risk"]),
                dict(name="feet_boot_left", visibility="visible", tags=[]),
                dict(name="feet_full_left", visibility="partially_visible", tags=[])]
        self.assertEqual(footwear_style_records(rows), rows[:1])

    def test_old_gallery_fallback_and_new_branch_maintenance(self):
        selection = load_current_best_selection()
        selection.pop("feet_policy", None)
        selection["candidate_indexes"].pop("facebook-dinov3-vitb16-feet-multiscale-v1", None)
        with patch("ffxiv_prism.matching.inference.load_current_best_selection", return_value=selection):
            self.assertEqual(len(current_best_configs()["FEET"].ensemble_model_ids), 1)
        self.assertNotIn("feet_dinov3", {t.name for t in active_target_plan(selection, "20990101-000000")})
        selection["feet_policy"] = "dinov3-consensus"
        selection["candidate_indexes"]["facebook-dinov3-vitb16-feet-multiscale-v1"] = "data/indexes/example-feet"
        with patch("ffxiv_prism.matching.inference.load_current_best_selection", return_value=selection):
            config = current_best_configs()["FEET"]
            self.assertEqual(len(config.ensemble_model_ids), 2)
            self.assertTrue(config.footwear_style_consensus)
        self.assertIn("feet_dinov3", {t.name for t in active_target_plan(selection, "20990101-000000")})
