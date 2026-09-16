from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ffxiv_prism.matching.inference import (
    SlotInferenceConfig,
    apply_ensemble_top1_gate,
    apply_multislot_consistency_gate,
    apply_occupied_slot_top1_gate,
    constrain_candidates_to_pool,
    crop_records,
    crop_weight,
    current_best_configs,
    fuse_branch_search_results,
    fuse_crops_then_branch_results,
    fuse_crop_results,
    inspect_assets,
    no_visible_equipment_decision,
    attach_appearance_equivalent_candidates,
    collapse_appearance_equivalent_candidates,
    prepend_no_equipment_candidate,
    rerank_occupied_slot_search_results,
    restore_body_supported_occupied_hands_top1,
    select_slot_candidate_index_rows,
    slot_region_presence_decision,
    slot_result,
    validate_input_image,
    write_html_report,
)
from ffxiv_prism.matching.spatial_profiles import (
    SpatialProfileIndex,
    rerank_candidates_with_spatial_profile,
)
from ffxiv_prism.paths import PROJECT_ROOT
from ffxiv_prism.vision.outfit_regions import CropRegion


class GlamourInferenceTests(unittest.TestCase):
    def test_diagnostic_crop_is_excluded_from_retrieval_weight(self) -> None:
        record = self.record("full-shoe.jpg")
        record["role"] = "diagnostic"
        self.assertEqual(crop_weight(record), 0.0)

    def test_style_policy_can_reduce_a_cross_slot_crop_weight(self) -> None:
        record = self.record("legs-context.jpg")
        record["style_weight_multiplier"] = 0.25
        self.assertAlmostEqual(crop_weight(record), 0.25)

    def test_short_legwear_policy_activates_waist_thigh_retrieval_crop(
        self,
    ) -> None:
        from ffxiv_prism.matching.garment_morphology import (
            StylePrediction,
            apply_style_crop_multipliers,
            style_policy,
        )

        prediction = StylePrediction.from_probabilities(
            "LEGS",
            (0.80, 0.05, 0.05, 0.05, 0.05),
        )
        record = {
            "name": "legs_waist_thighs",
            "role": "diagnostic",
            "visibility": "visible",
            "confidence": 0.9,
        }

        adjusted = apply_style_crop_multipliers(
            [record],
            style_policy(prediction, minimum_confidence=0.5),
        )[0]

        self.assertEqual(adjusted["role"], "auxiliary")
        self.assertGreater(crop_weight(adjusted), 0.0)
        self.assertEqual(record["role"], "diagnostic")

        long_prediction = StylePrediction.from_probabilities(
            "LEGS",
            (0.02, 0.02, 0.02, 0.02, 0.92),
        )
        unchanged = apply_style_crop_multipliers(
            [record],
            style_policy(long_prediction, minimum_confidence=0.5),
        )[0]
        self.assertEqual(unchanged["role"], "diagnostic")
        self.assertEqual(crop_weight(unchanged), 0.0)

    def test_source_edge_risk_is_reported_as_low_ranking_quality(self) -> None:
        record = self.record("shoe.jpg")
        record["tags"] = ("source_edge_truncation_risk",)
        result = slot_result(
            SlotInferenceConfig("FEET", Path("index"), None),
            [record],
            [self.candidate(1, 0.5, 1), self.candidate(2, 0.4, 2)],
        )
        self.assertEqual(result["ranking_quality"]["level"], "low")
        self.assertIn("full shoe", result["ranking_quality"]["reasons"][0])

        head_result = slot_result(
            SlotInferenceConfig("HEAD", Path("index"), None),
            [record],
            [self.candidate(1, 0.5, 1), self.candidate(2, 0.4, 2)],
        )
        self.assertIn(
            "full headwear",
            head_result["ranking_quality"]["reasons"][0],
        )

    def test_fuses_max_scores_and_deduplicates_piece(self) -> None:
        records = [
            self.record("a.jpg", confidence=0.9),
            self.record("b.jpg", confidence=0.8),
        ]
        search = {
            "a.jpg": [self.candidate(7, 0.6, 1), self.candidate(8, 0.5, 2)],
            "b.jpg": [self.candidate(7, 0.7, 1), self.candidate(9, 0.4, 2)],
        }
        ranked = fuse_crop_results(records, search, top_k=3)
        self.assertEqual([row["piece_id"] for row in ranked], [7, 8, 9])
        self.assertEqual(len(ranked[0]["evidence"]), 2)

    def test_rrf_rewards_consensus(self) -> None:
        records = [self.record("a.jpg"), self.record("b.jpg")]
        search = {
            "a.jpg": [self.candidate(1, 0.9, 1), self.candidate(2, 0.8, 2)],
            "b.jpg": [self.candidate(3, 0.9, 1), self.candidate(2, 0.8, 2)],
        }
        ranked = fuse_crop_results(records, search, top_k=3, fusion="rrf", rrf_k=1)
        self.assertEqual(ranked[0]["piece_id"], 2)

    def test_branch_rrf_rewards_cross_model_consensus(self) -> None:
        searches = [
            {"a.jpg": [self.candidate(1, 0.9, 1), self.candidate(2, 0.8, 2)]},
            {"a.jpg": [self.candidate(3, 0.9, 1), self.candidate(2, 0.8, 2)]},
        ]
        fused = fuse_branch_search_results(searches, top_k=3, rrf_k=1)
        self.assertEqual(fused["a.jpg"][0]["piece_id"], 2)

    def test_branch_rrf_supports_frozen_branch_weights(self) -> None:
        primary = {
            "a.jpg": [self.candidate(1, 0.9, 1), self.candidate(2, 0.8, 2)]
        }
        adapted = {
            "a.jpg": [self.candidate(2, 0.9, 1), self.candidate(1, 0.8, 2)]
        }
        fused = fuse_branch_search_results(
            [primary, adapted],
            top_k=2,
            rrf_k=1,
            branch_weights=(1.0, 1.25),
        )
        self.assertEqual(fused["a.jpg"][0]["piece_id"], 2)

    def test_branch_rrf_can_preserve_a_fixed_candidate_pool(self) -> None:
        primary = {
            "a.jpg": [self.candidate(1, 0.9, 1), self.candidate(2, 0.8, 2)]
        }
        weak_extra = {
            "a.jpg": [self.candidate(3, 0.9, 1), self.candidate(2, 0.8, 2)]
        }
        pool = fuse_branch_search_results([primary], top_k=2, rrf_k=1)
        fused = fuse_branch_search_results(
            [primary, weak_extra],
            top_k=2,
            rrf_k=1,
            candidate_pool=pool,
        )
        self.assertEqual({row["piece_id"] for row in fused["a.jpg"]}, {1, 2})

    def test_crop_then_branch_fusion_preserves_primary_candidate_pool(self) -> None:
        records = [self.record("a.jpg"), self.record("b.jpg")]
        primary = {
            "a.jpg": [self.candidate(1, 0.9, 1), self.candidate(2, 0.8, 2)],
            "b.jpg": [self.candidate(1, 0.7, 1), self.candidate(2, 0.6, 2)],
        }
        extra = {
            "a.jpg": [self.candidate(3, 0.9, 1), self.candidate(2, 0.8, 2)],
            "b.jpg": [self.candidate(3, 0.7, 1), self.candidate(2, 0.6, 2)],
        }
        fused = fuse_crops_then_branch_results(
            records,
            [primary, extra],
            top_k=2,
            candidate_k=2,
            crop_fusion="max",
            crop_rrf_k=60,
            branch_rrf_k=1,
            final_pool_branches=1,
        )
        self.assertEqual([row["piece_id"] for row in fused], [2, 1])

    def test_body_uses_validated_crop_then_branch_safe_baseline(self) -> None:
        config = current_best_configs()["BODY"]
        self.assertEqual(config.ensemble_fusion_order, "crop_then_branch")
        self.assertEqual(config.ensemble_final_pool_branches, 1)
        self.assertEqual(config.ensemble_model_ids, (
            "var/models/dinov3-vitb16-pretrain-lvd1689m",
        ))
        self.assertEqual(config.ensemble_query_checkpoints, (None,))
        self.assertEqual(config.ensemble_branch_weights, (1.0, 1.0))
        self.assertEqual(config.ensemble_gate_baseline_branch_weights, ())
        self.assertIsNone(config.ensemble_top1_gate_max_base_rank)
        self.assertEqual(
            config.style_model_id,
            "google/siglip2-base-patch16-224",
        )
        self.assertEqual(config.style_query_crop_name, "body_context")
        self.assertEqual(config.style_rerank_k, 100)
        self.assertAlmostEqual(config.style_blend, 0.70)
        self.assertAlmostEqual(config.benchmark_top1 or 0.0, 0.5972359328726555)

    def test_ensemble_top1_gate_promotes_supported_candidate_only(self) -> None:
        baseline = [
            self.candidate(1, 0.9, 1),
            self.candidate(2, 0.8, 2),
            self.candidate(3, 0.7, 3),
        ]
        candidate = [
            self.candidate(2, 1.0, 1),
            self.candidate(1, 0.9, 2),
            self.candidate(3, 0.7, 3),
        ]
        adapted = [
            self.candidate(2, 1.0, 1),
            self.candidate(1, 0.9, 2),
            self.candidate(3, 0.7, 3),
        ]
        result = apply_ensemble_top1_gate(
            baseline,
            candidate,
            adapted,
            max_base_rank=10,
            max_last_branch_rank=1,
            max_base_last_branch_rank=10,
            min_last_branch_rank_ratio=1.0,
        )
        self.assertEqual([row["piece_id"] for row in result], [2, 1, 3])
        self.assertTrue(result[0]["ensemble_top1_gate"]["promoted"])

    def test_ensemble_top1_gate_rejects_when_old_winner_lacks_support(self) -> None:
        baseline = [self.candidate(piece_id, 1.0, piece_id) for piece_id in range(1, 12)]
        candidate = [baseline[1], baseline[0], *baseline[2:]]
        adapted = [baseline[1], *baseline[2:], baseline[0]]
        result = apply_ensemble_top1_gate(
            baseline,
            candidate,
            adapted,
            max_base_rank=10,
            max_last_branch_rank=1,
            max_base_last_branch_rank=10,
            min_last_branch_rank_ratio=1.0,
        )
        self.assertEqual([row["piece_id"] for row in result], list(range(1, 12)))

    def test_slot_index_selection_keeps_vector_rows_aligned(self) -> None:
        metadata = [
            {"slot": "HEAD", "local_path": "head.png"},
            {"slot": "BODY", "local_path": "body-a.png"},
            {"slot": "BODY", "local_path": "body-b.png"},
        ]
        embeddings = np.asarray(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float32,
        )
        selected_metadata, selected_embeddings = select_slot_candidate_index_rows(
            metadata,
            embeddings,
            "BODY",
        )
        self.assertEqual(
            [row["local_path"] for row in selected_metadata],
            ["body-a.png", "body-b.png"],
        )
        np.testing.assert_array_equal(
            selected_embeddings,
            np.asarray([[2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        )

    def test_promoted_multi_crop_slots_use_validated_rrf(self) -> None:
        configs = current_best_configs()
        for slot in ("HANDS", "LEGS", "FEET"):
            with self.subTest(slot=slot):
                self.assertEqual(configs[slot].fusion, "rrf")
                self.assertEqual(configs[slot].rrf_k, 1)

    def test_feet_uses_confidence_gated_height_reranker(self) -> None:
        config = current_best_configs()["FEET"]
        self.assertEqual(
            config.style_model_id,
            "google/siglip2-base-patch16-224",
        )
        self.assertEqual(config.style_query_crop_name, "feet_height_context")
        self.assertEqual(config.style_rerank_k, 50)
        self.assertAlmostEqual(config.style_blend, 0.70)
        self.assertAlmostEqual(config.style_min_confidence, 0.70)

    def test_hands_uses_frozen_occupied_slot_top1_gate(self) -> None:
        config = current_best_configs()["HANDS"]
        self.assertEqual(config.occupied_slot_multiplier, 0.25)
        self.assertEqual(config.occupied_slot_gate_max_base_rank, 3)
        self.assertEqual(config.occupied_slot_body_support_top_k, 3)
        self.assertAlmostEqual(config.benchmark_top1 or 0.0, 0.03023983315954119)

    def test_occupied_slot_gate_promotes_nearby_exact_hands_piece(self) -> None:
        occupied = {
            **self.candidate(1, 0.9, 1),
            "link_type": "occupied_slot",
        }
        exact = {**self.candidate(2, 0.8, 2), "link_type": "exact_slot"}
        adjusted = rerank_occupied_slot_search_results(
            {"crop.jpg": [occupied, exact]}, occupied_multiplier=0.25
        )["crop.jpg"]
        result = apply_occupied_slot_top1_gate(
            [occupied, exact], adjusted, max_base_rank=3
        )
        self.assertEqual([row["piece_id"] for row in result], [2, 1])
        self.assertTrue(result[0]["occupied_slot_gate"]["promoted"])

    def test_body_support_restores_matching_multislot_hands_candidate(self) -> None:
        body = {
            "slot": "BODY",
            "candidates": [
                {"rank": 1, "piece_id": 50, "name_en": "Fat Cat Loungewear"}
            ],
        }
        promoted = {
            **self.candidate(2, 0.8, 1),
            "name_en": "Wristbands",
            "occupied_slot_gate": {
                "promoted": True,
                "base_rank": 2,
                "demoted_piece_id": 1,
            },
            "reference_match": False,
        }
        demoted = {
            **self.candidate(1, 0.9, 2),
            "name_en": "Fat Cat Loungewear",
            "reference_match": True,
        }
        hands = {
            "slot": "HANDS",
            "candidates": [promoted, demoted],
            "reference": {},
            "top_score_margin": None,
        }
        result = restore_body_supported_occupied_hands_top1(
            [body, hands], body_support_top_k=3
        )[1]
        self.assertEqual([row["piece_id"] for row in result["candidates"]], [1, 2])
        self.assertTrue(result["reference"]["top1_hit"])
        self.assertTrue(result["candidates"][0]["cross_slot_consistency"]["restored"])

    def test_body_mismatch_keeps_exact_hands_promotion(self) -> None:
        body = {
            "slot": "BODY",
            "candidates": [{"rank": 1, "piece_id": 50, "name_en": "Troian Lorica"}],
        }
        hands = {
            "slot": "HANDS",
            "candidates": [
                {
                    **self.candidate(2, 0.8, 1),
                    "name_en": "Armguards",
                    "occupied_slot_gate": {
                        "promoted": True,
                        "base_rank": 3,
                        "demoted_piece_id": 1,
                    },
                },
                {**self.candidate(1, 0.9, 2), "name_en": "Other Attire"},
            ],
        }
        result = restore_body_supported_occupied_hands_top1(
            [body, hands], body_support_top_k=3
        )[1]
        self.assertEqual(result["candidates"][0]["piece_id"], 2)

    def test_multislot_candidate_is_removed_when_another_slot_lacks_support(
        self,
    ) -> None:
        body = {
            "slot": "BODY",
            "candidates": [
                {
                    **self.candidate(10, 0.95, 1),
                    "name_en": "One-piece Attire",
                    "link_type": "exact_slot",
                },
                self.candidate(20, 0.90, 2),
            ],
        }
        hands = {
            "slot": "HANDS",
            "candidates": [
                self.candidate(30, 0.90, 1),
                {
                    **self.candidate(10, 0.50, 8),
                    "name_en": "One-piece Attire",
                    "link_type": "occupied_slot",
                },
            ],
        }

        results = apply_multislot_consistency_gate(
            [body, hands],
            {10: ("BODY", "HANDS")},
            support_top_k=5,
        )

        self.assertEqual(
            [row["piece_id"] for row in results[0]["candidates"]],
            [20],
        )
        self.assertEqual(
            [row["piece_id"] for row in results[1]["candidates"]], [30]
        )
        self.assertEqual(
            results[0]["cross_slot_consistency"]["removed_candidate_count"], 1
        )
        self.assertTrue(
            results[0]["cross_slot_consistency"]["ranking_changed"]
        )

    def test_multislot_candidate_stays_reliable_with_cross_slot_support(
        self,
    ) -> None:
        body_candidate = {
            **self.candidate(10, 0.95, 1),
            "name_en": "One-piece Attire",
        }
        hands_candidate = {
            **self.candidate(10, 0.90, 2),
            "name_en": "One-piece Attire",
        }

        results = apply_multislot_consistency_gate(
            [
                {"slot": "BODY", "candidates": [body_candidate]},
                {
                    "slot": "HANDS",
                    "candidates": [self.candidate(30, 1.0, 1), hands_candidate],
                },
            ],
            {10: ("BODY", "HANDS")},
            support_top_k=5,
        )

        self.assertTrue(results[0]["candidates"][0]["recommendation_reliable"])
        self.assertEqual(
            results[0]["candidates"][0]["cross_slot_consistency"]["state"],
            "supported",
        )

    def test_no_visible_equipment_requires_every_crop_to_be_confident(self) -> None:
        absent = no_visible_equipment_decision(
            ((0.94, 0.06), (0.88, 0.12)),
            minimum_probability=0.80,
        )
        mixed = no_visible_equipment_decision(
            ((0.94, 0.06), (0.35, 0.65)),
            minimum_probability=0.80,
        )

        self.assertTrue(absent["no_visible_equipment"])
        self.assertFalse(mixed["no_visible_equipment"])

    def test_no_equipment_is_prepended_without_discarding_visual_candidates(
        self,
    ) -> None:
        diagnostic = no_visible_equipment_decision(
            ((0.94, 0.06), (0.88, 0.12)),
            minimum_probability=0.80,
        )
        result = prepend_no_equipment_candidate(
            [self.candidate(1, 0.8, 1), self.candidate(2, 0.7, 2)],
            {
                "slot": "HEAD",
                "game_item_id": 10032,
                "name_en": "The Emperor's New Hat",
                "name_zh": "皇帝的新帽子",
                "source": "https://example.invalid/items",
            },
            diagnostic,
        )

        self.assertEqual([row["rank"] for row in result], [1, 2, 3])
        self.assertEqual([row["piece_id"] for row in result], [-10032, 1, 2])
        self.assertTrue(result[0]["is_no_equipment"])
        self.assertEqual(result[0]["game_item_id"], 10032)
        self.assertAlmostEqual(result[0]["no_equipment_score"], 0.88)

    def test_slot_presence_distinguishes_bare_from_out_of_frame(self) -> None:
        visible_bare = self.record("hand.jpg")
        out_of_frame = self.record("leg.jpg")
        out_of_frame["visibility"] = "out_of_frame"
        out_of_frame["confidence"] = 0.1

        self.assertTrue(
            slot_region_presence_decision([visible_bare])["contains_slot"]
        )
        missing = slot_region_presence_decision([out_of_frame])
        self.assertFalse(missing["contains_slot"])
        self.assertEqual(
            missing["reason"],
            "slot_crops_out_of_frame_or_pose_uncertain",
        )

    def test_emperor_no_equipment_catalog_contains_all_core_slots(self) -> None:
        from ffxiv_prism.cli.match import _load_emperor_no_equipment_catalog

        catalog = _load_emperor_no_equipment_catalog()

        self.assertEqual(
            {slot: row["game_item_id"] for slot, row in catalog.items()},
            {
                "HEAD": 10032,
                "BODY": 10033,
                "HANDS": 10034,
                "LEGS": 10035,
                "FEET": 10036,
            },
        )
        self.assertEqual(catalog["HANDS"]["name_zh"], "皇帝的新手套")

    def test_promoted_rank_ensemble_slots_have_secondary_indexes(self) -> None:
        configs = current_best_configs()
        self.assertEqual(
            configs["HEAD"].ensemble_model_ids,
            (
                "google/siglip2-base-patch16-224",
                "var/models/dinov3-vitb16-pretrain-lvd1689m",
            ),
        )
        self.assertEqual(len(configs["HEAD"].ensemble_index_dirs), 2)
        self.assertEqual(
            configs["HEAD"].ensemble_query_checkpoints,
            (
                None,
                PROJECT_ROOT
                / "data"
                / "models"
                / "current-best"
                / "refresh_v110_head_fold01_dinov3_e3.pt",
            ),
        )
        self.assertEqual(configs["HEAD"].ensemble_rrf_k, 10)
        self.assertEqual(configs["HEAD"].ensemble_pool_branches, 2)
        self.assertEqual(
            configs["HEAD"].morphology_checkpoint,
            PROJECT_ROOT
            / "data"
            / "models"
            / "current-best"
            / "head_morphology_v73_fold01_qv65_upper_binary_ridge.npz",
        )
        self.assertEqual(
            configs["FEET"].ensemble_model_ids,
            ("google/siglip2-base-patch16-224",),
        )
        self.assertEqual(len(configs["FEET"].ensemble_index_dirs), 1)
        self.assertEqual(configs["FEET"].ensemble_query_checkpoints, (None,))
        self.assertEqual(configs["FEET"].ensemble_rrf_k, 1)
        self.assertIsNone(configs["FEET"].ensemble_final_pool_branches)
        self.assertEqual(
            configs["LEGS"].ensemble_model_ids,
            ("var/models/dinov3-vitb16-pretrain-lvd1689m",),
        )
        self.assertEqual(configs["LEGS"].ensemble_final_pool_branches, 1)

    def test_promoted_legs_uses_final_dinov2_candidate_pool(self) -> None:
        config = current_best_configs()["LEGS"]
        self.assertIsNone(config.spatial_index)
        self.assertEqual(config.ensemble_rrf_k, 1)
        self.assertEqual(config.ensemble_pool_branches, 1)
        self.assertEqual(
            config.style_model_id,
            "google/siglip2-base-patch16-224",
        )
        self.assertEqual(config.style_query_crop_name, "legs_waist_thighs")
        self.assertEqual(config.style_rerank_k, 100)
        self.assertAlmostEqual(config.style_blend, 0.30)

    def test_final_candidate_pool_restores_missing_primary_rows(self) -> None:
        pool = [self.candidate(1, 1.0, 1), self.candidate(2, 0.9, 2)]
        reranked = [self.candidate(3, 1.1, 1), self.candidate(2, 1.0, 2)]

        result = constrain_candidates_to_pool(pool, reranked)

        self.assertEqual([row["piece_id"] for row in result], [2, 1])

    def test_exact_model_alternative_is_promoted_next_to_visual_anchor(
        self,
    ) -> None:
        candidates = [
            self.candidate(1, 1.0, 1),
            self.candidate(2, 0.9, 2),
        ]

        result = attach_appearance_equivalent_candidates(
            candidates,
            {1: (3, 4), 3: (1,), 4: (1,)},
            piece_catalog={
                3: {
                    "piece_id": 3,
                    "name_en": "Equivalent Piece",
                    "name_zh": "同模装备",
                },
                4: {
                    "piece_id": 4,
                    "name_en": "Another Equivalent Piece",
                    "name_zh": "另一件同模装备",
                },
            },
            max_equivalents_per_anchor=None,
        )

        self.assertEqual([row["piece_id"] for row in result], [1, 2])
        self.assertEqual(
            [
                row["piece_id"]
                for row in result[0]["appearance_alternatives"]
            ],
            [3, 4],
        )
        self.assertEqual(
            result[0]["appearance_alternatives"][0]["piece_id"],
            3,
        )
        self.assertEqual(
            result[0]["appearance_alternatives"][0]["name_zh"],
            "同模装备",
        )
        self.assertEqual(
            result[0]["appearance_alternatives"][0]["relation"],
            "same_model",
        )

    def test_same_model_candidates_collapse_to_highest_ranked_anchor(self) -> None:
        candidates = [
            self.candidate(10, 0.95, 1),
            self.candidate(20, 0.90, 2),
            self.candidate(30, 0.85, 3),
            self.candidate(40, 0.80, 4),
        ]
        equivalence = {
            10: (20,),
            # Deliberately one-way and transitive to exercise normalization.
            20: (30,),
        }
        attached = attach_appearance_equivalent_candidates(
            candidates,
            equivalence,
            max_equivalents_per_anchor=None,
        )

        result = collapse_appearance_equivalent_candidates(
            attached,
            equivalence,
        )

        self.assertEqual([row["piece_id"] for row in result], [10, 40])
        self.assertEqual([row["rank"] for row in result], [1, 2])
        self.assertEqual(
            result[0]["appearance_alternatives"][0]["piece_id"],
            20,
        )
        self.assertEqual(
            result[0]["appearance_alternatives"][0]["original_rank"],
            2,
        )

    def test_versioned_same_model_equivalence_is_loadable(self) -> None:
        from ffxiv_prism.cli import match as match_cli

        with patch.object(match_cli, "_APPEARANCE_EQUIVALENCE", None):
            equivalents = match_cli._load_appearance_equivalence()

        self.assertIn(247, equivalents[236])
        self.assertIn(4079, equivalents[4064])

    def test_missing_same_model_equivalence_fails_closed(self) -> None:
        from ffxiv_prism.cli import match as match_cli

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-equivalence.json"
            with (
                patch.object(
                    match_cli,
                    "APPEARANCE_EQUIVALENCE_PATH",
                    missing,
                ),
                patch.object(match_cli, "_APPEARANCE_EQUIVALENCE", None),
                self.assertRaisesRegex(
                    FileNotFoundError,
                    "Required same-model equivalence asset is missing",
                ),
            ):
                match_cli._load_appearance_equivalence()

    def test_spatial_profile_can_correct_adjacent_retrieval_order(self) -> None:
        candidates = [
            self.candidate(piece_id, 11.0 - piece_id, piece_id)
            for piece_id in range(1, 11)
        ]
        profiles = np.full((10, 1, 2), 0.5, dtype=np.float16)
        profiles[0, 0] = (0.0, 1.0)
        profiles[1, 0] = (1.0, 0.0)
        index = SpatialProfileIndex(
            profiles=profiles,
            projection=np.eye(2, dtype=np.float32),
            metadata_indices=np.arange(10),
            image_row_ids=np.arange(10),
            piece_ids_by_profile=tuple((piece_id,) for piece_id in range(1, 11)),
            model_id="test",
            preprocess="gray-row-profile-v1",
            candidate_crop="full",
        )
        reranked = rerank_candidates_with_spatial_profile(
            candidates,
            index,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            rerank_k=10,
            blend=0.35,
        )
        self.assertEqual(reranked[0]["piece_id"], 2)
        self.assertEqual(reranked[0]["retrieval_score"], 9.0)
        self.assertIn("spatial_score", reranked[0])
    def test_crop_records_require_one_path_per_region(self) -> None:
        region = CropRegion("BODY", "body", (0, 0, 20, 20), 0.8, "visible")
        with self.assertRaises(ValueError):
            crop_records([region], [])

    def test_validates_real_image_and_rejects_small_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (220, 240), "red").save(path)
            self.assertEqual(validate_input_image(path)["width"], 220)
            Image.new("RGB", (100, 100), "red").save(path)
            with self.assertRaises(ValueError):
                validate_input_image(path)

    def test_asset_inspection_detects_aligned_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.jsonl").write_text(
                '{"slot":"BODY"}\n', encoding="utf-8"
            )
            np.save(root / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
            config = SlotInferenceConfig("BODY", root, None)
            report = inspect_assets({"BODY": config})
            self.assertTrue(report["ok"])
            self.assertEqual(report["slots"]["BODY"]["embedding_shape"], [1, 4])

    def test_html_report_escapes_candidate_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            write_html_report(
                path,
                {
                    "input": {},
                    "artifacts": {},
                    "warnings": [],
                    "slots": [
                        {
                            "slot": "BODY",
                            "benchmark": {},
                            "reference": {
                                "item_name": "Expected <Body>",
                                "rank": 1,
                                "top1_hit": True,
                                "candidate_k": 1000,
                            },
                            "candidates": [
                                {
                                    "rank": 1,
                                    "piece_id": 2,
                                    "name_en": "<script>alert(1)</script>",
                                    "score": 0.5,
                                    "reference_match": True,
                                }
                            ],
                        }
                    ],
                },
            )
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("<script>alert", content)
            self.assertIn("&lt;script&gt;", content)
            self.assertIn("Expected &lt;Body&gt;", content)
            self.assertIn("Known ground truth", content)

    def test_html_report_labels_no_equipment_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            write_html_report(
                path,
                {
                    "input": {},
                    "artifacts": {},
                    "warnings": [],
                    "slots": [
                        {
                            "slot": "HEAD",
                            "benchmark": {},
                            "candidates": [
                                {
                                    "rank": 1,
                                    "piece_id": -10032,
                                    "game_item_id": 10032,
                                    "name_en": "The Emperor's New Hat",
                                    "name_zh": "皇帝的新帽子",
                                    "score": 0.88,
                                    "no_equipment_score": 0.88,
                                    "is_no_equipment": True,
                                }
                            ],
                        }
                    ],
                },
            )

            content = path.read_text(encoding="utf-8")
            self.assertIn("皇帝的新帽子", content)
            self.assertIn("未检测到可见装备", content)
            self.assertIn("game_item_id=10032", content)
            self.assertNotIn("piece_id=-10032", content)

    def test_html_report_displays_same_model_names_below_candidate_name(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            write_html_report(
                path,
                {
                    "input": {},
                    "artifacts": {},
                    "warnings": [],
                    "slots": [
                        {
                            "slot": "BODY",
                            "benchmark": {},
                            "candidates": [
                                {
                                    "rank": 1,
                                    "piece_id": 1,
                                    "name_en": "Current Piece",
                                    "name_zh": "当前装备",
                                    "score": 0.88,
                                    "appearance_alternatives": [
                                        {
                                            "piece_id": 2,
                                            "name_en": "Equivalent <Coat>",
                                            "name_zh": "同模<外套>",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
            )

            content = path.read_text(encoding="utf-8")
            name_position = content.index("当前装备")
            same_model_position = content.index("同模装备：")
            identifier_position = content.index("piece_id=1")
            self.assertLess(name_position, same_model_position)
            self.assertLess(same_model_position, identifier_position)
            self.assertIn("同模&lt;外套&gt;", content)
            self.assertIn("Equivalent &lt;Coat&gt;", content)

    @staticmethod
    def record(path: str, confidence: float = 1.0) -> dict[str, object]:
        return {
            "path": path,
            "name": path,
            "slot": "BODY",
            "visibility": "visible",
            "confidence": confidence,
            "role": "primary",
        }

    @staticmethod
    def candidate(piece_id: int, score: float, rank: int) -> dict[str, object]:
        return {
            "piece_id": piece_id,
            "score": score,
            "rank": rank,
            "name_en": f"Piece {piece_id}",
            "local_path": f"piece-{piece_id}.png",
        }


if __name__ == "__main__":
    unittest.main()
