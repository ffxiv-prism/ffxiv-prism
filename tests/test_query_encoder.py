from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ffxiv_prism.jsonl import write_jsonl
from scripts.train_glamour_adapter import TrainingSample
from scripts.train_glamour_query_encoder import (
    LowRankResidualPooler,
    SlotAttentionPooler,
    apply_training_reviews,
    filter_crop_records_by_slots,
    group_training_samples,
    load_foreground_allowlist,
    prune_training_samples,
    unfreeze_last_blocks,
    validate_index_model_id,
)
import torch


class QueryEncoderTests(unittest.TestCase):
    def test_rejects_candidate_index_from_another_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            (index_dir / "summary.json").write_text(
                '{"model_id": "facebook/dinov2-base"}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_index_model_id(index_dir, "local/dinov3")

    def test_low_rank_adapter_starts_as_normalized_cls_identity(self) -> None:
        pooler = LowRankResidualPooler(8, rank=2)
        tokens = torch.randn(3, 5, 8)

        output = pooler(tokens, "BODY")
        expected = torch.nn.functional.normalize(tokens[:, 0], dim=-1)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_filters_crop_records_to_selected_slots(self) -> None:
        records = {
            "10:0": [
                {"slot": "BODY", "path": "body.jpg"},
                {"slot": "HANDS", "path": "hands.jpg"},
            ],
            "11:0": [{"slot": "HEAD", "path": "head.jpg"}],
        }

        filtered = filter_crop_records_by_slots(records, ["BODY", "HEAD"])

        self.assertEqual(
            filtered,
            {
                "10:0": [{"slot": "BODY", "path": "body.jpg"}],
                "11:0": [{"slot": "HEAD", "path": "head.jpg"}],
            },
        )

    def test_visual_review_can_reject_image_or_one_slot(self) -> None:
        records = {
            "reject": [{"slot": "BODY"}, {"slot": "HEAD"}],
            "partial": [{"slot": "BODY"}, {"slot": "FEET"}],
        }
        reviews = {
            "reject": {"decision": "reject"},
            "partial": {
                "decision": "accept",
                "slot_usability": {"FEET": "unusable"},
            },
        }

        filtered = apply_training_reviews(records, reviews)

        self.assertEqual(filtered["reject"], [])
        self.assertEqual(filtered["partial"], [{"slot": "BODY"}])

    def test_slot_attention_starts_as_cls_identity(self) -> None:
        pooler = SlotAttentionPooler(8)
        tokens = torch.randn(2, 5, 8)

        output = pooler(tokens, "BODY")
        expected = torch.nn.functional.normalize(tokens[:, 0], dim=-1)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_slot_attention_excludes_dinov3_register_tokens(self) -> None:
        pooler = SlotAttentionPooler(2, num_register_tokens=2)
        with torch.no_grad():
            pooler.queries["BODY"].copy_(torch.tensor([1.0, 0.0]))
            pooler.gates["BODY"].fill_(20.0)
            pooler.output["BODY"].weight.copy_(torch.eye(2))
        tokens = torch.tensor(
            [[[0.0, 1.0], [10.0, 0.0], [9.0, 0.0], [0.0, 2.0], [0.0, 2.0]]]
        )

        output = pooler(tokens, "BODY")

        self.assertGreater(float(output[0, 1].detach()), 0.99)

    def test_unfreezes_dinov3_style_encoder_blocks(self) -> None:
        class Encoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
                )

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Encoder()
                self.norm = torch.nn.LayerNorm(2)

        model = Model()
        names = unfreeze_last_blocks(model, 1)

        self.assertTrue(any(name.startswith("model.layer.1") for name in names))
        self.assertFalse(any(name.startswith("model.layer.0") for name in names))
        self.assertTrue(any(name.startswith("norm.") for name in names))

    def test_groups_multiple_crops_from_one_example_slot(self) -> None:
        samples = {
            slot: []
            for slot in ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
        }
        samples["BODY"] = [
            TrainingSample("a.jpg", (2, 3), 0.8),
            TrainingSample("b.jpg", (2, 3), 0.6),
        ]
        records = {
            "example:0": [
                {"path": Path("a.jpg"), "slot": "BODY"},
                {"path": Path("b.jpg"), "slot": "BODY"},
            ]
        }

        grouped = group_training_samples(samples, records)

        self.assertEqual(len(grouped["BODY"]), 1)
        self.assertEqual(grouped["BODY"][0].paths, ("a.jpg", "b.jpg"))
        self.assertEqual(grouped["BODY"][0].positives, (2, 3))
        self.assertAlmostEqual(grouped["BODY"][0].weight, 0.7)

    def test_can_group_multiple_views_of_one_glamour(self) -> None:
        samples = {
            slot: []
            for slot in ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
        }
        samples["FEET"] = [
            TrainingSample("front.jpg", (7,), 0.9),
            TrainingSample("side.jpg", (7,), 0.5),
        ]
        records = {
            "42:0": [{"path": Path("front.jpg"), "slot": "FEET"}],
            "42:1": [{"path": Path("side.jpg"), "slot": "FEET"}],
        }

        grouped = group_training_samples(
            samples,
            records,
            group_ids={"42:0": "42", "42:1": "42"},
        )

        self.assertEqual(len(grouped["FEET"]), 1)
        self.assertEqual(grouped["FEET"][0].paths, ("front.jpg", "side.jpg"))
        self.assertAlmostEqual(grouped["FEET"][0].weight, 0.7)

    def test_domain_weight_scales_group_loss_weight(self) -> None:
        samples = {
            slot: []
            for slot in ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
        }
        samples["BODY"] = [TrainingSample("rare.jpg", (4,), 0.5)]
        records = {
            "9:0": [{"path": Path("rare.jpg"), "slot": "BODY"}],
        }

        grouped = group_training_samples(
            samples,
            records,
            group_weights={"9:0": 2.0},
        )

        self.assertAlmostEqual(grouped["BODY"][0].weight, 1.0)

    def test_distinct_group_weights_are_applied_to_the_right_groups(self) -> None:
        samples = {
            slot: []
            for slot in ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
        }
        samples["BODY"] = [
            TrainingSample("old.jpg", (4,), 0.5),
            TrainingSample("new.jpg", (5,), 0.5),
        ]
        records = {
            "old:0": [{"path": Path("old.jpg"), "slot": "BODY"}],
            "new:0": [{"path": Path("new.jpg"), "slot": "BODY"}],
        }

        grouped = group_training_samples(
            samples,
            records,
            group_weights={"old:0": 2.0, "new:0": 0.5},
        )
        weights_by_path = {
            sample.paths[0]: sample.weight
            for sample in grouped["BODY"]
        }

        self.assertAlmostEqual(weights_by_path["old.jpg"], 1.0)
        self.assertAlmostEqual(weights_by_path["new.jpg"], 0.25)

    def test_foreground_allowlist_honors_flags_and_visual_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "example_id": "clean",
                        "automatic_stats": {"quality_flags": []},
                        "vision_review": {"decision": None},
                    },
                    {
                        "example_id": "flagged",
                        "automatic_stats": {"quality_flags": ["large_hole_repair"]},
                        "vision_review": {"decision": None},
                    },
                    {
                        "example_id": "accepted",
                        "automatic_stats": {"quality_flags": ["large_hole_repair"]},
                        "vision_review": {"decision": "accept"},
                    },
                ],
            )

            allowed = load_foreground_allowlist(path)

        self.assertEqual(allowed, {"clean", "accepted"})

    def test_hand_training_keeps_complete_arm_context(self) -> None:
        samples = {
            slot: []
            for slot in ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
        }
        records = {"example:0": []}
        for index, (name, role) in enumerate(
            (
                ("hands_left", "primary"),
                ("hands_right", "primary"),
                ("hands_forearm_left", "auxiliary"),
                ("hands_forearm_right", "auxiliary"),
                ("hands_arm_cover_context", "context"),
            )
        ):
            path = f"{index}.jpg"
            samples["HANDS"].append(TrainingSample(path, (1,), 1.0))
            records["example:0"].append(
                {
                    "path": Path(path),
                    "slot": "HANDS",
                    "name": name,
                    "role": role,
                    "visibility": "visible",
                    "confidence": 1.0,
                    "tags": [],
                }
            )

        selected = prune_training_samples(samples, records, 3)

        self.assertEqual(len(selected["HANDS"]), 3)
        self.assertIn(
            "4.jpg",
            {sample.path for sample in selected["HANDS"]},
        )


if __name__ == "__main__":
    unittest.main()
