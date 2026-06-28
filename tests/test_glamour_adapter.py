from __future__ import annotations

import unittest

import torch

from ffxiv_prism.vision.adaptation import (
    SlotResidualAdapter,
    load_adapter_checkpoint,
)
from scripts.train_glamour_adapter import (
    cascade_outcomes,
    deterministic_glamour_split,
)


class GlamourAdapterTests(unittest.TestCase):
    def test_split_is_grouped_deterministic_and_balanced(self) -> None:
        glamour_ids = [str(value) for value in range(50)]
        first = deterministic_glamour_split(glamour_ids, seed=42)
        second = deterministic_glamour_split(reversed(glamour_ids), seed=42)

        self.assertEqual(first, second)
        self.assertEqual(list(first.values()).count("train"), 30)
        self.assertEqual(list(first.values()).count("validation"), 10)
        self.assertEqual(list(first.values()).count("test"), 10)

    def test_adapter_starts_as_identity(self) -> None:
        adapter = SlotResidualAdapter(dimension=8, rank=2)
        inputs = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)

        outputs = adapter(inputs, "BODY")

        self.assertTrue(torch.allclose(outputs, inputs, atol=1e-6))

    def test_checkpoint_loader_restores_adapter(self) -> None:
        adapter = SlotResidualAdapter(dimension=8, rank=2)
        with self.subTest("round trip"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "adapter.pt"
                torch.save(
                    {
                        "state_dict": adapter.state_dict(),
                        "dimension": 8,
                        "rank": 2,
                        "slots": list(adapter.slots),
                    },
                    path,
                )
                restored, payload = load_adapter_checkpoint(path)

                self.assertEqual(payload["dimension"], 8)
                self.assertEqual(restored.rank, 2)

    def test_cascade_uses_precision_top1_and_recall_tail(self) -> None:
        base = {
            "example_id": "example",
            "slot": "BODY",
            "status": "eligible",
            "eligible_expected_piece_ids": [2],
            "expected_appearance_key": "target",
        }
        precision = {
            **base,
            "results": [
                {"rank": 1, "piece_id": 1, "appearance_key": "precision"},
                {"rank": 2, "piece_id": 3, "appearance_key": "other"},
            ],
        }
        recall = {
            **base,
            "results": [
                {"rank": 1, "piece_id": 4, "appearance_key": "recall-first"},
                {"rank": 2, "piece_id": 2, "appearance_key": "target"},
            ],
        }

        [combined] = cascade_outcomes([precision], [recall], top_k=3)

        self.assertEqual(
            [row["piece_id"] for row in combined["results"]],
            [1, 4, 2],
        )
        self.assertEqual(combined["appearance_rank"], 3)
        self.assertTrue(combined["appearance_hit_at_k"])
