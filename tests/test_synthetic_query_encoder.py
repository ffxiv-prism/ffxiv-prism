from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.train_synthetic_query_encoder import (
    build_catalog_samples,
    collect_piece_ids_for_role,
    load_synthetic_batch,
    split_catalog_samples,
)


class SyntheticQueryEncoderTests(unittest.TestCase):
    def test_piece_scope_uses_only_eligible_examples_in_requested_role(self) -> None:
        examples = [
            {
                "example_id": "1:0",
                "glamour_id": 1,
                "core_slot_labels": {
                    "HANDS": {"piece_ids": [10, 11]},
                },
            },
            {
                "example_id": "2:0",
                "glamour_id": 2,
                "core_slot_labels": {
                    "HANDS": {"piece_ids": [20]},
                },
            },
        ]

        pieces = collect_piece_ids_for_role(
            examples,
            {"1:0": {"HANDS"}, "2:0": {"BODY"}},
            {"1": "train", "2": "train"},
            {"HANDS"},
            role="train",
        )

        self.assertEqual(pieces, {"HANDS": {10, 11}})

    def test_catalog_scope_preserves_full_candidate_local_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "excluded.png"
            second = Path(directory) / "included.png"
            Image.new("RGB", (4, 4), "red").save(first)
            Image.new("RGB", (4, 4), "blue").save(second)
            metadata = [
                {
                    "slot": "HANDS",
                    "local_path": str(first),
                    "pieces": [{"piece_id": 20}],
                },
                {
                    "slot": "HANDS",
                    "local_path": str(second),
                    "pieces": [{"piece_id": 10}],
                },
            ]
            samples = build_catalog_samples(
                metadata,
                {"HANDS": {10: (1,), 20: (0,)}},
                allowed_piece_ids={"HANDS": {10}},
            )

        self.assertEqual(len(samples["HANDS"]), 1)
        self.assertEqual(samples["HANDS"][0]["local_index"], 1)
        self.assertEqual(samples["HANDS"][0]["positives"], (1,))

    def test_catalog_scope_rejects_mixed_train_validation_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.png"
            Image.new("RGB", (4, 4), "red").save(path)
            metadata = [
                {
                    "slot": "HANDS",
                    "local_path": str(path),
                    "pieces": [{"piece_id": 10}, {"piece_id": 20}],
                }
            ]

            with self.assertRaisesRegex(ValueError, "validation pieces"):
                build_catalog_samples(
                    metadata,
                    {"HANDS": {10: (0,), 20: (0,)}},
                    allowed_piece_ids={"HANDS": {10}},
                    forbidden_piece_ids={"HANDS": {20}},
                )

    def test_split_is_stable_and_disjoint(self) -> None:
        samples = {
            slot: []
            for slot in ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
        }
        samples["HANDS"] = [
            {
                "path": f"render-{index}.png",
                "local_index": index,
                "positives": (index,),
            }
            for index in range(100)
        ]

        first = split_catalog_samples(samples, validation_percent=20)
        second = split_catalog_samples(samples, validation_percent=20)

        self.assertEqual(first, second)
        train_paths = {row["path"] for row in first[0]["HANDS"]}
        validation_paths = {row["path"] for row in first[1]["HANDS"]}
        self.assertFalse(train_paths.intersection(validation_paths))
        self.assertEqual(len(train_paths | validation_paths), 100)

    def test_synthetic_batch_honors_candidate_crop_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            image = Image.new("RGB", (20, 10), "red")
            ImageDraw.Draw(image).rectangle((10, 0, 19, 9), fill="blue")
            image.save(path)
            batch = load_synthetic_batch(
                [
                    {
                        "path": str(path),
                        "candidate_crop_box_normalized": [0.5, 0.0, 1.0, 1.0],
                    }
                ],
                augment=False,
            )
            self.assertEqual(tuple(batch.shape), (1, 3, 224, 224))
            self.assertGreater(
                float(batch[0, 2].mean()), float(batch[0, 0].mean())
            )


if __name__ == "__main__":
    unittest.main()
