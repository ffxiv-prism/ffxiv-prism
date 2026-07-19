from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.train_synthetic_query_encoder import (
    load_synthetic_batch,
    split_catalog_samples,
)


class SyntheticQueryEncoderTests(unittest.TestCase):
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
