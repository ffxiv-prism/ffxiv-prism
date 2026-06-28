from __future__ import annotations

import unittest

from scripts.train_synthetic_query_encoder import (
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


if __name__ == "__main__":
    unittest.main()
