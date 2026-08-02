from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_hands_extent_routed_retrieval import (
    accumulate_rrf_scores,
    load_extent_map,
    partition_candidate_index,
)


class HandsExtentRoutedRetrievalTests(unittest.TestCase):
    def test_partition_splits_multi_piece_candidate_without_losing_embedding(
        self,
    ) -> None:
        metadata = [
            {
                "slot": "HANDS",
                "model_gender": "female",
                "pieces": [
                    {"piece_id": 1, "name_en": "Short"},
                    {"piece_id": 2, "name_en": "Long"},
                ],
            }
        ]
        embeddings = np.asarray([[1.0, 2.0]], dtype=np.float32)
        rows, arrays, diagnostics = partition_candidate_index(
            metadata,
            embeddings,
            {(1, "female"): "short", (2, "female"): "long"},
        )
        self.assertEqual(rows["short"][0]["pieces"][0]["piece_id"], 1)
        self.assertEqual(rows["long"][0]["pieces"][0]["piece_id"], 2)
        np.testing.assert_array_equal(arrays["short"], embeddings)
        np.testing.assert_array_equal(arrays["long"], embeddings)
        self.assertEqual(diagnostics["missing_piece_gender_count"], 0)

    def test_rrf_accumulates_repeated_crop_evidence(self) -> None:
        records = [
            {
                "path": "a.jpg",
                "name": "hands_left",
                "slot": "HANDS",
                "visibility": "visible",
            },
            {
                "path": "b.jpg",
                "name": "hands_right",
                "slot": "HANDS",
                "visibility": "visible",
            },
        ]
        search = {
            "a.jpg": [{"piece_id": 7, "rank": 1, "name_en": "Seven"}],
            "b.jpg": [{"piece_id": 7, "rank": 2, "name_en": "Seven"}],
        }
        scores, keys = accumulate_rrf_scores(records, search, rrf_k=60)
        self.assertGreater(scores[7], 1 / 100)
        self.assertTrue(keys[7])

    def test_extent_map_rejects_conflicting_piece_gender_rows(self) -> None:
        with self.assertRaises(ValueError):
            load_extent_map(
                [
                    {
                        "piece_id": 1,
                        "model_gender": "female",
                        "extent_class": "short",
                    },
                    {
                        "piece_id": 1,
                        "model_gender": "female",
                        "extent_class": "long",
                    },
                ]
            )


if __name__ == "__main__":
    unittest.main()
