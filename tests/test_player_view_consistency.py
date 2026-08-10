from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_player_view_consistency import (
    aggregate_embeddings,
    evaluate_consistency,
)


class PlayerViewConsistencyTests(unittest.TestCase):
    def test_aggregate_embeddings_filters_crop_profile(self) -> None:
        examples = [
            {
                "example_id": "10:0",
                "glamour_id": 10,
                "local_path": "data/images/10/view.png",
                "core_slot_labels": {
                    "HANDS": {
                        "item_name": "Test Gloves",
                        "piece_ids": [7],
                    }
                },
            }
        ]
        paths = [
            "cache/10-view/crops/10-view-hands_left.jpg",
            "cache/10-view/crops/10-view-hands_forearm_left.jpg",
        ]
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        rows = aggregate_embeddings(
            examples,
            paths,
            embeddings,
            slot="HANDS",
            allowed_suffixes={"hands_left"},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["crop_count"], 1)
        np.testing.assert_allclose(rows[0]["embedding"], [1.0, 0.0])

    def test_consistency_ranks_same_glamour_and_cross_glamour_appearance(self) -> None:
        rows = [
            {
                "example_id": "10:0",
                "glamour_id": "10",
                "expected_name": "Shared Gloves",
                "appearance_key": "shared gloves",
                "embedding": np.asarray([1.0, 0.0]),
            },
            {
                "example_id": "10:1",
                "glamour_id": "10",
                "expected_name": "Shared Gloves",
                "appearance_key": "shared gloves",
                "embedding": np.asarray([0.99, 0.01]),
            },
            {
                "example_id": "20:0",
                "glamour_id": "20",
                "expected_name": "Shared Gloves",
                "appearance_key": "shared gloves",
                "embedding": np.asarray([0.8, 0.2]),
            },
            {
                "example_id": "30:0",
                "glamour_id": "30",
                "expected_name": "Other Gloves",
                "appearance_key": "other gloves",
                "embedding": np.asarray([0.0, 1.0]),
            },
        ]

        result = evaluate_consistency(rows, cutoffs=(1, 3))

        self.assertEqual(result["same_glamour"]["query_count"], 2)
        self.assertEqual(result["same_glamour"]["hits"]["1"], 2)
        self.assertEqual(
            result["same_appearance_other_glamour"]["query_count"],
            3,
        )
        self.assertEqual(
            result["same_appearance_other_glamour"]["hits"]["3"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
