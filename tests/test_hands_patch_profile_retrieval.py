from __future__ import annotations

import unittest

import numpy as np

from ffxiv_prism.matching.patch_profiles import PatchProfileIndex
from scripts.evaluate_hands_patch_profile_retrieval import (
    rank_piece_ids,
    select_query_records,
)


class HandsPatchProfileRetrievalTests(unittest.TestCase):
    def test_selects_visible_named_query_crop(self) -> None:
        records = {
            "1:0": [
                {
                    "name": "hands_forearm_left",
                    "visibility": "visible",
                    "path": "left.jpg",
                },
                {
                    "name": "hands_forearm_right",
                    "visibility": "visible",
                    "path": "right.jpg",
                },
            ]
        }

        selected = select_query_records(records, query_crop="hands_forearm_left")

        self.assertEqual(selected["1:0"]["path"], "left.jpg")

    def test_ranks_piece_by_best_profile_score(self) -> None:
        index = PatchProfileIndex(
            profiles=np.zeros((2, 1, 1), dtype=np.float16),
            projection=np.zeros((1, 1), dtype=np.float32),
            metadata_indices=np.asarray((0, 1)),
            piece_ids_by_profile=((10,), (20,)),
            model_genders=("female", "female"),
            model_id="test",
            preprocess="color",
            candidate_crop="left_arm_tight",
            grid_size=1,
        )
        metadata = [
            {"pieces": [{"piece_id": 10, "name_en": "A"}]},
            {"pieces": [{"piece_id": 20, "name_en": "B"}]},
        ]

        ranked, names = rank_piece_ids(
            np.asarray((0, 1)),
            np.asarray((0.2, 0.8)),
            index,
            metadata,
        )

        self.assertEqual(ranked, [20, 10])
        self.assertEqual(names[20], "B")


if __name__ == "__main__":
    unittest.main()
