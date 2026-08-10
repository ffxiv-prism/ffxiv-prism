from __future__ import annotations

import unittest

from scripts.evaluate_retrieval_funnel_official import (
    rank_body_cases_official,
)


class OfficialRetrievalTruthTests(unittest.TestCase):
    def test_uses_only_explicit_piece_ids_not_normalized_names(self) -> None:
        examples = [
            {
                "example_id": "1:0",
                "glamour_id": 1,
                "core_slot_labels": {
                    "BODY": {
                        "item_name": "Augmented Example Coat",
                        "piece_ids": [10, 11],
                        "piece_match_type": "exact_item",
                    }
                },
            }
        ]
        records = {
            "1:0": [
                {
                    "slot": "BODY",
                    "visibility": "visible",
                    "role": "primary",
                    "confidence": 1.0,
                    "tags": [],
                    "path": "crop.png",
                }
            ]
        }
        search = {
            "crop.png": [
                {
                    "piece_id": 12,
                    "name_en": "Example Coat",
                    "score": 1.0,
                },
                {
                    "piece_id": 11,
                    "name_en": "Official Same Model",
                    "score": 0.9,
                },
            ]
        }
        metadata = [
            {
                "slot": "BODY",
                "pieces": [
                    {"piece_id": 10},
                    {"piece_id": 11},
                    {"piece_id": 12},
                ],
            }
        ]
        cases = rank_body_cases_official(
            examples,
            records,
            search,
            metadata,
        )
        self.assertEqual(cases[0]["rank"], 2)
        self.assertEqual(
            cases[0]["eligible_expected_piece_ids"],
            [10, 11],
        )


if __name__ == "__main__":
    unittest.main()
