from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_glamour_embedding_model import search_crops
from scripts.evaluate_retrieval_funnel import (
    combine_stage_cases,
    rank_body_cases,
    summarize_funnel_cases,
    validate_cutoffs,
)


class RetrievalFunnelTest(unittest.TestCase):
    def test_validate_cutoffs_normalizes_and_checks_candidate_limit(self) -> None:
        self.assertEqual(validate_cutoffs([10, 1, 10], 10), (1, 10))
        with self.assertRaises(ValueError):
            validate_cutoffs([0, 1], 10)
        with self.assertRaises(ValueError):
            validate_cutoffs([1, 11], 10)

    def test_search_crops_can_route_by_crop_and_gender(self) -> None:
        metadata = [
            {
                "slot": "HANDS",
                "candidate_crop": "left_arm_tight",
                "model_gender": gender,
                "pieces": [{"piece_id": piece_id, "name_en": gender}],
                "image_row_id": piece_id,
                "local_path": f"{gender}.png",
                "source_image_url": f"https://example.test/{gender}.png",
            }
            for piece_id, gender in ((1, "female"), (2, "male"))
        ]
        embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        records = {
            "example-1": [
                {
                    "example_id": "example-1",
                    "slot": "HANDS",
                    "name": "hands_left",
                    "path": "left.png",
                }
            ]
        }

        results = search_crops(
            metadata,
            embeddings,
            records,
            {"left.png": np.asarray([1.0, 0.0], dtype=np.float32)},
            10,
            candidate_crop_routes={"hands_left": ("left_arm_tight",)},
            candidate_gender_by_example={"example-1": "male"},
        )

        self.assertEqual(
            [row["piece_id"] for row in results["left.png"]],
            [2],
        )

    def test_search_crops_preserves_piece_order_when_rows_share_pieces(self) -> None:
        metadata = []
        for row_id, piece_ids in enumerate(((1, 2), (2, 3), (4,))):
            metadata.append(
                {
                    "slot": "FEET",
                    "pieces": [
                        {"piece_id": piece_id, "name_en": str(piece_id)}
                        for piece_id in piece_ids
                    ],
                    "image_row_id": row_id,
                    "local_path": f"{row_id}.png",
                    "source_image_url": f"https://example.test/{row_id}.png",
                }
            )
        results = search_crops(
            metadata,
            np.asarray([[1.0, 0.0], [0.8, 0.2], [0.5, 0.5]], dtype=np.float32),
            {
                "example": [
                    {
                        "example_id": "example",
                        "slot": "FEET",
                        "name": "feet",
                        "path": "feet.png",
                    }
                ]
            },
            {"feet.png": np.asarray([1.0, 0.0], dtype=np.float32)},
            4,
        )
        self.assertEqual(
            [row["piece_id"] for row in results["feet.png"]],
            [1, 2, 3, 4],
        )

    def test_rank_body_cases_accepts_same_appearance_name(self) -> None:
        examples = [
            {
                "example_id": "example-1",
                "glamour_id": 1,
                "model_gender": "female",
                "model_race": "hyur",
                "core_slot_labels": {
                    "BODY": {
                        "item_name": "Shared Coat",
                        "piece_ids": [99],
                        "piece_match_type": "exact_item",
                    }
                },
            }
        ]
        records = {
            "example-1": [
                {
                    "path": "crop.png",
                    "slot": "BODY",
                    "visibility": "visible",
                    "confidence": 1.0,
                    "role": "primary",
                    "tags": (),
                }
            ]
        }
        search = {
            "crop.png": [
                {"piece_id": 1, "name_en": "Wrong Coat", "score": 0.9},
                {"piece_id": 2, "name_en": "Shared Coat", "score": 0.8},
            ]
        }
        metadata = [
            {
                "slot": "BODY",
                "pieces": [
                    {"piece_id": 1, "name_en": "Wrong Coat"},
                    {"piece_id": 2, "name_en": "Shared Coat"},
                ],
            }
        ]

        cases = rank_body_cases(examples, records, search, metadata)

        self.assertEqual(cases[0]["status"], "eligible")
        self.assertEqual(cases[0]["rank"], 2)

    def test_rrf_rewards_consensus_across_crops(self) -> None:
        examples = [
            {
                "example_id": "example-1",
                "glamour_id": 1,
                "core_slot_labels": {
                    "HANDS": {
                        "item_name": "Consensus Gloves",
                        "piece_ids": [2],
                        "piece_match_type": "exact_item",
                    }
                },
            }
        ]
        records = {
            "example-1": [
                {
                    "path": path,
                    "slot": "HANDS",
                    "visibility": "visible",
                    "confidence": 1.0,
                    "role": "primary",
                    "tags": (),
                }
                for path in ("left.png", "right.png")
            ]
        }
        search = {
            "left.png": [
                {"piece_id": 1, "name_en": "Distractor", "score": 0.99},
                {
                    "piece_id": 2,
                    "name_en": "Consensus Gloves",
                    "score": 0.80,
                },
            ],
            "right.png": [
                {"piece_id": 3, "name_en": "Other", "score": 0.98},
                {
                    "piece_id": 2,
                    "name_en": "Consensus Gloves",
                    "score": 0.79,
                },
            ],
        }
        metadata = [
            {
                "slot": "HANDS",
                "pieces": [
                    {"piece_id": 1, "name_en": "Distractor"},
                    {"piece_id": 2, "name_en": "Consensus Gloves"},
                    {"piece_id": 3, "name_en": "Other"},
                ],
            }
        ]

        cases = rank_body_cases(
            examples,
            records,
            search,
            metadata,
            slot="HANDS",
            fusion="rrf",
            rrf_k=1,
        )

        self.assertEqual(cases[0]["rank"], 1)

    def test_summarize_funnel_reports_coverage_and_conversion(self) -> None:
        base = [
            {
                "example_id": "a",
                "glamour_id": 1,
                "status": "eligible",
                "rank": 10,
            },
            {
                "example_id": "b",
                "glamour_id": 2,
                "status": "eligible",
                "rank": None,
            },
            {
                "example_id": "c",
                "glamour_id": 3,
                "status": "truth_not_in_index",
                "rank": None,
            },
        ]
        query = [
            {**base[0], "rank": 2},
            {**base[1], "rank": 100},
            base[2],
        ]
        reranker = [
            {**base[0], "rank": 1},
            {**base[1], "rank": 5},
            base[2],
        ]
        cases = combine_stage_cases(base, query, reranker)

        summary = summarize_funnel_cases(cases, (1, 10, 100))

        self.assertEqual(summary["index_coverage"]["hits"], 2)
        self.assertEqual(summary["index_coverage"]["denominator"], 3)
        self.assertEqual(summary["fold_query"]["hits"]["100"], 2)
        self.assertEqual(summary["reranker"]["hits"]["1"], 1)
        self.assertEqual(
            summary["reranker_top1_conversion_from_query_pool"]["rate"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
