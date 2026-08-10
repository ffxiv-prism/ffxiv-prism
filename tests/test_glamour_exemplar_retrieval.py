from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_glamour_exemplar_retrieval import search_exemplars


class GlamourExemplarRetrievalTests(unittest.TestCase):
    def test_deduplicates_multiple_crops_of_same_appearance(self) -> None:
        exemplars = [
            exemplar("a1", "coat", [1.0, 0.0], 10),
            exemplar("a2", "coat", [0.9, 0.1], 10),
            exemplar("b1", "robe", [0.0, 1.0], 20),
        ]
        records = {"query": [{"path": "query.jpg", "slot": "BODY"}]}

        results = search_exemplars(
            exemplars,
            records,
            {"query.jpg": np.asarray([1.0, 0.0], dtype=np.float32)},
            candidate_k=10,
        )

        self.assertEqual(
            [row["name_en"] for row in results["query.jpg"]],
            ["coat", "robe"],
        )
        self.assertEqual(results["query.jpg"][0]["source_example_id"], "a1")
        self.assertAlmostEqual(
            results["query.jpg"][0]["neighbor_vote_fraction_5"],
            2.0 / 3.0,
        )


def exemplar(
    example_id: str,
    name: str,
    embedding: list[float],
    piece_id: int,
) -> dict:
    return {
        "example_id": example_id,
        "glamour_id": example_id,
        "path": f"{example_id}.jpg",
        "embedding": np.asarray(embedding, dtype=np.float32),
        "appearance_key": name,
        "name_en": name,
        "piece_ids": [piece_id],
        "model_race": "Hyur",
        "model_gender": "female",
    }


if __name__ == "__main__":
    unittest.main()
