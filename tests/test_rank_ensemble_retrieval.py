from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import evaluate_glamour_embedding_model as embedding_model
from scripts.evaluate_rank_ensemble_retrieval import (
    constrain_to_final_candidate_pool,
    fuse_branch_searches,
    load_branch_search_cache,
    save_branch_search_cache,
    select_slot_index_rows,
)


class RankEnsembleRetrievalTests(unittest.TestCase):
    def test_final_pool_preserves_every_primary_candidate(self) -> None:
        pool = [
            {"piece_id": 1, "rank": 1, "score": 1.0},
            {"piece_id": 2, "rank": 2, "score": 0.9},
        ]
        reranked = [
            {"piece_id": 3, "rank": 1, "score": 1.1},
            {"piece_id": 2, "rank": 2, "score": 1.0},
        ]

        result = constrain_to_final_candidate_pool(pool, reranked)

        self.assertEqual([row["piece_id"] for row in result], [2, 1])

    def test_candidate_pool_excludes_new_deep_candidates(self) -> None:
        first = {
            "crop": [
                {"piece_id": 1, "rank": 1, "score": 1.0},
                {"piece_id": 2, "rank": 2, "score": 0.9},
            ]
        }
        second = {
            "crop": [
                {"piece_id": 3, "rank": 1, "score": 1.0},
                {"piece_id": 2, "rank": 2, "score": 0.9},
            ]
        }
        pool = fuse_branch_searches([first], rrf_k=1, candidate_k=2)
        fused = fuse_branch_searches(
            [first, second],
            rrf_k=1,
            candidate_k=2,
            candidate_pool=pool,
        )
        self.assertEqual({row["piece_id"] for row in fused["crop"]}, {1, 2})

    def test_branch_weights_can_reduce_or_increase_one_branch_influence(self) -> None:
        first = {
            "crop": [
                {"piece_id": 1, "rank": 1, "score": 1.0},
                {"piece_id": 2, "rank": 2, "score": 0.9},
            ]
        }
        second = {
            "crop": [
                {"piece_id": 2, "rank": 1, "score": 1.0},
                {"piece_id": 1, "rank": 2, "score": 0.9},
            ]
        }
        fused = fuse_branch_searches(
            [first, second],
            rrf_k=1,
            candidate_k=2,
            branch_weights=(1.0, 2.0),
        )
        self.assertEqual(fused["crop"][0]["piece_id"], 2)

    def test_compact_branch_search_cache_round_trip(self) -> None:
        metadata = [
            {
                "pieces": [{"piece_id": 7, "name_en": "Seven"}],
                "image_row_id": 1,
                "local_path": "seven.png",
                "source_image_url": "https://example.test/seven.png",
            }
        ]
        search = {
            "crop.png": [
                {"piece_id": 7, "rank": 1, "score": 0.8, "name_en": "Seven"}
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "search.npz"
            index_dir = Path("index")
            query_cache = Path("query.npz")
            save_branch_search_cache(
                cache,
                search,
                candidate_k=3,
                index_dir=index_dir,
                query_cache=query_cache,
            )
            loaded = load_branch_search_cache(
                cache,
                metadata,
                candidate_k=3,
                index_dir=index_dir,
                query_cache=query_cache,
            )
        self.assertEqual(loaded["crop.png"][0]["piece_id"], 7)
        self.assertEqual(loaded["crop.png"][0]["rank"], 1)
        self.assertAlmostEqual(loaded["crop.png"][0]["score"], 0.8, places=6)

    def test_auto_routes_local_dinov3_to_dino_encoder(self) -> None:
        with patch("ffxiv_prism.matching.runtime.encoders.DinoV2Encoder") as encoder:
            embedding_model.build_encoder(
                "var/models/dinov3-vitb16-pretrain-lvd1689m",
                "auto",
                "cpu",
                "color",
            )
        encoder.assert_called_once_with(
            "var/models/dinov3-vitb16-pretrain-lvd1689m",
            "dinov3",
            "cpu",
            "color",
            "cls",
        )

    def test_select_slot_index_rows_keeps_embedding_alignment(self) -> None:
        metadata = [
            {"slot": "HEAD", "local_path": "head.png"},
            {"slot": "BODY", "local_path": "body-a.png"},
            {"slot": "BODY", "local_path": "body-b.png"},
        ]
        embeddings = np.asarray(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float32,
        )

        selected_metadata, selected_embeddings = select_slot_index_rows(
            metadata,
            embeddings,
            "BODY",
        )

        self.assertEqual(
            [row["local_path"] for row in selected_metadata],
            ["body-a.png", "body-b.png"],
        )
        self.assertTrue(
            np.array_equal(
                selected_embeddings,
                np.asarray([[2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            )
        )


if __name__ == "__main__":
    unittest.main()
