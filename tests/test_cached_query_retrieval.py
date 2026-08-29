from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_cached_query_retrieval import build_quality_by_id
from scripts.evaluate_query_only_retrieval import load_query_embedding_cache


class CachedQueryRetrievalTests(unittest.TestCase):
    def test_build_quality_by_id_supports_per_slot_eligibility(self) -> None:
        examples = [{"example_id": "a"}, {"example_id": "b"}]
        quality = build_quality_by_id(
            examples,
            slot="HANDS",
            eligibility={"a": {"HANDS"}, "b": {"BODY"}},
        )
        self.assertEqual(quality, {"a": True, "b": False})

    def test_build_quality_by_id_requires_exactly_one_source(self) -> None:
        with self.assertRaises(ValueError):
            build_quality_by_id([], slot="HANDS")
        with self.assertRaises(ValueError):
            build_quality_by_id(
                [],
                slot="HANDS",
                quality_rows=[],
                eligibility={},
            )

    def test_adapted_query_cache_is_reused_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "query.npz"
            np.savez_compressed(
                cache,
                paths=np.asarray(["crop.png"]),
                embeddings=np.asarray([[0.5, 0.5]], dtype=np.float32),
                checkpoint=np.asarray("checkpoint.pt"),
                model_id=np.asarray("model"),
                slot=np.asarray("BODY"),
                fold=np.asarray(1),
                fold_role=np.asarray("validation"),
                pose_normalize_hands=np.asarray(0),
            )

            loaded = load_query_embedding_cache(
                cache,
                {"example": [{"path": "crop.png"}]},
                checkpoint=Path("checkpoint.pt"),
                model_id="model",
                slot="BODY",
                fold=1,
                fold_role="validation",
                pose_normalize_hands=False,
            )

            self.assertIsNotNone(loaded)
            self.assertTrue(np.array_equal(loaded["crop.png"], [0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
