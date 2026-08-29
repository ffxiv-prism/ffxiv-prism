from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_glamour_fusion import parse_weights
from scripts.evaluate_glamour_embedding_model import (
    load_crop_embedding_cache,
    load_existing_candidate_embeddings,
    load_or_build_candidate_embeddings,
    parse_split_roles,
    pool_dino_tokens,
)


class GlamourFusionTests(unittest.TestCase):
    def test_parse_weights_accepts_comma_separated_values(self) -> None:
        self.assertEqual(parse_weights("0.25, 0.5,0.75"), [0.25, 0.5, 0.75])

    def test_parse_weights_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_weights("1.1")

    def test_parse_weights_requires_a_value(self) -> None:
        with self.assertRaises(ValueError):
            parse_weights(" , ")

    def test_empty_metadata_cannot_overwrite_an_index(self) -> None:
        with self.assertRaises(ValueError):
            load_or_build_candidate_embeddings(
                encoder=None,
                metadata=[],
                index_dir=None,
                batch_size=1,
            )

    def test_mismatched_metadata_cannot_overwrite_an_existing_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary)
            embeddings_path = index_dir / "embeddings.npy"
            metadata_path = index_dir / "metadata.jsonl"
            np.save(
                embeddings_path,
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                allow_pickle=False,
            )
            original_metadata = {"local_path": "original.png", "slot": "BODY"}
            metadata_path.write_text(
                json.dumps(original_metadata) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                load_or_build_candidate_embeddings(
                    encoder=None,
                    metadata=[{"local_path": "different.png", "slot": "BODY"}],
                    index_dir=index_dir,
                    batch_size=1,
                )

            self.assertTrue(np.array_equal(np.load(embeddings_path), [[1.0, 0.0]]))
            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                json.dumps(original_metadata) + "\n",
            )

    def test_dino_mean_pooling_excludes_cls_token(self) -> None:
        tokens = torch.tensor([[[10.0, 10.0], [1.0, 3.0], [3.0, 5.0]]])

        pooled = pool_dino_tokens(tokens, "mean")

        self.assertTrue(torch.equal(pooled, torch.tensor([[2.0, 4.0]])))

    def test_crop_cache_can_be_validated_without_loading_an_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "query.npz"
            np.savez_compressed(
                cache,
                paths=np.asarray(["crop.png"]),
                embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
                model_id=np.asarray("model"),
                preprocess=np.asarray("color"),
                pooling=np.asarray("cls"),
            )

            loaded = load_crop_embedding_cache(
                {"example": [{"path": "crop.png"}]},
                cache,
                model_id="model",
                preprocess="color",
                pooling="cls",
            )

            self.assertIsNotNone(loaded)
            self.assertTrue(np.array_equal(loaded["crop.png"], [1.0, 2.0]))

    def test_existing_candidate_index_loads_without_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary)
            metadata = [{"local_path": "candidate.png", "slot": "BODY"}]
            (index_dir / "metadata.jsonl").write_text(
                json.dumps(metadata[0]) + "\n",
                encoding="utf-8",
            )
            np.save(
                index_dir / "embeddings.npy",
                np.asarray([[1.0, 0.0]], dtype=np.float32),
            )

            loaded = load_existing_candidate_embeddings(metadata, index_dir)

            self.assertEqual(loaded.shape, (1, 2))
            del loaded
            gc.collect()

    def test_split_role_parser_rejects_unknown_role(self) -> None:
        self.assertEqual(
            parse_split_roles("validation,development_test"),
            {"validation", "development_test"},
        )
        with self.assertRaises(ValueError):
            parse_split_roles("test")
