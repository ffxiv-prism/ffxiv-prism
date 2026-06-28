from __future__ import annotations

import unittest

from scripts.evaluate_glamour_fusion import parse_weights
from scripts.evaluate_glamour_embedding_model import (
    load_or_build_candidate_embeddings,
    parse_split_roles,
    pool_dino_tokens,
)
import torch


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

    def test_dino_mean_pooling_excludes_cls_token(self) -> None:
        tokens = torch.tensor([[[10.0, 10.0], [1.0, 3.0], [3.0, 5.0]]])

        pooled = pool_dino_tokens(tokens, "mean")

        self.assertTrue(torch.equal(pooled, torch.tensor([[2.0, 4.0]])))

    def test_split_role_parser_rejects_unknown_role(self) -> None:
        self.assertEqual(
            parse_split_roles("validation,development_test"),
            {"validation", "development_test"},
        )
        with self.assertRaises(ValueError):
            parse_split_roles("test")
