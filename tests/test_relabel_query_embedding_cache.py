from __future__ import annotations

import unittest

import numpy as np

from scripts.relabel_query_embedding_cache import reference_order


class RelabelQueryEmbeddingCacheTests(unittest.TestCase):
    def test_maps_equivalent_crop_suffixes_into_reference_order(self) -> None:
        source = [
            r"D:\masked\a\crops\a-hands_left.jpg",
            r"D:\masked\b\crops\b-hands_right.jpg",
        ]
        reference = [
            r"D:\raw\b\crops\b-hands_right.jpg",
            r"D:\raw\a\crops\a-hands_left.jpg",
        ]

        order = reference_order(source, reference)

        np.testing.assert_array_equal(order, np.asarray((1, 0)))

    def test_rejects_missing_crop_suffix(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing 1"):
            reference_order(
                [r"D:\masked\a\crops\a-hands_left.jpg"],
                [r"D:\raw\b\crops\b-hands_left.jpg"],
            )


if __name__ == "__main__":
    unittest.main()
