from __future__ import annotations

import unittest

from scripts.close_glamour_labels_with_official_equivalence import (
    close_piece_ids,
)


class OfficialLabelClosureTests(unittest.TestCase):
    def test_closes_transitively_from_variant_fallback_piece(self) -> None:
        graph = {
            1: {2},
            2: {1, 3},
            3: {2},
        }
        self.assertEqual(close_piece_ids({1}, graph), {1, 2, 3})

    def test_preserves_unmapped_singleton(self) -> None:
        self.assertEqual(close_piece_ids({9}, {}), {9})


if __name__ == "__main__":
    unittest.main()
