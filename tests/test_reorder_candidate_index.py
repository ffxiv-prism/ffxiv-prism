from __future__ import annotations

import unittest

import numpy as np

from scripts.reorder_candidate_index import reorder_indices


class ReorderCandidateIndexTests(unittest.TestCase):
    def test_reorders_same_candidates_by_path_and_crop(self) -> None:
        source = [
            {"local_path": "a.png", "candidate_crop": "left"},
            {"local_path": "a.png", "candidate_crop": "right"},
            {"local_path": "b.png", "candidate_crop": "left"},
        ]
        reference = [source[2], source[0], source[1]]

        result = reorder_indices(source, reference)

        np.testing.assert_array_equal(result, np.asarray((2, 0, 1)))

    def test_rejects_non_identical_candidate_sets(self) -> None:
        source = [{"local_path": "a.png", "candidate_crop": "left"}]
        reference = [{"local_path": "b.png", "candidate_crop": "left"}]

        with self.assertRaisesRegex(ValueError, "missing 1"):
            reorder_indices(source, reference)

    def test_rejects_duplicate_keys(self) -> None:
        source = [
            {"local_path": "a.png", "candidate_crop": "left"},
            {"local_path": "a.png", "candidate_crop": "left"},
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            reorder_indices(source, source[:1])


if __name__ == "__main__":
    unittest.main()
