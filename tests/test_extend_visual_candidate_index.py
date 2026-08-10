from __future__ import annotations

import unittest

from scripts.extend_visual_candidate_index import (
    find_new_rows,
    validate_additions,
)


class ExtendVisualCandidateIndexTests(unittest.TestCase):
    def test_find_new_rows_preserves_current_row_payload(self) -> None:
        source = [
            {
                "source_image_url": "https://example.test/old.png",
                "pieces": [{"gearset_slug": "old"}],
            }
        ]
        current = [
            {
                "source_image_url": "https://example.test/old.png",
                "pieces": [{"gearset_slug": "old", "piece_id": 1}],
            },
            {
                "source_image_url": "https://example.test/new.png",
                "slot": "FEET",
                "pieces": [{"gearset_slug": "neo-queens", "piece_id": 2}],
            },
        ]

        additions = find_new_rows(source, current)

        self.assertEqual(len(additions), 1)
        self.assertEqual(additions[0]["slot"], "FEET")
        self.assertEqual(additions[0]["pieces"][0]["piece_id"], 2)

    def test_validate_additions_rejects_unexpected_gearset(self) -> None:
        rows = [
            {
                "pieces": [{"gearset_slug": "not-neo-queens"}],
            }
        ]

        with self.assertRaises(ValueError):
            validate_additions(
                rows,
                expected_count=1,
                expected_gearset_slug="neo-queens",
            )


if __name__ == "__main__":
    unittest.main()
