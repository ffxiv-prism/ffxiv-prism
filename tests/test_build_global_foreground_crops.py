from __future__ import annotations

import unittest

from scripts.build_global_foreground_crops import (
    filter_examples_by_fold,
    filter_examples_by_quality,
    select_regions,
)


class BuildGlobalForegroundCropsTests(unittest.TestCase):
    def test_filters_grouped_glamour_assignments(self) -> None:
        examples = [
            {"example_id": "10:0", "glamour_id": 10},
            {"example_id": "10:1", "glamour_id": 10},
            {"example_id": "20:0", "glamour_id": 20},
        ]
        manifest = {"assignments": {"10": "blind_test", "20": "train"}}

        selected = filter_examples_by_fold(
            examples,
            manifest,
            split="blind_test",
        )

        self.assertEqual([row["example_id"] for row in selected], ["10:0", "10:1"])

    def test_rejects_empty_assignments(self) -> None:
        with self.assertRaisesRegex(ValueError, "no grouped assignments"):
            filter_examples_by_fold([], {}, split="blind_test")

    def test_filters_quality_passes_and_preserves_order(self) -> None:
        examples = [
            {"example_id": "10:0"},
            {"example_id": "10:1"},
            {"example_id": "20:0"},
        ]
        quality = [
            {"example_id": "10:0", "passed": True},
            {"example_id": "10:1", "passed": False},
            {"example_id": "20:0", "passed": True},
        ]

        selected = filter_examples_by_quality(examples, quality)

        self.assertEqual([row["example_id"] for row in selected], ["10:0", "20:0"])

    def test_rejects_missing_quality_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing 1"):
            filter_examples_by_quality([{"example_id": "10:0"}], [])

    def test_selects_only_requested_crop_slots(self) -> None:
        regions = [
            {"slot": "HEAD", "name": "head_tight"},
            {"slot": "HANDS", "name": "hands_left"},
            {"slot": "HANDS", "name": "hands_right"},
        ]

        selected = select_regions(regions, frozenset(("HANDS",)))

        self.assertEqual(
            [row["name"] for row in selected],
            ["hands_left", "hands_right"],
        )


if __name__ == "__main__":
    unittest.main()
