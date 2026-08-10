from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_feet_visibility_triage import (
    has_feet_label,
    load_thresholds,
    triage_sort_key,
)


class BuildFeetVisibilityTriageTests(unittest.TestCase):
    def test_has_feet_label_requires_piece_ids(self) -> None:
        self.assertTrue(
            has_feet_label(
                {"core_slot_labels": {"FEET": {"piece_ids": [10]}}}
            )
        )
        self.assertFalse(
            has_feet_label(
                {"core_slot_labels": {"FEET": {"piece_ids": []}}}
            )
        )

    def test_load_thresholds_prefers_high_recall_triage(self) -> None:
        thresholds = {
            "min_lower_bottom_ratio": 0.9,
            "min_bottom_band_share": 0.5,
            "max_shoe_share": 0.05,
            "max_edge_gap_ratio": 0.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostic.json"
            path.write_text(
                json.dumps(
                    {"high_recall_triage": {"thresholds": thresholds}}
                ),
                encoding="utf-8",
            )

            loaded, source = load_thresholds(path)

        self.assertEqual(loaded, thresholds)
        self.assertEqual(source, "high_recall_triage.thresholds")

    def test_triage_sort_prioritizes_lower_shoe_share(self) -> None:
        hidden = {
            "example_id": "hidden",
            "features": {
                "shoe_share_person": 0.0,
                "lower_garment_bottom_band_share": 1.0,
                "lower_garment_bottom_ratio": 1.0,
                "person_bottom_edge_gap_ratio": 0.0,
            },
        }
        visible = {
            "example_id": "visible",
            "features": {
                "shoe_share_person": 0.04,
                "lower_garment_bottom_band_share": 1.0,
                "lower_garment_bottom_ratio": 1.0,
                "person_bottom_edge_gap_ratio": 0.0,
            },
        }

        self.assertLess(triage_sort_key(hidden), triage_sort_key(visible))


if __name__ == "__main__":
    unittest.main()
