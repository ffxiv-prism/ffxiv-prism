from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ffxiv_prism.matching.item_dyes import load_item_dye_counts, enrich_report_dye_counts
from ffxiv_prism.matching.inference import write_html_report


class ReportItemDyesTests(unittest.TestCase):
    def test_csv_keeps_zero_and_rejects_ambiguous_or_incomplete_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.csv"
            path.write_text(
                "#,Name,EquipSlotCategory,DyeCount\n"
                "1,Fixed,3,0\n2,Dual,3,2\n3,Ambiguous,3,0\n"
                "4,Ambiguous,3,1\n5,Dual,0,0\n6,Incomplete,3,1\n"
                "7,Incomplete,3,\n8,Invalid,3,-1\n",
                encoding="utf-8-sig",
            )
            by_id, by_name = load_item_dye_counts(path)
            self.assertEqual(by_id[1], 0)
            self.assertEqual(by_id[2], 2)
            self.assertEqual(by_name, {"Fixed": 0, "Dual": 2})
            self.assertNotIn(5, by_id)
            self.assertNotIn(8, by_id)

    def test_jsonl_keeps_ambiguous_and_incomplete_names_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            rows = [{"game_item_id": item_id, "name_en": name,
                     "equip_slot_category_id": slot, "dye_count": count}
                    for item_id, name, slot, count in [
                        (1, "Fixed", 3, 0), (2, "Dual", 3, 2),
                        (3, "Ambiguous", 3, 0), (4, "Ambiguous", 3, 1),
                        (5, "Dual", 0, 0), (6, "Incomplete", 3, 1),
                        (7, "Incomplete", 3, None), (8, "Invalid", 3, -1)]]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            by_id, by_name = load_item_dye_counts(path)
            self.assertEqual(by_name, {"Fixed": 0, "Dual": 2})
            self.assertNotIn(5, by_id)
            self.assertNotIn(8, by_id)

    def test_current_table_overrides_stale_csv_even_when_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "xiv_datamining_item_en.csv").write_text(
                "#,Name,EquipSlotCategory,DyeCount\n1,Dual,3,1\n2,Unknown,3,2\n",
                encoding="utf-8",
            )
            current = root / "cafemaker_item_names_raw.jsonl"
            current.write_text("\n".join(json.dumps(row) for row in [
                {"game_item_id": 1, "name_en": "Dual", "equip_slot_category_id": 3, "dye_count": 2},
                {"game_item_id": 2, "name_en": "Unknown", "equip_slot_category_id": 3, "dye_count": None},
            ]), encoding="utf-8")
            with patch("ffxiv_prism.matching.item_dyes.GAME_DATA_RAW_DIR", root):
                self.assertEqual(load_item_dye_counts(), ({1: 2}, {"Dual": 2}))
                current.write_text('{"game_item_id": 1}\n', encoding="utf-8")
                self.assertEqual(load_item_dye_counts()[0], {1: 1, 2: 2})
                current.unlink()
                self.assertEqual(load_item_dye_counts()[0], {1: 1, 2: 2})

    def test_missing_source_produces_unknown_not_undyeable(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_item_dye_counts(Path(directory) / "missing.csv"), ({}, {}))

    def test_item_id_takes_priority_and_same_models_are_resolved_individually(self):
        payload = {"slots": [{"candidates": [
            {"game_item_id": 1, "name_en": "Dual", "score": 0.8,
             "appearance_alternatives": [{"name_en": "Dual"}]},
            {"game_item_id": 999, "name_en": "Dual"},
            {"name_en": "Missing"},
        ]}]}
        original = copy.deepcopy(payload)
        with patch("ffxiv_prism.matching.item_dyes.load_item_dye_counts", return_value=({1: 0}, {"Dual": 2})):
            rows = enrich_report_dye_counts(payload)["slots"][0]["candidates"]
        self.assertEqual(rows[0]["dye_count"], 0)
        self.assertEqual(rows[0]["appearance_alternatives"][0]["dye_count"], 2)
        self.assertIsNone(rows[1]["dye_count"])
        self.assertIsNone(rows[2]["dye_count"])
        self.assertEqual(payload, original)

    def test_report_distinguishes_dyeable_undyeable_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "ffxiv_prism.matching.item_dyes.load_item_dye_counts",
            return_value=({}, {"Dyeable": 2, "Fixed": 0}),
        ), patch("ffxiv_prism.matching.item_names.load_chinese_item_names", return_value={}):
            report = Path(directory) / "report.html"
            write_html_report(report, {"slots": [{"slot": "HEAD", "candidates": [
                {"name_en": "Dyeable", "appearance_alternatives": [{"name_en": "Fixed"}]},
                {"name_en": "Fixed"}, {"name_en": "Unknown <item>"},
            ]}]})
            content = report.read_text(encoding="utf-8")
            self.assertIn('class="equipment-name dyeable">Dyeable', content)
            self.assertEqual(content.count('class="equipment-name undyeable">Fixed'), 2)
            self.assertIn('class="equipment-name dye-unknown">Unknown &lt;item&gt;', content)
            self.assertIn('（染色信息未知）', content)
            self.assertIn('class="dye-legend muted"', content)


if __name__ == "__main__":
    unittest.main()
