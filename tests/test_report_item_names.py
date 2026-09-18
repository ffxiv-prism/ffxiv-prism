from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote

from ffxiv_prism.matching.inference import write_html_report
from ffxiv_prism.matching.item_names import (
    build_chinese_item_names,
    enrich_report_item_names,
    load_chinese_item_names,
)


class ReportItemNamesTests(unittest.TestCase):
    def test_stale_index_names_are_filled_without_changing_retrieval_results(self):
        payload = {"slots": [{"slot": "FEET", "candidates": [{
            "piece_id": 35, "rank": 1, "score": 0.75,
            "name_en": "Vana'dielian Shoes of Healing", "name_zh": None,
            "appearance_alternatives": [{
                "piece_id": 905, "name_en": "Critical Hit C-2 Tour Shoes",
                "name_zh": "", "original_rank": 3,
            }],
        }]}]}
        original = copy.deepcopy(payload)
        result = enrich_report_item_names(payload, {
            "Vana'dielian Shoes of Healing": "瓦纳·迪尔治愈皮鞋",
            "Critical Hit C-2 Tour Shoes": "暴击C2便鞋",
        })
        self.assertEqual(payload, original)
        candidate = result["slots"][0]["candidates"][0]
        self.assertEqual(candidate.pop("name_zh"), "瓦纳·迪尔治愈皮鞋")
        self.assertEqual(candidate["appearance_alternatives"][0].pop("name_zh"), "暴击C2便鞋")
        del original["slots"][0]["candidates"][0]["name_zh"]
        del original["slots"][0]["candidates"][0]["appearance_alternatives"][0]["name_zh"]
        self.assertEqual(result, original)

    def test_ambiguous_unknown_and_existing_names_are_not_overwritten(self):
        names = build_chinese_item_names([
            {"name_en": "Ambiguous", "name_zh": "甲"},
            {"name_en": "Ambiguous", "name_zh": "乙"},
            {"name_en": "Known", "name_zh": "丙"},
            {"name_en": "Known", "name_zh": "丙"},
        ])
        self.assertEqual(names, {"Known": "丙"})
        payload = {"slots": [{"candidates": [
            {"name_en": "Ambiguous", "name_zh": None},
            {"name_en": "Unknown", "name_zh": None},
            {"name_en": "Known", "name_zh": "已有译名"},
        ]}]}
        self.assertEqual(enrich_report_item_names(payload, names), payload)

    def test_catalogue_updates_are_visible_without_restarting(self):
        with patch("ffxiv_prism.matching.item_names.read_jsonl") as read:
            read.side_effect = [[], [], [{"name_en": "New", "name_zh": "新增"}], []]
            self.assertEqual(load_chinese_item_names(), {})
            self.assertEqual(load_chinese_item_names(), {"New": "新增"})

    def test_outfit_bundle_does_not_make_equipment_name_ambiguous(self):
        names = build_chinese_item_names([
            {"name_en": "Southern Seas Swimsuit", "name_zh": "南海女式连衣裙",
             "equip_slot_category_id": 4},
            {"name_en": "Southern Seas Swimsuit", "name_zh": "南海女式连衣裙组合",
             "equip_slot_category_id": 0},
        ])
        self.assertEqual(names, {"Southern Seas Swimsuit": "南海女式连衣裙"})

    def test_old_report_gets_chinese_name_and_wiki_link_when_rendered(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "ffxiv_prism.matching.item_names.load_chinese_item_names",
            return_value={"Critical Hit C-2 Tour Shoes": "暴击C2便鞋"},
        ):
            path = Path(directory) / "report.html"
            write_html_report(path, {"slots": [{"slot": "FEET", "candidates": [{
                "name_en": "Critical Hit C-2 Tour Shoes", "name_zh": None,
                "piece_id": 905, "rank": 1,
            }]}]})
            content = path.read_text(encoding="utf-8")
            self.assertIn("暴击C2便鞋<br>", content)
            self.assertIn("https://ff14.huijiwiki.com/wiki/物品:暴击C2便鞋", unquote(content))
            self.assertIn('target="_blank"', content)


if __name__ == "__main__":
    unittest.main()
