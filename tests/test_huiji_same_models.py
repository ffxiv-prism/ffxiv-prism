from __future__ import annotations

import unittest

from ffxiv_prism.huiji_model_equivalence import (
    build_piece_equivalence_from_models,
)
from ffxiv_prism.huiji_same_models import (
    build_piece_equivalence,
    parse_same_model_page,
)


class HuijiPageParserTests(unittest.TestCase):
    def test_parses_exact_and_main_model_groups(self) -> None:
        html = """
        <html><body>
          <h1 id="firstHeading">物品:深金御敌胸甲</h1>
          <h2><span id="同模型装备">同模型装备</span></h2>
          <h3>模型完全相同物品</h3>
          <ul>
            <li><a href="/wiki/物品:深金御敌胸甲">深金御敌胸甲</a></li>
            <li><a href="/wiki/物品:深金制敌胸甲">深金制敌胸甲</a></li>
          </ul>
          <h3>主模型相同物品</h3>
          <table>
            <tr><td><a href="/wiki/物品:战女神判官胸甲">战女神判官胸甲</a></td></tr>
          </table>
          <h2>其他信息</h2>
          <a href="/wiki/物品:不应收录">不应收录</a>
        </body></html>
        """
        parsed = parse_same_model_page(
            html,
            "https://ff14.huijiwiki.com/wiki/物品:深金御敌胸甲",
        )
        self.assertTrue(parsed.section_found)
        self.assertEqual(parsed.page_title, "深金御敌胸甲")
        self.assertEqual(
            [link.name_zh for link in parsed.exact_same_model],
            ["深金制敌胸甲"],
        )
        self.assertEqual(
            [link.name_zh for link in parsed.main_same_model],
            ["战女神判官胸甲"],
        )

    def test_builds_page_relation_closure_and_preserves_two_levels(self) -> None:
        pieces = [
            {
                "piece_id": 1,
                "slot": "BODY",
                "name_en": "Deepgold Cuirass of Fending",
                "name_zh": "深金御敌胸甲",
            },
            {
                "piece_id": 2,
                "slot": "BODY",
                "name_en": "Deepgold Cuirass of Striking",
                "name_zh": "深金制敌胸甲",
            },
            {
                "piece_id": 3,
                "slot": "BODY",
                "name_en": "Halonic Inquisitor's Cuirass",
                "name_zh": "战女神判官胸甲",
            },
        ]
        records = [
            {
                "status": "matched",
                "seed_name_en": "Deepgold Cuirass of Fending",
                "seed_name_zh": "深金御敌胸甲",
                "exact_same_model": [
                    {
                        "name_zh": "深金制敌胸甲",
                        "item_url": "https://ff14.huijiwiki.com/wiki/物品:深金制敌胸甲",
                    }
                ],
                "main_same_model": [
                    {
                        "name_zh": "战女神判官胸甲",
                        "item_url": "https://ff14.huijiwiki.com/wiki/物品:战女神判官胸甲",
                    }
                ],
            }
        ]
        payload = build_piece_equivalence(pieces, records)
        self.assertFalse(payload["inference_used"])
        self.assertEqual(payload["exact_model_equivalents"]["1"], [2])
        self.assertEqual(payload["direct_equivalents"]["1"], [2, 3])
        self.assertEqual(payload["direct_equivalents"]["3"], [1, 2])


class HuijiModelRuleTests(unittest.TestCase):
    def test_reproduces_armour_main_and_exact_model_groups(self) -> None:
        pieces = [
            {
                "piece_id": 1,
                "slot": "BODY",
                "name_en": "Deepgold Cuirass of Fending",
                "game_item_id": 25647,
                "game_item_ids_json": "[]",
            },
            {
                "piece_id": 2,
                "slot": "BODY",
                "name_en": "Halonic Inquisitor's Cuirass",
                "game_item_id": 11273,
                "game_item_ids_json": "[]",
            },
            {
                "piece_id": 3,
                "slot": "BODY",
                "name_en": "Exact Variant",
                "game_item_id": None,
                "game_item_ids_json": "[]",
            },
            {
                "piece_id": 4,
                "slot": "BODY",
                "name_en": "Other Category",
                "game_item_id": 4,
                "game_item_ids_json": "[]",
            },
        ]
        items = [
            {
                "game_item_id": 25647,
                "name_en": "Deepgold Cuirass of Fending",
                "item_ui_category_id": 35,
                "model_main": [173, 3, 0, 0],
            },
            {
                "game_item_id": 11273,
                "name_en": "Halonic Inquisitor's Cuirass",
                "item_ui_category_id": 35,
                "model_main": [173, 1, 0, 0],
            },
            {
                "game_item_id": 3,
                "name_en": "Exact Variant",
                "item_ui_category_id": 35,
                "model_main": [173, 3, 0, 0],
            },
            {
                "game_item_id": 4,
                "name_en": "Other Category",
                "item_ui_category_id": 36,
                "model_main": [173, 3, 0, 0],
            },
        ]
        payload = build_piece_equivalence_from_models(pieces, items)
        self.assertFalse(payload["inference_used"])
        self.assertEqual(payload["mapped_piece_count"], 4)
        self.assertEqual(payload["exact_model_equivalents"]["1"], [3])
        self.assertEqual(payload["direct_equivalents"]["1"], [2, 3])
        self.assertNotIn("4", payload["direct_equivalents"])


if __name__ == "__main__":
    unittest.main()
