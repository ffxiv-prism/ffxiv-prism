# 功能：验证 Eorzea Collection HTML 解析器能提取装备槽位、图片和性别/种族标签。

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.parser import (
    extract_gearset_links,
    parse_gearset_detail,
)
from ffxiv_prism.collectors.eorzea_collection.image_url import (
    generate_same_gearset_variant_urls,
    parse_gearsets_image_url,
)


class EorzeaParserTests(unittest.TestCase):
    def test_extract_gearset_links(self) -> None:
        html = '<a href="/gearset/test-fending">Test Fending</a><a href="/glamour/123">Ignore</a>'
        self.assertEqual(
            extract_gearset_links(html, "https://ffxiv.eorzeacollection.com/gearsets"),
            ["https://ffxiv.eorzeacollection.com/gearset/test-fending"],
        )

    def test_parse_gearset_detail_equipment(self) -> None:
        html = """
        <html><head><title>Test Set | Eorzea Collection</title></head><body>
        <img src="https://gearsets.eorzeacollection.com/alliance/test/test/test-set/hyur-female-front.png" alt="Female Viera preview">
        <img src="/resources/v7/ec/favicon/favicon-32x32.png" alt="icon">
        <h2>— Equipment —</h2>
        HEAD
        <a href="https://na.finalfantasyxiv.com/lodestone/playguide/db/item/abc/">Test Hat</a>
        <a href="/glamours?piece=test-hat">Glamours using this piece</a>
        BODY
        <a href="https://na.finalfantasyxiv.com/lodestone/playguide/db/item/def/">Test Coat</a>
        <a href="/glamours?piece=test-coat">Glamours using this piece</a>
        <img src="https://gearsets.eorzeacollection.com/alliance/test/test/test-set/hyur-female-body.png" alt="Body thumbnail">
        How to obtain
        </body></html>
        """
        record = parse_gearset_detail(html, "https://ffxiv.eorzeacollection.com/gearset/test-set")
        self.assertEqual(record.slug, "test-set")
        self.assertEqual(record.title, "Test Set")
        self.assertEqual(
            [(piece.slot, piece.name) for piece in record.pieces],
            [
                ("HEAD", "Test Hat"),
                ("BODY", "Test Coat"),
            ],
        )
        self.assertFalse(hasattr(record.pieces[0], "glamours_using_piece_url"))
        self.assertTrue(any(image.model_gender == "female" and image.race == "hyur" for image in record.images))
        self.assertTrue(all("resources" not in image.source_image_url for image in record.images))

    def test_parse_multislot_body_piece_from_equipment_html(self) -> None:
        html = """
        <html><head><title>Gaia's Set | Eorzea Collection</title></head><body>
        <img src="https://gearsets.eorzeacollection.com/mogstation/other/gaias/hyur-female-front.png">
        <img src="https://gearsets.eorzeacollection.com/mogstation/other/gaias/hyur-female-body.png">
        <img src="https://gearsets.eorzeacollection.com/mogstation/other/gaias/hyur-female-hands.png">
        <img src="https://gearsets.eorzeacollection.com/mogstation/other/gaias/hyur-female-legs.png">
        <img src="https://gearsets.eorzeacollection.com/mogstation/other/gaias/hyur-female-feet.png">
        <section class="b-info-box">
        <h4 class="b-heading u-no-top-margin">Equipment</h4>
        <div>
        <div class="c-gear-slot c-gear-slot-head u-in-box"></div>
        <div>
        <a href="https://na.finalfantasyxiv.com/lodestone/playguide/db/item/a1dad7ffe06" class="c-gear-slot c-gear-slot-body b-info-box-item eorzeadb_link">
        <img src="icon.png">
        </a>
        <div class="c-gear-slot-item">
        <a href="https://na.finalfantasyxiv.com/lodestone/playguide/db/item/a1dad7ffe06" class="eorzeadb_link">
        <span class="c-gear-slot-item-name u-rarity-basic">Gaia's Attire</span>
        </a>
        </div>
        <a href="/glamours/body/22115">glamours using this piece</a>
        </div>
        <div class="c-gear-slot c-gear-slot-hands u-in-box"></div>
        <div class="c-gear-slot c-gear-slot-legs u-in-box"></div>
        <div class="c-gear-slot c-gear-slot-feet u-in-box"></div>
        </div>
        <div class="u-no-sidebar-only"></div>
        </section>
        </body></html>
        """
        record = parse_gearset_detail(html, "https://ffxiv.eorzeacollection.com/gearset/gaias")
        self.assertEqual(record.present_slots, ["BODY"])
        self.assertEqual(record.missing_slots, ["HEAD", "HANDS", "LEGS", "FEET"])
        self.assertEqual(len(record.pieces), 1)
        self.assertEqual(record.pieces[0].name, "Gaia's Attire")
        self.assertEqual(record.pieces[0].slot, "BODY")
        self.assertEqual(record.pieces[0].occupied_slots, ["BODY", "HANDS", "LEGS", "FEET"])
        self.assertEqual(record.pieces[0].blocked_slots, ["HANDS", "LEGS", "FEET"])

    def test_parse_current_list_card_equipment_markup(self) -> None:
        html = """
        <html><head><title>Neo Queen's Set | Eorzea Collection</title></head><body>
        <img src="https://gearsets.eorzeacollection.com/mogstation/scions/neo-queens/hyur-female-front.png">
        <img src="https://gearsets.eorzeacollection.com/mogstation/scions/neo-queens/hyur-female-head.png">
        <img src="https://gearsets.eorzeacollection.com/mogstation/scions/neo-queens/hyur-female-body.png">
        <img src="https://gearsets.eorzeacollection.com/mogstation/scions/neo-queens/hyur-female-hands.png">
        <img src="https://gearsets.eorzeacollection.com/mogstation/scions/neo-queens/hyur-female-legs.png">
        <img src="https://gearsets.eorzeacollection.com/mogstation/scions/neo-queens/hyur-female-feet.png">
        <h2>— Equipment —</h2>
        <div>
          <div class="list box">
            <span class="gear-icon-box-slot-name">HEAD</span>
            <div class="list-item">
              <div class="list-item-title"><span>Neo Queen's Tiara</span></div>
              <a href="/glamours/head/27852">Glamours using this piece</a>
            </div>
          </div>
          <div class="list box">
            <span class="gear-icon-box-slot-name">BODY</span>
            <div class="list-item">
              <div class="list-item-title"><span>Neo Queen's Dress</span></div>
              <a href="/glamours/body/27853">Glamours using this piece</a>
            </div>
          </div>
          <div class="list box"><span class="gear-icon-box-slot-name">HANDS</span></div>
          <div class="list box"><span class="gear-icon-box-slot-name">LEGS</span></div>
          <div class="list box"><span class="gear-icon-box-slot-name">FEET</span></div>
        </div>
        </body></html>
        """
        record = parse_gearset_detail(
            html,
            "https://ffxiv.eorzeacollection.com/gearset/neo-queens",
        )
        self.assertEqual(record.present_slots, ["HEAD", "BODY"])
        self.assertEqual(record.missing_slots, ["HANDS", "LEGS", "FEET"])
        self.assertEqual(
            [(piece.slot, piece.name) for piece in record.pieces],
            [
                ("HEAD", "Neo Queen's Tiara"),
                ("BODY", "Neo Queen's Dress"),
            ],
        )
        self.assertEqual(
            record.pieces[0].source_item_url,
            "https://ffxiv.eorzeacollection.com/glamours/head/27852",
        )
        self.assertEqual(
            record.pieces[1].occupied_slots,
            ["BODY", "HANDS", "LEGS", "FEET"],
        )
        self.assertEqual(record.pieces[1].blocked_slots, ["HANDS", "LEGS", "FEET"])

    def test_title_cleanup_handles_repeated_site_name(self) -> None:
        html = """
        <html><head><title>Vana'dielian Fending Set | Eorzea Collection Eorzea Collection</title></head><body>
        <h2>Equipment</h2>
        </body></html>
        """
        record = parse_gearset_detail(html, "https://ffxiv.eorzeacollection.com/gearset/vanadielian-fending")
        self.assertEqual(record.title, "Vana'dielian Fending Set")

    def test_parse_static_gearsets_image_url(self) -> None:
        url = "https://gearsets.eorzeacollection.com/crafted/leveling/courtly-lovers/courtly-lovers-healing/hyur-male-legs.png"
        meta = parse_gearsets_image_url(url)
        self.assertIsNotNone(meta)
        assert meta is not None
        self.assertEqual(meta.source_type, "crafted")
        self.assertEqual(meta.source_path, "crafted/leveling/courtly-lovers")
        self.assertEqual(meta.collection_slug, "courtly-lovers")
        self.assertEqual(meta.gearset_slug, "courtly-lovers-healing")
        self.assertEqual(meta.race, "hyur")
        self.assertEqual(meta.model_gender, "male")
        self.assertEqual(meta.slot, "LEGS")

    def test_generate_same_gearset_variant_urls(self) -> None:
        urls = generate_same_gearset_variant_urls(
            [
                "https://gearsets.eorzeacollection.com/alliance/vanadiel/windurst/vanadielian-fending/hyur-male-legs.png"
            ]
        )
        self.assertIn(
            "https://gearsets.eorzeacollection.com/alliance/vanadiel/windurst/vanadielian-fending/viera-female-dyed-feet.png",
            urls,
        )


if __name__ == "__main__":
    unittest.main()
