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
    parse_glamour_detail,
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
        self.assertEqual(
            record.pieces[0].glamours_using_piece_url,
            "https://ffxiv.eorzeacollection.com/glamours?piece=test-hat",
        )
        self.assertTrue(any(image.model_gender == "female" and image.race == "hyur" for image in record.images))
        self.assertTrue(all("resources" not in image.source_image_url for image in record.images))

    def test_parse_glamour_detail_profile(self) -> None:
        html = """
        <html><head><title>Street Mage | Eorzea Collection</title></head><body>
        Female Miqo'te RDM
        <img src="/storage/images/glamour-1.jpg" alt="Female Miqo'te red mage glamour">
        </body></html>
        """
        record = parse_glamour_detail(html, "https://ffxiv.eorzeacollection.com/glamour/123")
        self.assertEqual(record.slug, "123")
        self.assertEqual(record.genders, ["Female"])
        self.assertIn("Miqo'te", record.races)
        self.assertEqual(record.jobs, ["RDM"])
        self.assertEqual(record.images, [])

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
