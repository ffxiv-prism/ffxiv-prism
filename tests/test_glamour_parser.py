from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from ffxiv_prism.collectors.eorzea_collection.glamour_parser import (
    appearance_name_key,
    enrich_equipment_from_sqlite,
    parse_glamour_page,
)


HTML = """
<html><head>
<meta property="og:url" content="https://ffxiv.eorzeacollection.com/glamour/302518/gaia">
<meta property="og:image" content="https://glamours.eorzeacollection.com/302518/cover.png">
</head><body><main>
<h1 class="title"><span class="has-text-primary">Gaia</span></h1>
<div><label>Tags:</label><a class="tag">Black</a><a class="tag">Grunge</a></div>
<div><label>Submitted on:</label><div class="tag">Aug 27th, 2025</div></div>
<img src="https://glamours.eorzeacollection.com/302518/cover.png">
<img src="https://glamours.eorzeacollection.com/302518/gaia-0.png">
<img alt="Glamour icon" src="/resources/v7/icons/ffxiv/content/glamour.png">
<img alt="Glamour image" src="https://glamours.eorzeacollection.com/999999/unrelated.png">
<div class="divider">Description</div><div class="box">Location: Test</div>
<div class="list box">
  <span class="gear-icon-box-slot-name">HEAD</span>
  <a class="eorzeadb_link" href="https://na.finalfantasyxiv.com/lodestone/playguide/db/item/abc"><img class="gear-icon-image" alt="Alternative Cap" src="https://icons.example/cap.png"></a>
  <div class="list-item-description"><div class="tag"><span>⬤ Dark Purple</span></div><div class="tag"><span>◯ Undyed</span></div></div>
  <a href="/glamours/head/26285">Glamours</a>
</div>
<div class="list box"><span class="gear-icon-box-slot-name">HANDS</span></div>
<div class="list box">
  <span class="gear-icon-box-slot-name">RING</span>
  <img class="gear-icon-image" alt="Test Ring" src="https://icons.example/ring.png">
</div>
<div class="divider">Creator</div><div class="box"><a href="/creator/4317"><h3>Elysia Vys</h3><h4>⧫ phoenix</h4></a></div>
<div class="divider">About this glamour</div><div class="box">
  <div class="columns"><div class="column"><span class="subtitle">Model</span></div><div class="column"><span>Au Ra Female</span></div></div>
  <div class="columns"><div class="column"><span class="subtitle">Level to Equip</span></div><div class="column"><span>90</span></div></div>
</div>
<article class="collection glamour"><a class="glamour-image" href="/glamour/123/related"></a></article>
</main></body></html>
"""


class GlamourParserTests(unittest.TestCase):
    def test_parse_labeled_glamour_page(self) -> None:
        record = parse_glamour_page(HTML, "https://ffxiv.eorzeacollection.com/glamour/302518")
        self.assertEqual(record["glamour_id"], 302518)
        self.assertEqual(record["slug"], "gaia")
        self.assertEqual(record["title"], "Gaia")
        self.assertEqual(record["submitted_at"], "2025-08-27")
        self.assertEqual(record["tags"], ["Black", "Grunge"])
        self.assertEqual(record["model_race"], "Au Ra")
        self.assertEqual(record["model_gender"], "female")
        self.assertEqual(
            record["image_urls"],
            [
                "https://glamours.eorzeacollection.com/302518/cover.png",
                "https://glamours.eorzeacollection.com/302518/gaia-0.png",
            ],
        )
        self.assertEqual(record["related_glamour_urls"], ["https://ffxiv.eorzeacollection.com/glamour/123/related"])

        self.assertEqual(len(record["equipment"]), 2)
        head, hands = record["equipment"]
        self.assertEqual(head["slot"], "HEAD")
        self.assertEqual(head["item_name"], "Alternative Cap")
        self.assertEqual(head["site_item_id"], 26285)
        self.assertEqual(head["dyes"], ["Dark Purple", "Undyed"])
        self.assertTrue(hands["is_empty"])


    def test_enrichment_marks_same_model_variant_without_claiming_exact_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE gear_pieces (
                    piece_id INTEGER,
                    slot TEXT,
                    name_en TEXT,
                    occupied_slots_json TEXT,
                    blocked_slots_json TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO gear_pieces VALUES (?, ?, ?, ?, ?)",
                (7, "HANDS", "Nightsteel Gauntlets of Maiming", "[]", "[]"),
            )
            conn.commit()
            conn.close()

            record = {
                "equipment": [
                    {
                        "slot": "HANDS",
                        "is_core_slot": True,
                        "is_empty": False,
                        "item_name": "Augmented Nightsteel Gauntlets of Maiming",
                    }
                ]
            }
            enrich_equipment_from_sqlite(record, db_path)
            item = record["equipment"][0]
            self.assertEqual(item["piece_match_type"], "appearance_equivalent")
            self.assertEqual(item["matched_piece_ids"], [7])
            self.assertEqual(item["matched_piece_names"], ["Nightsteel Gauntlets of Maiming"])
            self.assertEqual(item["appearance_equivalence_rule"], "normalized_variant_name")

    def test_enrichment_expands_canonical_embedding_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE gear_pieces (
                    piece_id INTEGER,
                    slot TEXT,
                    name_en TEXT,
                    occupied_slots_json TEXT,
                    blocked_slots_json TEXT
                );
                CREATE TABLE images (
                    image_row_id INTEGER,
                    family_slug TEXT
                );
                CREATE TABLE gear_piece_images (
                    piece_id INTEGER,
                    image_row_id INTEGER
                );
                """
            )
            conn.executemany(
                "INSERT INTO gear_pieces VALUES (?, ?, ?, ?, ?)",
                [
                    (10, "BODY", "Shisui Joi of Fending", "[]", "[]"),
                    (11, "BODY", "Shisui Joi of Healing", "[]", "[]"),
                    (12, "HEAD", "Shisui Mengu of Fending", "[]", "[]"),
                ],
            )
            conn.executemany(
                "INSERT INTO images VALUES (?, ?)",
                [(100, "shisui"), (101, "shisui"), (102, "shisui")],
            )
            conn.executemany(
                "INSERT INTO gear_piece_images VALUES (?, ?)",
                [(10, 100), (11, 101), (12, 102)],
            )
            conn.commit()
            conn.close()

            record = {
                "equipment": [
                    {
                        "slot": "BODY",
                        "is_core_slot": True,
                        "is_empty": False,
                        "item_name": "Shisui Joi of Fending",
                    }
                ]
            }
            enrich_equipment_from_sqlite(
                record,
                db_path,
                appearance_equivalence={10: [11], 11: [10]},
            )
            item = record["equipment"][0]
            self.assertEqual(item["piece_match_type"], "exact_item")
            self.assertEqual(item["matched_piece_ids"], [10, 11])
            self.assertEqual(
                item["appearance_equivalence_rule"],
                "canonical_embedding_similarity",
            )

    def test_appearance_name_key_normalizes_known_same_model_variants(self) -> None:
        self.assertEqual(
            appearance_name_key("Augmented Nightsteel Gauntlets of Maiming"),
            appearance_name_key("Nightsteel Gauntlets of Maiming"),
        )
        self.assertEqual(
            appearance_name_key("Fieldfiend's Costume Hat"),
            appearance_name_key("Fieldfiend's Hat"),
        )
        self.assertEqual(
            appearance_name_key("Anemos Channeler's Armlets"),
            appearance_name_key("Channeler's Armlets"),
        )

    def test_appearance_name_key_keeps_unrelated_items_distinct(self) -> None:
        self.assertNotEqual(
            appearance_name_key("Ascension Longkilt of Casting"),
            appearance_name_key("Anabaseios Longkilt of Casting"),
        )

if __name__ == "__main__":
    unittest.main()
