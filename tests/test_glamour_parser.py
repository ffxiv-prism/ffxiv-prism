from __future__ import annotations

import unittest

from ffxiv_prism.collectors.eorzea_collection.glamour_parser import parse_glamour_page


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


if __name__ == "__main__":
    unittest.main()
