"""Alternate headgear views must survive discovery and retain one identity."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ffxiv_prism.collectors.eorzea_collection.image_url import (
    generate_static_image_urls,
    is_full_preview_image,
    parse_gearsets_image_url,
)
from ffxiv_prism.collectors.eorzea_collection.parser import parse_gearset_detail
from ffxiv_prism.dataset.sqlite_store import build_eorzea_sqlite
from ffxiv_prism.vision.retrieval import load_candidate_images
from scripts.build_slot_aligned_candidate_index import expand_slot_aligned_rows
from tests.test_sqlite_store import image_row, write_jsonl


BASE = "https://gearsets.eorzeacollection.com/raid/example/example-fending/"
PAGE = "https://ffxiv.eorzeacollection.com/gearset/example-fending"


class VisorImageTests(unittest.TestCase):
    def test_both_genders_and_dyes_map_to_head_without_losing_state(self):
        for gender in ("female", "male"):
            for dye in ("", "dyed-"):
                url = BASE + f"hyur-{gender}-{dye}visor.png"
                meta = parse_gearsets_image_url(url)
                self.assertEqual(meta.slot, "HEAD")
                self.assertEqual(meta.image_part, "visor")
                self.assertEqual(meta.dye_state, "dyed" if dye else "normal")
                self.assertFalse(is_full_preview_image(url))
        self.assertIsNone(parse_gearsets_image_url(BASE + "hyur-female-unknown.png"))

    def test_discovery_includes_alternate_view_without_changing_explicit_parts(self):
        seed = [BASE + "hyur-female-head.png"]
        urls = generate_static_image_urls(seed, races=("hyur",), genders=("female",), dye_states=("normal",))
        self.assertIn(BASE + "hyur-female-visor.png", urls)
        self.assertEqual(generate_static_image_urls(seed, races=("hyur",), genders=("female",), dye_states=("normal",), parts=("head",)), seed)

    def test_page_with_visor_only_is_discoverable(self):
        html = f'<img src="{BASE}hyur-female-visor.png">'
        result = parse_gearset_detail(html, PAGE)
        views = [image for image in result.images if image.image_part == "visor"]
        self.assertTrue(views)
        self.assertTrue(all(image.slot == "HEAD" for image in views))
        self.assertTrue(all(image.gearset_slug == "example-fending" for image in views))

    def test_both_states_reach_multiscale_index_as_same_piece(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "sets.jsonl", [{"slug":"test-set", "source_page_url":PAGE}])
            write_jsonl(root / "pieces.jsonl", [{"gearset_slug":"test-set", "slot":"HEAD", "name":"Test Headgear", "name_en":"Test Headgear", "source_item_url":"https://example.test/item/1", "occupied_slots":["HEAD"]}])
            images = []
            for part in ("head", "visor"):
                row = image_row(BASE + f"hyur-female-{part}.png", "HEAD")
                row["image_part"] = part
                images.append(row)
            write_jsonl(root / "images.jsonl", images)
            build_eorzea_sqlite(db_path=root / "test.sqlite", gearsets_path=root / "sets.jsonl", gear_pieces_path=root / "pieces.jsonl", images_path=root / "images.jsonl", summary_path=root / "summary.json")
            candidates = load_candidate_images(root / "test.sqlite", slots=("HEAD",))
            self.assertEqual({r["image_part"] for r in candidates}, {"head", "visor"})
            self.assertEqual(len({r["pieces"][0]["piece_id"] for r in candidates}), 1)
            self.assertTrue(all(r["pieces"][0]["link_type"] == "exact_slot" for r in candidates))
            aligned = expand_slot_aligned_rows(candidates, head_crop_policy="multiscale")
            self.assertEqual(len(aligned), 6)
            self.assertEqual(len({r["local_path"] for r in aligned}), 2)
            self.assertEqual(len({r["source_image_row_id"] for r in aligned}), 2)


if __name__ == "__main__":
    unittest.main()
