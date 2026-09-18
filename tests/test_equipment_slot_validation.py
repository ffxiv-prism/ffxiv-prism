from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from ffxiv_prism.collectors.eorzea_collection.parser import parse_gearset_detail
from ffxiv_prism.dataset.sqlite_store import insert_gear_piece_images
from ffxiv_prism.matching.equipment_slots import (
    official_slots_by_name, filter_candidates_for_slot,
)
from ffxiv_prism.matching.inference import select_slot_candidate_index_rows
from scripts.enrich_gear_pieces_with_zh_names import apply_official_equip_slots


class EquipmentSlotValidationTests(unittest.TestCase):
    def test_empty_feet_slot_does_not_consume_weapon_or_accessory(self):
        for other_slot in ("mainhand", "earrings"):
            with self.subTest(other_slot=other_slot):
                html = f'''<h4>Equipment</h4>
                <div class="c-gear-slot c-gear-slot-feet"></div>
                <a class="c-gear-slot c-gear-slot-{other_slot}"
                   href="https://na.finalfantasyxiv.com/lodestone/playguide/db/item/bad">
                  <span class="c-gear-slot-item-name">Not shoes</span>
                </a></section>'''
                result = parse_gearset_detail(html, "https://ffxiv.eorzeacollection.com/gearset/test")
                self.assertEqual(result.pieces, [])

    def test_text_fallback_stops_at_accessories(self):
        html = '''<h4>Equipment</h4><p>FEET</p>
        <a href="https://na.finalfantasyxiv.com/lodestone/playguide/db/item/boots">Boots</a>
        <p>EARRINGS</p>
        <a href="https://na.finalfantasyxiv.com/lodestone/playguide/db/item/earrings">Ear cuffs</a>'''
        result = parse_gearset_detail(html, "https://ffxiv.eorzeacollection.com/gearset/test")
        self.assertEqual([(p.slot, p.name) for p in result.pieces], [("FEET", "Boots")])

    def test_official_non_armour_evidence_clears_inferred_feet(self):
        row = {"slot": "FEET", "occupied_slots": ["FEET"], "blocked_slots": []}
        apply_official_equip_slots(row, [13], {13: {"Head": 0, "Body": 0, "Feet": 0}})
        self.assertEqual(row["occupied_slots"], [])
        self.assertTrue(row["official_equip_slot_source"])

    def test_index_filter_preserves_vector_alignment_and_valid_multislot_items(self):
        official = official_slots_by_name([
            {"name_en": "Staff", "official_equip_slot_source": "official", "official_occupied_slots": []},
            {"name_en": "Boots", "official_equip_slot_source": "official", "official_occupied_slots": ["FEET"]},
            {"name_en": "Full outfit", "official_equip_slot_source": "official", "official_occupied_slots": ["BODY", "FEET"]},
        ])
        rows = [
            {"slot": "FEET", "pieces": [{"name_en": "Staff"}]},
            {"slot": "FEET", "pieces": [{"name_en": "Staff"}, {"name_en": "Boots"}]},
            {"slot": "FEET", "pieces": [{"name_en": "Full outfit"}]},
            {"slot": "BODY", "pieces": [{"name_en": "Full outfit"}]},
        ]
        vectors = np.arange(12, dtype=np.float32).reshape(4, 3)
        selected, selected_vectors = select_slot_candidate_index_rows(rows, vectors, "FEET", official_slots=official)
        self.assertEqual([r["pieces"][0]["name_en"] for r in selected], ["Boots", "Full outfit"])
        np.testing.assert_array_equal(selected_vectors, vectors[[1, 2]])
        self.assertEqual(len(rows[1]["pieces"]), 2)

    def test_same_model_alternatives_are_checked_too(self):
        candidates = [{"name_en": "Boots", "rank": 2, "appearance_alternatives": [
            {"name_en": "Staff"}, {"name_en": "Unknown boots"},
        ]}]
        result = filter_candidates_for_slot(candidates, "FEET", {"Staff": frozenset()})
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[0]["appearance_alternatives"], [{"name_en": "Unknown boots"}])

    def test_database_links_ignore_false_exact_slot_and_keep_occupied_slot(self):
        pieces = [
            {"gearset_slug": "test", "slot": "FEET", "name": "Staff", "occupied_slots": ["FEET"],
             "official_equip_slot_source": "official", "official_occupied_slots": []},
            {"gearset_slug": "test", "slot": "BODY", "name": "Full outfit",
             "official_equip_slot_source": "official", "official_occupied_slots": ["BODY", "FEET"]},
        ]
        conn = Mock()
        conn.execute.return_value.fetchone.return_value = (1,)
        count = insert_gear_piece_images(
            conn, pieces, [{"gearset_slug": "test", "slot": "FEET", "source_image_url": "feet.png"}],
            {("test", "FEET", None, "Staff"): 1, ("test", "BODY", None, "Full outfit"): 2},
            {"feet.png": 3},
        )
        self.assertEqual(count, 1)
        self.assertEqual(conn.executemany.call_args.args[1], [(2, 3, "occupied_slot")])


if __name__ == "__main__":
    unittest.main()
