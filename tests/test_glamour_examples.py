from __future__ import annotations

import unittest

from ffxiv_prism.dataset.glamour_examples import resolve_core_slots


class GlamourExamplesTests(unittest.TestCase):
    def test_multislot_item_labels_blocked_slots(self) -> None:
        labels = resolve_core_slots(
            [
                {
                    "slot": "BODY",
                    "is_core_slot": True,
                    "is_empty": False,
                    "item_name": "Gaia's Attire",
                    "matched_piece_ids": [3866],
                    "occupied_slots": ["BODY", "HANDS", "LEGS", "FEET"],
                    "dyes": ["Undyed"],
                },
                {"slot": "HANDS", "is_core_slot": True, "is_empty": True},
                {"slot": "LEGS", "is_core_slot": True, "is_empty": True},
                {"slot": "FEET", "is_core_slot": True, "is_empty": True},
            ]
        )
        self.assertEqual(labels["BODY"]["label_type"], "direct")
        self.assertEqual(labels["HANDS"]["label_type"], "occupied_by_multislot")
        self.assertEqual(labels["LEGS"]["piece_ids"], [3866])
        self.assertEqual(labels["FEET"]["source_slot"], "BODY")
        self.assertEqual(labels["HEAD"]["label_type"], "empty_or_unlabeled")


if __name__ == "__main__":
    unittest.main()
