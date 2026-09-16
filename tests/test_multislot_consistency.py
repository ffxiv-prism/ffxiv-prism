from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ffxiv_prism.matching.inference import (
    apply_multislot_consistency_gate,
    write_html_report,
    write_json_report,
)


def candidate(piece_id: int, rank: int = 1, **extra) -> dict:
    return {"piece_id": piece_id, "rank": rank, "score": 1.0 / rank, **extra}


def result(slot: str, *candidates: dict, **extra) -> dict:
    return {"slot": slot, "status": "ok", "candidates": list(candidates), **extra}


class MultislotConsistencyTests(unittest.TestCase):
    def gate(self, results, occupied, top_k=5):
        return apply_multislot_consistency_gate(
            results, occupied, support_top_k=top_k
        )

    def test_latest_report_neo_queen_dress_is_absent_from_json_and_html(self):
        # 2026-09-16 report 0a026739f780: strong hands/feet matches,
        # but no body candidate and only rank 12 in legs.
        dress = {"name_en": "Neo Queen's Dress", "name_zh": "新生女王礼服"}
        slots = [
            result("BODY", candidate(1)),
            result("HANDS", candidate(11739, **dress), candidate(2, 2)),
            result("LEGS", candidate(3), candidate(11739, 12, **dress)),
            result("FEET", candidate(11739, **dress), candidate(4, 2)),
        ]
        filtered = self.gate(slots, {11739: ("BODY", "HANDS", "LEGS", "FEET")})
        self.assertEqual(
            [[row["piece_id"] for row in slot["candidates"]] for slot in filtered],
            [[1], [2], [3], [4]],
        )
        with tempfile.TemporaryDirectory() as directory:
            payload = {"slots": filtered}
            json_path = Path(directory) / "results.json"
            html_path = Path(directory) / "report.html"
            write_json_report(json_path, payload)
            write_html_report(html_path, payload)
            self.assertNotIn("11739", json_path.read_text(encoding="utf-8"))
            self.assertNotIn("新生女王礼服", html_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), payload)

    def test_all_occupied_slots_must_meet_top_k_including_current_slot(self):
        for weak_rank in (5, 6):
            with self.subTest(weak_rank=weak_rank):
                slots = [
                    result("BODY", candidate(10)),
                    result("HANDS", candidate(10, weak_rank)),
                    result("LEGS", candidate(10, 2)),
                ]
                filtered = self.gate(slots, {10: ("BODY", "HANDS", "LEGS")})
                for slot in filtered:
                    self.assertEqual(len(slot["candidates"]), int(weak_rank == 5))
                    if weak_rank == 5:
                        self.assertEqual(
                            slot["candidates"][0]["cross_slot_consistency"]["state"],
                            "supported",
                        )
                    else:
                        self.assertEqual(slot["status"], "no_usable_candidates")

    def test_missing_empty_or_unusable_slot_cannot_supply_support(self):
        for other in ([], [result("HANDS")], [
            result("HANDS", candidate(10), status="no_usable_candidates")
        ]):
            with self.subTest(other=other):
                filtered = self.gate(
                    [result("BODY", candidate(10)), *other],
                    {10: ("BODY", "HANDS")},
                )
                self.assertFalse(filtered[0]["candidates"])

    def test_same_model_alternative_supplies_support_at_anchor_rank(self):
        filtered = self.gate([
            result("BODY", candidate(10)),
            result("HANDS", candidate(20, appearance_alternatives=[
                {"piece_id": 10, "original_rank": 99, "relation": "same_model"}
            ])),
        ], {10: ("BODY", "HANDS")})
        self.assertEqual(filtered[0]["candidates"][0]["piece_id"], 10)
        self.assertEqual(
            filtered[0]["candidates"][0]["cross_slot_consistency"]["support"]["HANDS"]["rank"],
            1,
        )
        self.assertEqual(len(filtered[1]["candidates"][0]["appearance_alternatives"]), 1)

    def test_unsupported_same_model_alternative_is_removed_from_report(self):
        filtered = self.gate([
            result("BODY", candidate(20, appearance_alternatives=[
                {"piece_id": 10, "name_en": "Unsupported outfit"},
                {"piece_id": 30, "name_en": "Single-slot alternative"},
            ])),
        ], {10: ("BODY", "HANDS")})
        self.assertEqual(
            [row["piece_id"] for row in filtered[0]["candidates"][0]["appearance_alternatives"]],
            [30],
        )
        self.assertEqual(filtered[0]["cross_slot_consistency"]["removed_alternative_count"], 1)

    def test_removed_anchor_cannot_support_other_slots_regardless_of_order(self):
        slots = [
            result("BODY", candidate(10)),
            result("HANDS", candidate(20, appearance_alternatives=[{"piece_id": 10}])),
        ]
        # The supporting anchor fails because FEET is missing. BODY must
        # then lose its support too, regardless of processing order.
        for ordered in (slots, list(reversed(slots))):
            filtered = self.gate(ordered, {
                10: ("BODY", "HANDS"), 20: ("HANDS", "FEET")
            })
            self.assertTrue(all(not slot["candidates"] for slot in filtered))

    def test_removals_do_not_promote_weak_matches_into_support_pool(self):
        slots = [
            result("BODY", candidate(10), candidate(20, 2), candidate(30, 3)),
            result("HANDS", candidate(20)),
        ]
        filtered = self.gate(slots, {
            10: ("BODY", "FEET"), 20: ("BODY", "HANDS")
        }, top_k=1)
        self.assertEqual([row["piece_id"] for row in filtered[0]["candidates"]], [30])
        self.assertFalse(filtered[1]["candidates"])

    def test_reference_ranks_are_updated_without_mutating_input(self):
        slots = [result(
            "BODY", candidate(10, reference_match=True), candidate(20, 2),
            reference={"rank": 1, "top1_hit": True, "displayed_top_k_hit": True},
            top_score_margin=0.5,
        )]
        original = deepcopy(slots)
        filtered = self.gate(slots, {10: ("BODY", "HANDS")})
        self.assertEqual(slots, original)
        self.assertEqual(filtered[0]["candidates"][0]["rank"], 1)
        self.assertIsNone(filtered[0]["reference"]["rank"])
        self.assertFalse(filtered[0]["reference"]["top1_hit"])
        self.assertFalse(filtered[0]["reference"]["displayed_top_k_hit"])
        self.assertIsNone(filtered[0]["top_score_margin"])

    def test_single_slot_and_no_equipment_candidates_keep_their_order(self):
        slots = [result("HANDS", candidate(-10034, is_no_equipment=True), candidate(20, 2))]
        filtered = self.gate(slots, {20: ("HANDS",)})
        self.assertEqual(filtered[0]["candidates"], slots[0]["candidates"])

    def test_backfills_requested_count_from_deeper_eligible_candidates(self):
        slots = [result("HANDS", *[
            candidate(piece_id, rank)
            for rank, piece_id in enumerate(range(1, 31), start=1)
        ])]
        occupied = {piece_id: ("BODY", "HANDS") for piece_id in range(1, 11)}
        filtered = apply_multislot_consistency_gate(
            slots, occupied, support_top_k=5, display_top_k=20,
        )[0]
        self.assertEqual([row["piece_id"] for row in filtered["candidates"]], list(range(11, 31)))
        self.assertEqual([row["rank"] for row in filtered["candidates"]], list(range(1, 21)))
        diagnostic = filtered["cross_slot_consistency"]
        self.assertEqual(diagnostic["candidate_pool_count"], 30)
        self.assertEqual(diagnostic["backfilled_candidate_count"], 10)
        self.assertEqual(diagnostic["unfilled_candidate_count"], 0)
        self.assertEqual(diagnostic["eligible_candidate_count"], 20)

    def test_backfill_does_not_stop_at_four_times_display_count(self):
        filtered = apply_multislot_consistency_gate(
            [result("BODY", *[candidate(i, i) for i in range(1, 19)])],
            {i: ("BODY", "HANDS") for i in range(1, 16)},
            support_top_k=5, display_top_k=3,
        )[0]
        self.assertEqual([row["piece_id"] for row in filtered["candidates"]], [16, 17, 18])

    def test_backfill_skips_unsupported_multislot_rows_in_reserve(self):
        slots = [
            result("BODY", candidate(10), candidate(20, 2), candidate(30, 3), candidate(40, 4)),
            result("HANDS", candidate(30)),
        ]
        filtered = apply_multislot_consistency_gate(
            slots, {10: ("BODY", "FEET"), 30: ("BODY", "HANDS")},
            support_top_k=1, display_top_k=2,
        )
        self.assertEqual([row["piece_id"] for row in filtered[0]["candidates"]], [20, 40])
        self.assertFalse(filtered[1]["candidates"])

    def test_support_must_be_visible_when_display_count_is_below_five(self):
        slots = [
            result("BODY", candidate(10), candidate(20, 2)),
            result("HANDS", candidate(30), candidate(10, 2)),
        ]
        filtered = apply_multislot_consistency_gate(
            slots, {10: ("BODY", "HANDS")}, support_top_k=5, display_top_k=1,
        )
        self.assertEqual([row["piece_id"] for row in filtered[0]["candidates"]], [20])
        self.assertEqual([row["piece_id"] for row in filtered[1]["candidates"]], [30])
        self.assertEqual(filtered[0]["cross_slot_consistency"]["support_top_k"], 1)

    def test_reference_in_reserve_is_updated_after_backfill_and_final_truncation(self):
        for piece_id, expected_rank in [(30, 2), (40, None)]:
            with self.subTest(piece_id=piece_id):
                slots = [result("BODY", *[
                    candidate(i, rank, reference_match=i == piece_id)
                    for rank, i in enumerate([10, 20, 30, 40], start=1)
                ], reference={"rank": 3, "top1_hit": False, "displayed_top_k_hit": False})]
                filtered = apply_multislot_consistency_gate(
                    slots, {10: ("BODY", "HANDS")}, support_top_k=5, display_top_k=2,
                )[0]
                self.assertEqual(filtered["reference"]["rank"], expected_rank)
                self.assertEqual(filtered["reference"]["displayed_top_k_hit"], expected_rank is not None)

    def test_exhausted_pool_reports_shortfall_without_reintroducing_rejected_rows(self):
        filtered = apply_multislot_consistency_gate(
            [result("BODY", candidate(10), candidate(20, 2))],
            {10: ("BODY", "HANDS")}, support_top_k=5, display_top_k=3,
        )[0]
        self.assertEqual([row["piece_id"] for row in filtered["candidates"]], [20])
        self.assertEqual(filtered["cross_slot_consistency"]["unfilled_candidate_count"], 2)

    def test_out_of_frame_status_is_preserved_with_requested_count(self):
        filtered = apply_multislot_consistency_gate(
            [result("FEET", status="slot_not_in_frame")], {},
            support_top_k=5, display_top_k=20,
        )[0]
        self.assertEqual(filtered["status"], "slot_not_in_frame")
        self.assertFalse(filtered["candidates"])

    def test_rejects_nonpositive_display_top_k(self):
        with self.assertRaises(ValueError):
            apply_multislot_consistency_gate([], {}, support_top_k=5, display_top_k=0)

    def test_rejects_nonpositive_support_top_k(self):
        with self.assertRaises(ValueError):
            self.gate([], {}, top_k=0)


if __name__ == "__main__":
    unittest.main()
