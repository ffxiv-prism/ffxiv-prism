from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from ffxiv_prism.matching.head_detail import (
    DETAIL_NAMES, HEAD_TYPE_CLASSES, detail_presence_decision,
    detail_records, recall_detail_candidate,
)
from ffxiv_prism.matching.inference import crop_weight, current_best_configs, base_configs
from ffxiv_prism.vision.outfit_regions import CropRegion, build_head_detail_regions


def rows(*ids):
    return [dict(piece_id=pid, rank=i, score=1 / i) for i, pid in enumerate(ids, 1)]


class HeadDetailRecallTests(unittest.TestCase):
    def setUp(self):
        self.base = rows(*range(10, 30))
        self.views = dict(zip(DETAIL_NAMES, (rows(100, 80), rows(80, 101), rows(101, 90)), strict=True))
        self.eq = {100: (101,), 101: (100,)}

    def test_two_appearance_votes_recall_outside_pool_and_preserve_top_five(self):
        original = copy.deepcopy(self.base)
        result, diagnostic = recall_detail_candidate(self.base, self.views, self.eq)
        self.assertEqual(result[:5], original[:5])
        self.assertEqual(result[5]["piece_id"], 100)
        self.assertEqual(result[5]["rank"], 6)
        self.assertEqual([r["rank"] for r in result], list(range(1, 22)))
        self.assertEqual(diagnostic["support_ranks"][DETAIL_NAMES[1]], 2)
        self.assertIsNone(diagnostic["previous_rank"])
        self.assertEqual(self.base, original)

    def test_existing_appearance_moves_without_duplicate(self):
        self.base[12]["piece_id"] = 101
        result, diagnostic = recall_detail_candidate(self.base, self.views, self.eq)
        self.assertEqual(result[5]["piece_id"], 101)
        self.assertEqual(len(result), len(self.base))
        self.assertEqual(diagnostic["previous_rank"], 13)

    def test_high_ranked_appearance_is_unchanged(self):
        for index in (0, 4, 5):
            base = copy.deepcopy(self.base)
            base[index]["piece_id"] = 101
            result, diagnostic = recall_detail_candidate(base, self.views, self.eq)
            self.assertEqual(result, base)
            self.assertFalse(diagnostic["ranking_changed"])

    def test_missing_view_disagreement_and_unsupported_vote_fall_back(self):
        for views in (
            {DETAIL_NAMES[0]: rows(100)},
            dict(zip(DETAIL_NAMES, (rows(100), rows(80), rows(90)), strict=True)),
            dict(zip(DETAIL_NAMES, (rows(100), rows(80), rows(101)), strict=True)),
            dict(zip(DETAIL_NAMES, (rows(100), rows(*range(200, 210), 101), rows(101)), strict=True)),
        ):
            result, diagnostic = recall_detail_candidate(self.base, views, self.eq)
            self.assertEqual(result, self.base)
            self.assertFalse(diagnostic["active"])

    def test_records_require_all_three_unique_reliable_views(self):
        records = [dict(name=name, path=name, visibility="visible", confidence=.8) for name in DETAIL_NAMES]
        self.assertEqual(len(detail_records(records)), 3)
        self.assertEqual(detail_records(records[:2] + records[:1]), [])
        for bad in (dict(visibility="partially_visible"), dict(visibility="pose_uncertain"), dict(confidence=.54)):
            changed = copy.deepcopy(records)
            changed[0].update(bad)
            self.assertEqual(detail_records(changed), [])

    def test_detail_geometry_preserves_parent_quality_and_zero_weight(self):
        parent = CropRegion("HEAD", "head_lower_face", (20, 40, 120, 240), .9, "visible", "diagnostic")
        detail = build_head_detail_regions([parent])[0]
        self.assertEqual(detail.box, (20, 120, 120, 240))
        self.assertEqual(crop_weight(vars(detail)), 0)
        for visibility in ("pose_uncertain", "partially_visible"):
            bad = replace(parent, visibility=visibility, tags=("source_edge_truncation_risk",))
            result = build_head_detail_regions([bad])[0]
            self.assertEqual(result.visibility, visibility)
            self.assertIn("source_edge_truncation_risk", result.tags)

    def test_only_current_best_head_enables_detail_recall(self):
        self.assertTrue(current_best_configs()["HEAD"].head_detail_recall)
        self.assertFalse(base_configs()["HEAD"].head_detail_recall)
        self.assertTrue(all(not c.head_detail_recall for s, c in current_best_configs().items() if s != "HEAD"))

    def test_presence_requires_clear_local_accessory_and_ear_agreement(self):
        def profile(name, probability):
            values = [(1 - probability) / (len(HEAD_TYPE_CLASSES) - 1)] * len(HEAD_TYPE_CLASSES)
            values[HEAD_TYPE_CLASSES.index(name)] = probability
            return values
        profiles = dict(zip(DETAIL_NAMES, (profile("headset", .9), profile("bare", .8), profile("headset", .6)), strict=True))
        self.assertTrue(detail_presence_decision(profiles)["active"])
        for cls in ("bare", "hat", "helmet", "hair_ornament"):
            self.assertFalse(detail_presence_decision({n: profile(cls, .9) for n in DETAIL_NAMES})["active"])
        weak = dict(profiles, head_lower_face_detail=profile("headset", .74))
        self.assertFalse(detail_presence_decision(weak)["active"])
        conflict = dict(profiles, head_ear_right_detail=profile("mask", .9))
        self.assertFalse(detail_presence_decision(conflict)["active"])
        self.assertFalse(detail_presence_decision({})["active"])
        with self.assertRaises(ValueError):
            detail_presence_decision({n: [1] for n in DETAIL_NAMES})


if __name__ == "__main__":
    unittest.main()
