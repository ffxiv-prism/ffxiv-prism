from __future__ import annotations

import unittest

from ffxiv_prism.vision.outfit_regions import (
    Keypoint,
    build_outfit_regions,
    estimate_person_box,
    pose_quality,
    region_manifest,
    pose_selection_score,
    single_person_search_boxes,
)


class OutfitRegionTests(unittest.TestCase):
    def setUp(self) -> None:
        coordinates = [
            (100, 45), (94, 42), (106, 42), (88, 45), (112, 45),
            (78, 88), (122, 88), (65, 135), (135, 135), (55, 175), (145, 175),
            (85, 185), (115, 185), (82, 270), (118, 270), (80, 360), (120, 360),
        ]
        self.keypoints = [Keypoint(x, y, 0.9) for x, y in coordinates]

    def test_person_box_refines_around_single_pose(self) -> None:
        box = estimate_person_box(self.keypoints, (400, 500))
        self.assertLess(box[0], 55)
        self.assertGreater(box[2], 145)
        self.assertLess(box[1], 42)
        self.assertGreater(box[3], 360)
        self.assertLess(box[2] - box[0], 400)

    def test_builds_multiple_pose_relative_crops_per_slot(self) -> None:
        regions = build_outfit_regions(self.keypoints, (400, 500))
        by_slot = {}
        for region in regions:
            by_slot.setdefault(region.slot, []).append(region)
            left, top, right, bottom = region.box
            self.assertGreater(right, left)
            self.assertGreater(bottom, top)
        self.assertEqual(set(by_slot), {"HEAD", "BODY", "HANDS", "LEGS", "FEET"})
        self.assertEqual(len(by_slot["HEAD"]), 4)
        self.assertEqual(len(by_slot["BODY"]), 3)
        self.assertEqual(len(by_slot["HANDS"]), 5)
        self.assertEqual(len(by_slot["LEGS"]), 5)
        self.assertEqual(len(by_slot["FEET"]), 5)
        self.assertTrue(all(region.visibility == "visible" for region in regions))

    def test_overlap_aware_regions_are_tagged_for_retrieval_weighting(self) -> None:
        regions = build_outfit_regions(self.keypoints, (400, 500))
        by_name = {region.name: region for region in regions}
        self.assertEqual(by_name["body_upper_arm_overlap"].role, "auxiliary")
        self.assertIn("body_hands_overlap", by_name["body_upper_arm_overlap"].tags)
        self.assertIn("long_glove_or_sleeve_overlap", by_name["hands_forearm_left"].tags)
        self.assertIn("legs_feet_overlap", by_name["legs_boot_overlap"].tags)
        self.assertIn("long_boot_focus", by_name["feet_boot_left"].tags)
        self.assertIn("may_include_face_hair", by_name["head_tight"].tags)

    def test_region_manifest_exposes_role_and_tags(self) -> None:
        regions = build_outfit_regions(self.keypoints, (400, 500))
        manifest = region_manifest(self.keypoints, (0, 0, 200, 400), regions)
        body_overlap = next(region for region in manifest["regions"] if region["name"] == "body_upper_arm_overlap")
        self.assertEqual(body_overlap["role"], "auxiliary")
        self.assertIn("body_hands_overlap", body_overlap["tags"])

    def test_low_confidence_feet_are_not_marked_visible(self) -> None:
        keypoints = list(self.keypoints)
        keypoints[15] = Keypoint(80, 499, 0.1)
        keypoints[16] = Keypoint(120, 499, 0.1)
        feet = [region for region in build_outfit_regions(keypoints, (400, 500)) if region.slot == "FEET"]
        self.assertTrue(all(region.visibility != "visible" for region in feet))


    def test_pose_selection_score_applies_soft_center_prior(self) -> None:
        center = pose_selection_score(self.keypoints, (100, 0, 300, 400), (400, 400))
        edge = pose_selection_score(self.keypoints, (0, 0, 200, 400), (400, 400))
        self.assertGreater(center, edge)
    def test_wide_images_get_overlapping_single_person_search_windows(self) -> None:
        boxes = single_person_search_boxes((1000, 500))
        self.assertGreater(len(boxes), 8)
        self.assertIn((0, 0, 1000, 500), boxes)
        self.assertTrue(any(box[0] > 0 and box[2] < 1000 for box in boxes))

    def test_pose_quality_rewards_confident_coherent_pose(self) -> None:
        weak = [Keypoint(point.x, point.y, 0.1) for point in self.keypoints]
        self.assertGreater(
            pose_quality(self.keypoints, (0, 0, 200, 400)),
            pose_quality(weak, (0, 0, 200, 400)),
        )


if __name__ == "__main__":
    unittest.main()
