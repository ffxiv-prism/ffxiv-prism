from __future__ import annotations

import unittest

from PIL import Image

from ffxiv_prism.vision.outfit_regions import (
    CropRegion,
    Keypoint,
    build_outfit_regions,
    estimate_person_box,
    pose_quality,
    region_manifest,
    pose_selection_score,
    render_debug_overlay,
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

    def test_person_box_keeps_saved_thick_soled_shoes_and_headwear(self) -> None:
        # Pose coordinates from the user's saved 20.png regression.  The old
        # refined person box ended at y=697 while the full-shoe envelope ended
        # at y=724, clipping the platform soles before the second pose pass.
        values = (
            (295.8860, 180.0483, 0.9691),
            (308.4083, 168.3309, 0.9304),
            (283.5625, 165.5135, 0.9918),
            (324.7632, 175.7401, 0.9161),
            (264.9973, 168.9355, 0.9082),
            (351.8096, 228.2806, 0.9081),
            (252.5104, 234.5408, 0.9426),
            (420.8997, 268.9531, 0.7206),
            (250.0736, 333.7535, 0.9503),
            (432.8489, 262.2969, 0.7443),
            (241.8853, 410.4520, 0.9763),
            (347.7349, 379.3596, 0.8623),
            (280.2456, 382.2255, 0.8775),
            (322.6171, 517.1041, 0.9564),
            (245.6433, 512.3491, 0.9018),
            (355.8662, 652.9371, 0.8595),
            (197.3074, 644.1184, 0.9328),
        )
        keypoints = [Keypoint(*value) for value in values]
        person_box = estimate_person_box(keypoints, (512, 724))
        regions = build_outfit_regions(keypoints, (512, 724))
        full_feet = next(
            region for region in regions if region.name == "feet_full_context"
        )
        full_head = next(
            region for region in regions if region.name == "head_full_accessory"
        )
        self.assertGreaterEqual(person_box[3], full_feet.box[3])
        self.assertLessEqual(person_box[0], full_head.box[0])
        self.assertLessEqual(person_box[1], full_head.box[1])
        self.assertGreaterEqual(person_box[2], full_head.box[2])

    def test_debug_overlay_draws_full_headwear_and_shoe_envelopes(self) -> None:
        image = Image.new("RGB", (200, 200), "#111111")
        quiet_keypoints = [Keypoint(100, 100, 0.0) for _ in range(17)]
        regions = [
            CropRegion("HEAD", "head_context", (50, 50, 100, 100), 0.9, "visible"),
            CropRegion(
                "HEAD",
                "head_full_accessory",
                (20, 20, 140, 120),
                0.9,
                "visible",
                "diagnostic",
            ),
            CropRegion("FEET", "feet_context", (60, 130, 100, 160), 0.9, "visible"),
            CropRegion(
                "FEET",
                "feet_full_context",
                (30, 125, 150, 190),
                0.9,
                "visible",
                "diagnostic",
            ),
        ]
        overlay = render_debug_overlay(
            image,
            quiet_keypoints,
            (0, 0, 2, 2),
            regions,
        )
        self.assertEqual(overlay.getpixel((20, 80)), (255, 90, 95))
        self.assertEqual(overlay.getpixel((30, 180)), (155, 89, 182))

    def test_builds_multiple_pose_relative_crops_per_slot(self) -> None:
        regions = build_outfit_regions(self.keypoints, (400, 500))
        by_slot = {}
        for region in regions:
            by_slot.setdefault(region.slot, []).append(region)
            left, top, right, bottom = region.box
            self.assertGreater(right, left)
            self.assertGreater(bottom, top)
        self.assertEqual(set(by_slot), {"HEAD", "BODY", "HANDS", "LEGS", "FEET"})
        self.assertEqual(len(by_slot["HEAD"]), 5)
        self.assertEqual(len(by_slot["BODY"]), 3)
        self.assertEqual(len(by_slot["HANDS"]), 5)
        self.assertEqual(len(by_slot["LEGS"]), 5)
        self.assertEqual(len(by_slot["FEET"]), 8)
        self.assertTrue(all(region.visibility == "visible" for region in regions))

    def test_overlap_aware_regions_are_tagged_for_retrieval_weighting(self) -> None:
        regions = build_outfit_regions(self.keypoints, (400, 500))
        by_name = {region.name: region for region in regions}
        self.assertEqual(by_name["body_upper_arm_overlap"].role, "auxiliary")
        self.assertIn("body_hands_overlap", by_name["body_upper_arm_overlap"].tags)
        self.assertIn("long_glove_or_sleeve_overlap", by_name["hands_forearm_left"].tags)
        self.assertIn("legs_feet_overlap", by_name["legs_boot_overlap"].tags)
        self.assertIn("long_boot_focus", by_name["feet_boot_left"].tags)
        self.assertIn("ankle_focus", by_name["feet_left"].tags)
        self.assertIn(
            "expanded_for_full_shoe",
            by_name["feet_full_left"].tags,
        )
        self.assertEqual(by_name["feet_full_left"].role, "diagnostic")
        self.assertIn("may_include_face_hair", by_name["head_tight"].tags)
        self.assertEqual(by_name["head_full_accessory"].role, "diagnostic")
        self.assertIn(
            "expanded_for_full_headwear",
            by_name["head_full_accessory"].tags,
        )

    def test_full_headwear_crop_covers_wide_brims_and_tall_crowns(self) -> None:
        regions = build_outfit_regions(self.keypoints, (400, 500))
        by_name = {region.name: region for region in regions}
        face = self.keypoints[:5]
        face_width = max(point.x for point in face) - min(point.x for point in face)
        full = by_name["head_full_accessory"]
        upper = by_name["head_upper_accessory"]
        self.assertGreater(full.box[2] - full.box[0], face_width * 4.5)
        self.assertLess(full.box[1], upper.box[1])
        self.assertGreater(full.box[2] - full.box[0], upper.box[2] - upper.box[0])

    def test_full_headwear_crop_covers_saved_straw_hat_regression(self) -> None:
        # Pose coordinates from user sample 13.  The old 2.4-face-span crop
        # began at x=58 and clipped the straw hat's outer-left brim.
        keypoints = list(self.keypoints)
        coordinates = (
            (659.6062, 405.7280),
            (673.9458, 369.5177),
            (589.8486, 351.5281),
            (680.4749, 413.7467),
            (452.2987, 381.9752),
            (723.8136, 676.6807),
            (355.1537, 745.3043),
        )
        for index, (x, y) in enumerate(coordinates):
            keypoints[index] = Keypoint(x, y, 0.75)
        regions = build_outfit_regions(keypoints, (1080, 1576))
        full = next(
            region for region in regions if region.name == "head_full_accessory"
        )
        self.assertEqual(full.box[0], 0)
        self.assertEqual(full.box[1], 0)
        self.assertEqual(full.box[2], 1080)
        self.assertEqual(full.role, "diagnostic")
        self.assertIn("source_edge_truncation_risk", full.tags)

    def test_full_headwear_crop_leaves_margin_above_saved_tall_hat(self) -> None:
        # Pose coordinates from user sample 08.  At 1.9 face spans, the crop
        # started at y=186 exactly where the visible top hat begins.
        keypoints = list(self.keypoints)
        coordinates = (
            (306.8515, 286.1193),
            (317.9982, 276.9049),
            (301.7296, 277.4474),
            (345.1970, 273.3921),
            (296.5770, 278.6495),
            (352.9158, 323.4621),
            (295.5496, 325.0706),
        )
        for index, (x, y) in enumerate(coordinates):
            keypoints[index] = Keypoint(x, y, 0.90)
        regions = build_outfit_regions(keypoints, (640, 853))
        full = next(
            region for region in regions if region.name == "head_full_accessory"
        )
        self.assertLessEqual(full.box[1], 175)

    def test_feet_crops_extend_below_ankles_to_include_shoes(self) -> None:
        regions = build_outfit_regions(self.keypoints, (400, 500))
        by_name = {region.name: region for region in regions}
        left_ankle_y = self.keypoints[15].y
        shin = self.keypoints[15].y - self.keypoints[13].y
        self.assertGreaterEqual(
            by_name["feet_full_left"].box[3],
            round(left_ankle_y + shin * 0.50),
        )
        self.assertGreaterEqual(
            by_name["feet_full_context"].box[3],
            round(left_ankle_y + shin * 0.50),
        )

    def test_feet_crops_flag_probable_source_edge_truncation(self) -> None:
        keypoints = list(self.keypoints)
        keypoints[13] = Keypoint(82, 380, 0.9)
        keypoints[14] = Keypoint(118, 380, 0.9)
        keypoints[15] = Keypoint(80, 470, 0.9)
        keypoints[16] = Keypoint(120, 470, 0.9)
        feet = [
            region
            for region in build_outfit_regions(keypoints, (400, 500))
            if region.slot == "FEET"
        ]
        full_feet = [region for region in feet if region.role == "diagnostic"]
        self.assertTrue(
            all(region.visibility == "partially_visible" for region in full_feet)
        )
        self.assertTrue(
            all("source_edge_truncation_risk" in region.tags for region in feet)
        )

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
