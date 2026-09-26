from __future__ import annotations

import math
import unittest
from dataclasses import replace

from ffxiv_prism.matching.inference import crop_weight, current_best_configs
from ffxiv_prism.vision.outfit_regions import (
    Keypoint,
    build_lower_face_regions,
    build_outfit_regions,
)
from scripts.build_slot_aligned_candidate_index import crop_variants_for_slot


class HeadLocalCropTests(unittest.TestCase):
    def setUp(self):
        self.face = [
            Keypoint(x, y, 0.95)
            for x, y in ((200, 210), (185, 200), (215, 200), (164, 210), (236, 210))
        ]

    def test_covers_jaw_below_nose_and_both_ears(self):
        lower, left, right = build_lower_face_regions(self.face, (500, 500))
        for crop, point in (
            (lower, (200, 266)),
            (left, (164, 244)),
            (right, (236, 244)),
        ):
            with self.subTest(crop=crop.name):
                self.assertLessEqual(crop.box[0], point[0])
                self.assertLessEqual(crop.box[1], point[1])
                self.assertGreaterEqual(crop.box[2], point[0])
                self.assertGreaterEqual(crop.box[3], point[1])
                self.assertEqual(crop.visibility, "visible")

    def test_saved_tilted_face_includes_low_microphone(self):
        # Saved geometry only: the target label/ID never enters crop selection.
        face = [
            Keypoint(*p)
            for p in (
                (649.684, 384.941, 0.995),
                (675.785, 342.719, 0.998),
                (611.426, 358.308, 0.975),
                (733.742, 353.333, 0.940),
                (578.791, 390.849, 0.984),
            )
        ]
        lower = build_lower_face_regions(face, (2402, 2472))[0]
        # Visible mic capsule and boom are below the old face crop's y=451.
        self.assertLessEqual(lower.box[0], 620)
        self.assertGreaterEqual(lower.box[2], 680)
        self.assertLessEqual(lower.box[1], 470)
        self.assertGreaterEqual(lower.box[3], 515)
        self.assertLess(lower.box[2], 900)  # excludes the text at the right

    def test_geometry_does_not_depend_on_canvas_padding_or_shoulders(self):
        compact = build_lower_face_regions(self.face, (500, 500))
        padded = build_lower_face_regions(
            self.face + [Keypoint(9000, 9000, 0.99)] * 12, (10000, 10000)
        )
        self.assertEqual([r.box for r in compact], [r.box for r in padded])

    def test_translation_and_scaling(self):
        original = build_lower_face_regions(self.face, (500, 500))
        transformed = [replace(p, x=2 * p.x + 40, y=2 * p.y + 60) for p in self.face]
        shifted = build_lower_face_regions(transformed, (1100, 1100))
        for before, after in zip(original, shifted, strict=True):
            for i, (a, b) in enumerate(zip(before.box, after.box, strict=True)):
                self.assertAlmostEqual(b, 2 * a + (40 if i % 2 == 0 else 60), delta=1)

    def test_rotated_face_retains_projected_jaw(self):
        for degrees in (-60, -30, 0, 30, 60):
            angle = math.radians(degrees)
            c, s = math.cos(angle), math.sin(angle)

            def rotate(x, y):
                return (
                    200 + c * (x - 200) - s * (y - 200),
                    200 + s * (x - 200) + c * (y - 200),
                )

            face = [Keypoint(*rotate(p.x, p.y), p.score) for p in self.face]
            crop = build_lower_face_regions(face, (500, 500))[0]
            x, y = rotate(200, 266)
            with self.subTest(degrees=degrees):
                self.assertLessEqual(crop.box[0], x)
                self.assertLessEqual(crop.box[1], y)
                self.assertGreaterEqual(crop.box[2], x)
                self.assertGreaterEqual(crop.box[3], y)

    def test_mirroring_preserves_coverage(self):
        original = build_lower_face_regions(self.face, (500, 500))
        mirrored = build_lower_face_regions(
            [replace(p, x=500 - p.x) for p in self.face], (500, 500)
        )
        for a, b in zip(original, mirrored, strict=True):
            self.assertEqual(
                b.box, (500 - a.box[2], a.box[1], 500 - a.box[0], a.box[3])
            )

    def test_weak_eye_or_nose_disables_all_local_views(self):
        for index in (0, 1, 2):
            face = list(self.face)
            face[index] = replace(face[index], score=0.1)
            for crop in build_lower_face_regions(face, (500, 500)):
                self.assertEqual(crop.visibility, "pose_uncertain")
                self.assertIn("unreliable_face_landmarks", crop.tags)

    def test_high_confidence_profile_geometry_is_still_uncertain(self):
        narrow_eyes = list(self.face)
        narrow_eyes[1] = Keypoint(198, 200, 0.99)
        narrow_eyes[2] = Keypoint(202, 200, 0.99)
        lateral_nose = list(self.face)
        lateral_nose[0] = Keypoint(245, 210, 0.99)
        for face in (narrow_eyes, lateral_nose):
            for crop in build_lower_face_regions(face, (500, 500)):
                self.assertEqual(crop.visibility, "pose_uncertain")
                self.assertIn("foreshortened_face_geometry", crop.tags)

    def test_weak_ear_disables_only_its_view_and_cannot_inflate_scale(self):
        face = list(self.face)
        face[3] = Keypoint(9000, 9000, 0.1)
        lower, left, right = build_lower_face_regions(face, (500, 500))
        self.assertEqual(left.visibility, "pose_uncertain")
        self.assertEqual(
            lower.box, build_lower_face_regions(self.face, (500, 500))[0].box
        )
        self.assertEqual(right.visibility, "visible")

    def test_degenerate_eye_line_and_source_truncation_are_flagged(self):
        face = list(self.face)
        face[1] = face[2]
        self.assertTrue(
            all(
                r.visibility == "pose_uncertain"
                for r in build_lower_face_regions(face, (500, 500))
            )
        )
        near_bottom = [replace(p, y=p.y + 270) for p in self.face]
        for crop in build_lower_face_regions(near_bottom, (500, 500)):
            self.assertNotEqual(crop.visibility, "visible")
            self.assertIn("source_edge_truncation_risk", crop.tags)
            self.assertLessEqual(crop.box[3], 500)

    def test_new_views_are_zero_weight_and_keep_promoted_query(self):
        points = self.face + [Keypoint(200, 350, 0.95)] * 12
        by_name = {r.name: r for r in build_outfit_regions(points, (500, 500))}
        for region in build_lower_face_regions(self.face, (500, 500)):
            self.assertEqual(by_name[region.name], region)
            self.assertEqual(region.role, "diagnostic")
            self.assertEqual(crop_weight(vars(region)), 0)
        self.assertEqual(
            current_best_configs()["HEAD"].query_crop_names, ("head_upper_accessory",)
        )

    def test_candidate_policy_is_explicit_and_does_not_replace_multiscale(self):
        variants = crop_variants_for_slot("HEAD", head_crop_policy="lower-face")
        self.assertEqual(
            [n for n, _ in variants],
            ["head_lower_face", "head_ear_left", "head_ear_right"],
        )
        for _, (x1, y1, x2, y2) in variants:
            self.assertTrue(0 <= x1 < x2 <= 1)
            self.assertTrue(0 <= y1 < y2 <= 1)
        self.assertEqual(crop_variants_for_slot("HEAD"), (("full", (0, 0, 1, 1)),))
        self.assertEqual(
            [
                n
                for n, _ in crop_variants_for_slot(
                    "HEAD", head_crop_policy="multiscale"
                )
            ],
            ["full", "head_upper_accessory", "head_face_accessory"],
        )


if __name__ == "__main__":
    unittest.main()
