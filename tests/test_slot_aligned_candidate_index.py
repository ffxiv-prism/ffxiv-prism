from __future__ import annotations

import unittest

from scripts.build_slot_aligned_candidate_index import (
    crop_variants_for_slot,
    expand_slot_aligned_rows,
)


class SlotAlignedCandidateIndexTests(unittest.TestCase):
    def test_hands_include_complete_and_local_candidates(self) -> None:
        rows = [
            {
                "image_row_id": 9,
                "slot": "HANDS",
                "local_path": "hands.png",
            }
        ]

        expanded = expand_slot_aligned_rows(rows)

        self.assertEqual(len(expanded), 3)
        self.assertEqual(
            {row["candidate_crop"] for row in expanded},
            {"both_hands", "left_arm_tight", "right_arm_tight"},
        )
        self.assertEqual(
            {row["source_image_row_id"] for row in expanded},
            {9},
        )

    def test_multiscale_hands_adds_forearm_context_candidates(self) -> None:
        variants = crop_variants_for_slot(
            "HANDS",
            hands_crop_policy="multiscale",
        )

        self.assertEqual(
            [name for name, _ in variants],
            [
                "both_hands",
                "left_arm_tight",
                "right_arm_tight",
                "left_forearm_context",
                "right_forearm_context",
                "hands_upper_context",
            ],
        )
        self.assertGreater(variants[3][1][2], variants[1][1][2])


    def test_body_crop_removes_lower_mannequin(self) -> None:
        variants = crop_variants_for_slot("BODY")

        self.assertEqual(variants[0][0], "upper_body")
        self.assertLess(variants[0][1][3], 1.0)

    def test_multiscale_body_adds_tighter_torso_candidate(self) -> None:
        rows = [
            {
                "image_row_id": 3,
                "slot": "BODY",
                "local_path": "body.png",
            }
        ]

        expanded = expand_slot_aligned_rows(
            rows,
            body_crop_policy="multiscale",
        )

        self.assertEqual(
            [row["candidate_crop"] for row in expanded],
            ["upper_body", "torso_core"],
        )
        upper = expanded[0]["candidate_crop_box_normalized"]
        torso = expanded[1]["candidate_crop_box_normalized"]
        self.assertGreater(torso[1], upper[1])
        self.assertLess(torso[3], upper[3])

    def test_multiscale_legs_adds_side_and_overlap_candidates(self) -> None:
        variants = crop_variants_for_slot(
            "LEGS",
            legs_crop_policy="multiscale",
        )

        self.assertEqual(
            [name for name, _ in variants],
            ["full", "legs_left", "legs_right", "legs_thighs", "legs_boot_overlap"],
        )
        self.assertLess(variants[3][1][3], variants[0][1][3])

    def test_multiscale_feet_adds_side_and_boot_candidates(self) -> None:
        variants = crop_variants_for_slot(
            "FEET",
            feet_crop_policy="multiscale",
        )

        self.assertEqual(
            [name for name, _ in variants],
            ["full", "feet_left", "feet_right", "feet_boot_left", "feet_boot_right"],
        )
        self.assertGreater(variants[3][1][2], variants[1][1][2])


    def test_multiscale_head_adds_hat_and_face_candidates(self) -> None:
        variants = crop_variants_for_slot(
            "HEAD",
            head_crop_policy="multiscale",
        )

        self.assertEqual(
            [name for name, _ in variants],
            ["full", "head_upper_accessory", "head_face_accessory"],
        )
        self.assertLess(variants[1][1][3], variants[0][1][3])


if __name__ == "__main__":
    unittest.main()
