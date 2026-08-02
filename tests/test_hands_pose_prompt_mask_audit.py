from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_hands_pose_prompt_mask_audit import (
    pose_prompt_for_crop,
    select_pose_prompt_mask,
    select_reference_paths,
    stratified_examples,
)


class HandsPosePromptMaskAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.keypoints = {
            "left_elbow": {"x": 30.0, "y": 50.0, "score": 0.9},
            "left_wrist": {"x": 70.0, "y": 50.0, "score": 0.9},
        }

    def test_pose_prompt_stays_inside_crop_and_extends_long_armor(self) -> None:
        minimal = pose_prompt_for_crop(
            self.keypoints,
            (10, 20, 90, 80),
            side="left",
            morphology="minimal",
        )
        long_armor = pose_prompt_for_crop(
            self.keypoints,
            (10, 20, 90, 80),
            side="left",
            morphology="long_armor",
        )

        self.assertIsNotNone(minimal)
        self.assertIsNotNone(long_armor)
        assert minimal is not None and long_armor is not None
        self.assertEqual(minimal["input_labels"], [1, 1, 0, 0])
        self.assertEqual(long_armor["input_labels"], [1, 1, 1, 0, 0])
        self.assertLess(long_armor["input_box"][0], minimal["input_box"][0])
        for x, y in long_armor["input_points"]:
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, 80)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, 60)

    def test_mask_selection_rewards_prompt_agreement(self) -> None:
        masks = np.zeros((2, 10, 10), dtype=bool)
        masks[0, 4:7, 4:7] = True
        masks[1, :, :] = True
        points = [[5, 5], [0, 0]]
        labels = [1, 0]

        selected, diagnostics = select_pose_prompt_mask(
            masks,
            np.asarray([0.7, 0.9], dtype=np.float32),
            points,
            labels,
            [3, 3, 8, 8],
        )

        self.assertEqual(selected, 0)
        self.assertEqual(diagnostics[0]["positive_coverage"], 1.0)
        self.assertEqual(diagnostics[0]["negative_leak"], 0.0)
        self.assertGreater(
            diagnostics[0]["selection_score"],
            diagnostics[1]["selection_score"],
        )

    def test_stratified_sample_keeps_unique_glamours_and_appearances(self) -> None:
        names = (
            ("False Nails", "minimal"),
            ("Crystarium Gauntlets", "long_armor"),
            ("Fingerless Gloves", "glove"),
        )
        examples = []
        for morphology_index, (name, _) in enumerate(names):
            for index in range(3):
                value = morphology_index * 10 + index
                examples.append(
                    {
                        "example_id": f"{value}:0",
                        "glamour_id": value,
                        "core_slot_labels": {
                            "HANDS": {
                                "item_name": name,
                                "piece_ids": [100 + value],
                            }
                        },
                    }
                )

        selected = stratified_examples(examples, per_morphology=2, seed=7)

        self.assertEqual(len(selected), 6)
        self.assertEqual(len({row["glamour_id"] for row in selected}), 6)
        self.assertEqual(
            len(
                {
                    tuple(row["core_slot_labels"]["HANDS"]["piece_ids"])
                    for row in selected
                }
            ),
            6,
        )

    def test_reference_selection_requires_expected_piece_and_gender(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            female_path = Path(directory) / "female.png"
            male_path = Path(directory) / "male.png"
            Image.new("RGB", (4, 4), "red").save(female_path)
            Image.new("RGB", (4, 4), "blue").save(male_path)
            examples = [
                {
                    "example_id": "1:0",
                    "model_gender": "female",
                    "core_slot_labels": {
                        "HANDS": {"piece_ids": [10]},
                    },
                }
            ]
            metadata = [
                {
                    "slot": "HANDS",
                    "candidate_crop": "both_hands",
                    "model_gender": "male",
                    "race": "hyur",
                    "local_path": str(male_path),
                    "pieces": [{"piece_id": 10}],
                },
                {
                    "slot": "HANDS",
                    "candidate_crop": "both_hands",
                    "model_gender": "female",
                    "race": "hyur",
                    "local_path": str(female_path),
                    "pieces": [{"piece_id": 10}],
                },
            ]

            selected = select_reference_paths(examples, metadata)

        self.assertEqual(selected, {"1:0": female_path})


if __name__ == "__main__":
    unittest.main()
