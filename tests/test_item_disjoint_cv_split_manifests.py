from __future__ import annotations

import unittest

from scripts.build_item_disjoint_cv_split_manifests import (
    body_item_key,
    build_balanced_item_fold_map,
    build_item_disjoint_assignments,
)


class ItemDisjointCvSplitManifestTests(unittest.TestCase):
    def test_shared_body_item_is_never_split_between_train_and_validation(self) -> None:
        base = {str(value): "train" for value in range(12)}
        keys = {str(value): f"item:{value // 3}" for value in range(12)}

        for fold in range(3):
            assignments = build_item_disjoint_assignments(
                base,
                keys,
                fold=fold,
                folds=3,
                seed=11,
            )
            for item_key in set(keys.values()):
                roles = {
                    assignments[glamour_id]
                    for glamour_id, key in keys.items()
                    if key == item_key
                }
                self.assertEqual(len(roles), 1)

    def test_piece_ids_define_body_identity(self) -> None:
        example = {
            "core_slot_labels": {
                "BODY": {"piece_ids": [9, 3], "item_name": "Example"}
            }
        }
        self.assertEqual(body_item_key(example), "pieces:3,9")

    def test_balanced_map_keeps_eligible_case_loads_close(self) -> None:
        weights = {
            "item:a": 8,
            "item:b": 7,
            "item:c": 6,
            "item:d": 5,
            "item:e": 4,
            "item:f": 3,
        }
        fold_map = build_balanced_item_fold_map(
            weights,
            folds=3,
            seed=11,
        )
        loads = [
            sum(
                weight
                for key, weight in weights.items()
                if fold_map[key] == fold
            )
            for fold in range(3)
        ]
        self.assertLessEqual(max(loads) - min(loads), max(weights.values()))

    def test_explicit_fold_map_is_used_for_every_shared_item(self) -> None:
        base = {"1": "train", "2": "train", "3": "train"}
        keys = {"1": "item:a", "2": "item:a", "3": "item:b"}
        fold_map = {"item:a": 1, "item:b": 0}

        assignments = build_item_disjoint_assignments(
            base,
            keys,
            fold=1,
            folds=3,
            seed=11,
            item_fold_map=fold_map,
        )

        self.assertEqual(assignments["1"], "validation")
        self.assertEqual(assignments["2"], "validation")
        self.assertEqual(assignments["3"], "train")


if __name__ == "__main__":
    unittest.main()
