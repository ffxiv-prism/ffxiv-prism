from __future__ import annotations

import unittest

from scripts.build_inner_cv_split_manifests import build_fold_assignments


class InnerCvSplitManifestTests(unittest.TestCase):
    def test_each_official_train_group_is_validation_once(self) -> None:
        base = {
            **{str(value): "train" for value in range(100, 160)},
            "999": "validation",
            "1000": "blind_test",
        }
        folds = [
            build_fold_assignments(
                base,
                fold=fold,
                folds=3,
                seed=7,
            )
            for fold in range(3)
        ]

        for glamour_id, role in base.items():
            assigned = [payload[glamour_id] for payload in folds]
            if role == "train":
                self.assertEqual(assigned.count("validation"), 1)
                self.assertEqual(assigned.count("train"), 2)
            else:
                self.assertEqual(assigned, ["blind_test"] * 3)


if __name__ == "__main__":
    unittest.main()
