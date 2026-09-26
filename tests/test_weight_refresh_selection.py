from __future__ import annotations
import unittest
from scripts.weight_refresh_metrics import compare_variants
from scripts.register_weight_refresh import selected_checkpoint_keys


class WeightRefreshSelectionTests(unittest.TestCase):
    def case(self, eid, before, after, fold=1, race="A", gender="female"):
        return dict(example_id=eid, baseline=before, continued=after, fold=fold, race=race, gender=gender)

    def test_promotes_only_measured_improvement(self):
        cases = [self.case("one", 30, 2), self.case("two", 1, 1, fold=2)]
        result = compare_variants(cases, ["baseline", "continued"])
        self.assertEqual(result["selected"], "continued")
        equal = compare_variants([self.case("same", 1, 1)], ["baseline", "continued"])
        self.assertEqual(equal["selected"], "baseline")

    def test_pooled_gain_cannot_hide_a_fold_or_race_loss(self):
        cases = [self.case("gain1", 30, 2), self.case("gain2", 40, 3), self.case("loss", 8, 11, fold=2, race="B")]
        result = compare_variants(cases, ["baseline", "continued"])
        self.assertGreater(result["groups"]["all"]["continued"]["hits"]["10"], result["groups"]["all"]["baseline"]["hits"]["10"])
        self.assertEqual(result["selected"], "baseline")
        self.assertTrue(any(r["group"] == "race:B" for r in result["decisions"]["continued"]["regressions"]))

    def test_top10_gain_cannot_hide_top1_loss(self):
        cases = [self.case("gain", 30, 2), self.case("loss", 1, 3)]
        self.assertEqual(compare_variants(cases, ["baseline", "continued"])["selected"], "baseline")

    def test_missing_hits_and_empty_groups_are_not_success(self):
        self.assertEqual(compare_variants([self.case("missing", None, None)], ["baseline", "continued"])["selected"], "baseline")

    def test_mixed_head_promotion_keeps_the_other_released_branch(self):
        self.assertEqual(selected_checkpoint_keys("HEAD", "dinov3_only"), {"head_dinov3"})
        self.assertEqual(selected_checkpoint_keys("HEAD", "continued"), {"head_dinov2", "head_dinov3"})
        self.assertEqual(selected_checkpoint_keys("BODY", "baseline"), set())
        with self.assertRaises(ValueError):
            selected_checkpoint_keys("BODY", "dinov3_only")


if __name__ == "__main__":
    unittest.main()
