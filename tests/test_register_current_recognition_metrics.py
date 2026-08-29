from __future__ import annotations

import unittest

from scripts.register_current_recognition_metrics import (
    build_current_best_selection,
    selected_summary,
)


class RegisterCurrentRecognitionMetricsTests(unittest.TestCase):
    def test_body_requires_frozen_domain_safe_policy(self) -> None:
        summary = {"eligible_case_count": 1, "hits": {"1": 1}}
        payload = {
            "best_safe_policy": {
                "policy": {
                    "max_base_rank": 10,
                    "max_last_branch_rank": 1,
                    "min_last_branch_rank_ratio": 1.0,
                    "max_base_last_branch_rank": 10,
                },
                "domain_regressions": [],
                "summary": summary,
            }
        }
        self.assertIs(selected_summary("BODY", payload), summary)

    def test_hands_rejects_changed_gate_threshold(self) -> None:
        payload = {
            "best_safe_gate": {
                "occupied_multiplier": 0.25,
                "max_base_rank": 5,
                "domain_regressions": [],
                "cutoff_regressions": [],
                "summary": {},
            }
        }
        with self.assertRaisesRegex(ValueError, "max base rank 3"):
            selected_summary("HANDS", payload)

    def test_body_can_register_explicit_safe_baseline(self) -> None:
        summary = {"eligible_case_count": 1, "hits": {"1": 1}}
        payload = {"baseline_summary": summary, "best_safe_policy": None}
        self.assertIs(
            selected_summary("BODY", payload, body_policy="baseline"),
            summary,
        )

    def test_selection_uses_registered_metrics_and_checkpoint_prefix(self) -> None:
        slots = [
            {"slot": slot, "eligible": 100, "top1": 0.2, "top10": 0.8}
            for slot in ("BODY", "HEAD", "HANDS", "LEGS", "FEET")
        ]

        selection = build_current_best_selection(
            release_id="release-1",
            candidate_suffix="20260829-024239",
            body_policy="baseline",
            checkpoint_prefix="refresh_v110",
            slots=slots,
        )

        self.assertEqual(
            selection["checkpoints"]["head_dinov3"],
            "refresh_v110_head_fold01_dinov3_e3.pt",
        )
        self.assertIsNone(selection["checkpoints"]["body_dinov3"])
        self.assertEqual(selection["benchmarks"]["LEGS"]["eligible"], 100)


if __name__ == "__main__":
    unittest.main()
