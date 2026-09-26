from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.run_lifecycle_pipeline as lifecycle


class LifecyclePipelineTests(unittest.TestCase):
    def test_local_sample_plan_rebuilds_derived_assets_without_network(self) -> None:
        args = lifecycle.parse_args(
            [
                "--workflow",
                "sample-library",
                "--source-mode",
                "local",
                "--candidate-mode",
                "check",
            ]
        )

        stages = lifecycle.sample_stages(args, "20260828-010203")
        names = [stage.name for stage in stages]

        self.assertNotIn("discover", names)
        self.assertNotIn("download-images", names)
        self.assertEqual(
            names[:3], ["build-index", "validate-images", "build-sqlite"]
        )
        candidate = next(stage for stage in stages if stage.name == "candidate-indexes")
        self.assertEqual(candidate.name, "candidate-indexes")
        self.assertIn("--dry-run", candidate.command)

    def test_refresh_sample_plan_is_complete_and_versioned(self) -> None:
        args = lifecycle.parse_args(
            [
                "--workflow",
                "sample-library",
                "--source-mode",
                "refresh",
                "--candidate-mode",
                "extend",
                "--candidate-suffix",
                "20260828",
            ]
        )

        stages = lifecycle.sample_stages(args, "unused")
        names = [stage.name for stage in stages]

        self.assertLess(names.index("discover"), names.index("collect-details"))
        self.assertLess(
            names.index("build-image-manifest"), names.index("download-images")
        )
        self.assertLess(names.index("download-images"), names.index("build-index"))
        self.assertLess(names.index("build-sqlite"), names.index("build-same-model"))
        self.assertLess(names.index("candidate-indexes"), names.index("cafemaker-zh"))
        self.assertLess(names.index("download-images"), names.index("collect-details"))
        self.assertLess(names.index("sample-eligibility"), names.index("candidate-indexes"))
        self.assertIn("--refresh-source", stages[0].command)
        candidate = next(stage for stage in stages if stage.name == "candidate-indexes")
        self.assertNotIn("--dry-run", candidate.command)
        self.assertIn("--skip-empty", candidate.command)
        self.assertIn("--resume", candidate.command)
        self.assertIn("--activate", candidate.command)
        for stage in stages[names.index("candidate-indexes") + 1:]:
            self.assertTrue(stage.advisory, stage.name)

    def test_cached_plan_reuses_cafemaker_raw_tables(self) -> None:
        args = lifecycle.parse_args(
            [
                "--workflow",
                "sample-library",
                "--source-mode",
                "cached",
                "--candidate-mode",
                "skip",
            ]
        )

        names = [stage.name for stage in lifecycle.sample_stages(args, "unused")]

        self.assertIn("cafemaker-enrich-local", names)
        self.assertNotIn("cafemaker-zh", names)
        self.assertNotIn("cafemaker-models", names)

    def test_refresh_plan_can_wait_for_manual_browser_verification(self) -> None:
        args = lifecycle.parse_args(
            [
                "--workflow",
                "sample-library",
                "--source-mode",
                "refresh",
                "--browser-verification",
                "--browser-profile-dir",
                "var/cache/test-browser",
                "--browser-wait-seconds",
                "45",
            ]
        )

        stages = lifecycle.sample_stages(args, "20260828-010203")
        browser_stages = [
            stage.command
            for stage in stages
            if stage.name
            in {"discover", "collect-details", "build-image-manifest", "download-images"}
        ]

        self.assertEqual(len(browser_stages), 4)
        for command in browser_stages:
            self.assertIn("--browser-verification", command)
            self.assertIn("var/cache/test-browser", command)
            self.assertIn("45.0", command)

    def test_release_plan_checks_profile_and_images(self) -> None:
        args = lifecycle.parse_args(
            ["--workflow", "model-release", "--release-mode", "check"]
        )

        names = [stage.name for stage in lifecycle.release_stages(args)]

        self.assertEqual(
            names, ["tests", "profile-check", "image-coverage-check"]
        )
        self.assertIn("--check", lifecycle.release_asset_stage(args).command)

    def test_metrics_gate_rejects_stale_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "metrics.json"
            metrics.write_text(
                json.dumps(
                    {
                        "metric_status": "refresh_required",
                        "target_achieved_all_slots": False,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(lifecycle, "PROJECT_ROOT", root),
                patch.object(lifecycle, "METRICS_PATH", Path("metrics.json")),
            ):
                ok, problems, _payload = lifecycle.metrics_gate(
                    allow_below_target=False
                )

        self.assertFalse(ok)
        self.assertEqual(len(problems), 2)

    def test_equivalence_group_count_deduplicates_symmetric_members(self) -> None:
        self.assertEqual(
            lifecycle.count_equivalence_groups(
                {"1": [2, 3], "2": [1, 3], "3": [1, 2], "9": [10]}
            ),
            2,
        )

    def test_passed_stage_can_be_reused_when_only_resume_flag_is_added(self) -> None:
        stage = lifecycle.Stage(
            "candidate-indexes",
            "candidate",
            ("python", "script.py", "--resume"),
            "sample-library",
        )
        previous = {
            "status": "passed",
            "command": ["python", "script.py"],
        }

        result = lifecycle.reused_stage_result(stage, previous)

        self.assertIsNotNone(result)
        self.assertEqual(result.status, "reused")

    def test_legacy_report_candidate_suffix_is_read_from_command(self) -> None:
        suffix = lifecycle.candidate_suffix_from_report(
            {
                "run_id": "fallback",
                "stages": [
                    {
                        "name": "candidate-indexes",
                        "command": ["python", "script.py", "--suffix", "versioned"],
                    }
                ],
            }
        )

        self.assertEqual(suffix, "versioned")


if __name__ == "__main__":
    unittest.main()
