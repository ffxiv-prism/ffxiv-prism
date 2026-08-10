from __future__ import annotations

import argparse
import unittest

from scripts.run_data_pipeline import commands_for_stage


class RunDataPipelineTests(unittest.TestCase):
    def args(self, *, refresh_source: bool) -> argparse.Namespace:
        return argparse.Namespace(
            refresh_source=refresh_source,
            huiji_proxy=None,
            huiji_user_data_dir=None,
        )

    def test_discovery_uses_cache_by_default(self) -> None:
        command = commands_for_stage(
            "discover",
            self.args(refresh_source=False),
        )[0]

        self.assertNotIn("--refresh", command)
        self.assertIn("scripts/discover_eorzea_gearsets_by_model.py", command)

    def test_discovery_refresh_is_explicit(self) -> None:
        command = commands_for_stage(
            "discover",
            self.args(refresh_source=True),
        )[0]

        self.assertEqual(command[-1], "--refresh")

    def test_image_manifest_stage_is_resume_safe_list_cache_build(self) -> None:
        command = commands_for_stage(
            "build-image-manifest",
            self.args(refresh_source=False),
        )[0]

        self.assertIn(
            "scripts/build_eorzea_image_url_manifest_from_list_cache.py",
            command,
        )
        self.assertIn("--resume", command)


if __name__ == "__main__":
    unittest.main()
