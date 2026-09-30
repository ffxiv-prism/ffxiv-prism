from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ffxiv_prism.matching.report_export import write_report_bundle


class ReportExportTests(unittest.TestCase):
    def test_bundle_renders_original_html_and_registers_sibling_png(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            payload = {"slots": [], "artifacts": {}, "warnings": []}

            def render(html_path):
                html = html_path.read_text(encoding="utf-8")
                self.assertIn("FFXIV Prism 识别结果", html)
                self.assertIn("color-scheme: dark", html)
                html_path.with_suffix(".png").write_bytes(b"test screenshot")
                return html_path.with_suffix(".png")

            with patch("ffxiv_prism.matching.report_export.render_report_png", side_effect=render):
                png = write_report_bundle(output, payload)
            self.assertEqual(png, output / "report.png")
            result = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(result["artifacts"]["report_png"]).name, "report.png")
            self.assertEqual(result["warnings"], [])

    def test_export_failure_preserves_readable_reports_without_stale_png(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "report.png").write_bytes(b"previous result")
            payload = {"slots": [], "artifacts": {"report_png": "old.png"}, "warnings": []}
            with patch("ffxiv_prism.matching.report_export.render_report_png", side_effect=RuntimeError("browser unavailable")):
                self.assertIsNone(write_report_bundle(output, payload))
            self.assertFalse((output / "report.png").exists())
            result = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertNotIn("report_png", result["artifacts"])
            self.assertEqual(result["warnings"], [])
            self.assertNotIn("browser unavailable", (output / "report.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
