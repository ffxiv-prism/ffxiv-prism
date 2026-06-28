from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_glamour_blind import consume_blind_groups


class BlindEvaluationTests(unittest.TestCase):
    def test_consumed_blind_groups_cannot_remain_blind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split.json"
            manifest.write_text(
                json.dumps(
                    {
                        "assignments": {
                            "1": "blind_test",
                            "2": "train",
                        }
                    }
                ),
                encoding="utf-8",
            )
            consume_blind_groups(
                manifest,
                {"1"},
                Path("report.json"),
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(payload["assignments"]["1"], "confirmation_test")
        self.assertEqual(payload["counts"]["confirmation_test"], 1)
        self.assertEqual(len(payload["blind_evaluations"]), 1)


if __name__ == "__main__":
    unittest.main()
