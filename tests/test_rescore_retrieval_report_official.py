from __future__ import annotations

import unittest

from scripts.rescore_retrieval_report_official import (
    rescore_case,
    summarize,
)


class OfficialReportRescoreTests(unittest.TestCase):
    def test_rescores_only_against_explicit_piece_ids(self) -> None:
        case = {
            "example_id": "1:0",
            "glamour_id": 1,
            "slot": "BODY",
            "expected_name": "Example",
            "status": "eligible",
            "results": [
                {"piece_id": 99, "name_en": "Normalized Example"},
                {"piece_id": 11, "name_en": "Official Same Model"},
            ],
        }
        examples = {
            "1:0": {
                "core_slot_labels": {
                    "BODY": {
                        "piece_ids": [10, 11],
                    }
                }
            }
        }
        rescored = rescore_case(case, examples, 10)
        self.assertEqual(rescored["rank"], 2)
        self.assertFalse(rescored["hit_at_1"])
        self.assertTrue(rescored["hit_at_k"])
        summary = summarize([rescored], 10)
        self.assertEqual(summary["hits_at_1"], 0)
        self.assertEqual(summary["hits_at_k"], 1)


if __name__ == "__main__":
    unittest.main()
