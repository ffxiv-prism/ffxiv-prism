from __future__ import annotations

import unittest

from scripts.evaluate_cached_query_retrieval import build_quality_by_id


class CachedQueryRetrievalTests(unittest.TestCase):
    def test_build_quality_by_id_supports_per_slot_eligibility(self) -> None:
        examples = [{"example_id": "a"}, {"example_id": "b"}]
        quality = build_quality_by_id(
            examples,
            slot="HANDS",
            eligibility={"a": {"HANDS"}, "b": {"BODY"}},
        )
        self.assertEqual(quality, {"a": True, "b": False})

    def test_build_quality_by_id_requires_exactly_one_source(self) -> None:
        with self.assertRaises(ValueError):
            build_quality_by_id([], slot="HANDS")
        with self.assertRaises(ValueError):
            build_quality_by_id(
                [],
                slot="HANDS",
                quality_rows=[],
                eligibility={},
            )


if __name__ == "__main__":
    unittest.main()
