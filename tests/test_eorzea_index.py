from __future__ import annotations

import unittest

from ffxiv_prism.dataset.eorzea_index import dedupe_by_hash_and_url


class EorzeaIndexDedupeTests(unittest.TestCase):
    def test_newest_record_replaces_stale_bytes_for_same_source_url(self) -> None:
        rows = [
            {
                "source_image_url": "https://example.test/gear/body.png",
                "local_path": "data/body.png",
                "sha256": "old",
                "byte_size": 100,
            },
            {
                "source_image_url": "https://example.test/gear/body.png",
                "local_path": "data/body.png",
                "sha256": "new",
                "byte_size": 120,
            },
        ]

        result = dedupe_by_hash_and_url(rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sha256"], "new")
        self.assertEqual(result[0]["byte_size"], 120)

    def test_same_bytes_at_distinct_source_urls_remain_distinct_samples(self) -> None:
        rows = [
            {
                "source_image_url": "https://example.test/a.png",
                "local_path": "data/a.png",
                "sha256": "same",
            },
            {
                "source_image_url": "https://example.test/b.png",
                "local_path": "data/b.png",
                "sha256": "same",
            },
        ]

        result = dedupe_by_hash_and_url(rows)

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
