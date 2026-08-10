from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_glamour_item_queue import (
    extract_listing_links,
    load_excluded_ids,
    select_balanced_candidates,
    write_queue_shards,
)


class GlamourItemQueueTests(unittest.TestCase):
    def test_extracts_glamours_and_same_item_pagination(self) -> None:
        html = """
        <a href="/glamour/123/example">Example</a>
        <a href="https://ffxiv.eorzeacollection.com/glamour/456/other">Other</a>
        <a href="/glamours/body/99?page=2">Next</a>
        <a href="/glamours/body/100?page=2">Wrong item</a>
        """

        ids, pages = extract_listing_links(
            html,
            "https://ffxiv.eorzeacollection.com/glamours/body/99",
            99,
        )

        self.assertEqual(ids, {123, 456})
        self.assertEqual(
            pages,
            ["https://ffxiv.eorzeacollection.com/glamours/body/99?page=2"],
        )

    def test_balanced_selection_caps_each_item(self) -> None:
        candidates = [9, 8, 7, 6, 5, 4]
        sources = {
            9: {1},
            8: {1},
            7: {1},
            6: {2},
            5: {2},
            4: {2},
        }

        selected = select_balanced_candidates(
            candidates,
            sources,
            [1, 2],
            limit=10,
            limit_per_item=2,
        )

        self.assertEqual(selected, [9, 6, 8, 5])

    def test_loads_numeric_ids_from_multiple_existing_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.txt"
            second = Path(temp_dir) / "second.txt"
            first.write_text("123\nnot-an-id\n456\n", encoding="utf-8")
            second.write_text("456\n789\n", encoding="utf-8")

            excluded = load_excluded_ids(
                [first, Path(temp_dir) / "missing.txt", second]
            )

        self.assertEqual(excluded, {123, 456, 789})

    def test_writes_bounded_numbered_shards(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "queue.txt"
            paths = write_queue_shards(
                [9, 8, 7, 6, 5],
                output,
                shard_size=2,
            )

            self.assertEqual(
                [path.name for path in paths],
                ["queue.part001.txt", "queue.part002.txt", "queue.part003.txt"],
            )
            self.assertEqual(paths[0].read_text(encoding="utf-8"), "9\n8\n")
            self.assertEqual(paths[2].read_text(encoding="utf-8"), "5\n")


if __name__ == "__main__":
    unittest.main()
