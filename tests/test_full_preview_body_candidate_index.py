from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.build_full_preview_body_candidate_index import (
    filter_existing_rows,
    load_full_preview_body_rows,
)


class FullPreviewBodyCandidateIndexTests(unittest.TestCase):
    def test_links_front_preview_to_same_gearset_body_piece(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "test.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE images (
                    image_row_id INTEGER PRIMARY KEY,
                    gearset_slug TEXT,
                    source_image_url TEXT,
                    local_path TEXT,
                    image_part TEXT,
                    model_gender TEXT,
                    race TEXT,
                    dye_state TEXT,
                    width INTEGER,
                    height INTEGER,
                    qa_status TEXT
                );
                CREATE TABLE gear_pieces (
                    piece_id INTEGER PRIMARY KEY,
                    gearset_slug TEXT,
                    slot TEXT,
                    name_en TEXT,
                    name_zh TEXT
                );
                """
            )
            connection.execute(
                """
                INSERT INTO images VALUES
                (1, 'set-a', 'https://example.test/front.png', 'front.png',
                 'front', 'male', 'hrothgar', 'normal', 340, 690, 'ready')
                """
            )
            connection.executemany(
                "INSERT INTO gear_pieces VALUES (?, ?, ?, ?, ?)",
                [
                    (10, "set-a", "BODY", "Body A", None),
                    (11, "set-a", "HEAD", "Head A", None),
                ],
            )
            connection.commit()
            connection.close()

            rows = load_full_preview_body_rows(
                database,
                races=["hrothgar"],
                genders=["male"],
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["slot"], "BODY")
        self.assertEqual(rows[0]["candidate_crop_box_normalized"], [0.0, 0.0, 1.0, 0.72])
        self.assertEqual([piece["piece_id"] for piece in rows[0]["pieces"]], [10])
        self.assertEqual(
            rows[0]["pieces"][0]["link_type"],
            "same_gearset_exact_slot",
        )

    def test_filters_race_before_returning_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "test.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE images (
                    image_row_id INTEGER PRIMARY KEY,
                    gearset_slug TEXT,
                    source_image_url TEXT,
                    local_path TEXT,
                    image_part TEXT,
                    model_gender TEXT,
                    race TEXT,
                    dye_state TEXT,
                    width INTEGER,
                    height INTEGER,
                    qa_status TEXT
                );
                CREATE TABLE gear_pieces (
                    piece_id INTEGER PRIMARY KEY,
                    gearset_slug TEXT,
                    slot TEXT,
                    name_en TEXT,
                    name_zh TEXT
                );
                INSERT INTO images VALUES
                (1, 'set-a', 'https://example.test/front.png', 'front.png',
                 'front', 'female', 'viera', 'normal', 340, 690, 'ready');
                INSERT INTO gear_pieces VALUES
                (10, 'set-a', 'BODY', 'Body A', NULL);
                """
            )
            connection.commit()
            connection.close()

            rows = load_full_preview_body_rows(database, races=["hyur"])

        self.assertEqual(rows, [])

    def test_filters_stale_database_paths_before_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing_path = Path(directory) / "existing.png"
            existing_path.write_bytes(b"not decoded by this helper")
            rows = [
                {"image_row_id": 1, "local_path": str(existing_path)},
                {
                    "image_row_id": 2,
                    "local_path": str(Path(directory) / "missing.png"),
                },
            ]

            existing, missing = filter_existing_rows(rows)

        self.assertEqual([row["image_row_id"] for row in existing], [1])
        self.assertEqual([row["image_row_id"] for row in missing], [2])


if __name__ == "__main__":
    unittest.main()
