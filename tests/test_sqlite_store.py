from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.dataset.sqlite_store import build_eorzea_sqlite


class SqliteStoreTests(unittest.TestCase):
    def test_build_sqlite_links_exact_and_occupied_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gearsets_path = root / "gearsets.jsonl"
            gear_pieces_path = root / "pieces.jsonl"
            images_path = root / "images.jsonl"
            db_path = root / "dataset.sqlite"
            summary_path = root / "summary.json"

            write_jsonl(
                gearsets_path,
                [
                    {
                        "slug": "test-set",
                        "title": "Test Set",
                        "source_page_url": "https://example.test/gearset/test-set",
                        "dyeable": True,
                        "present_slots": ["BODY"],
                        "missing_slots": ["HANDS"],
                        "roles": ["Tanks"],
                        "model_genders": ["Female"],
                    }
                ],
            )
            write_jsonl(
                gear_pieces_path,
                [
                    {
                        "gearset_slug": "test-set",
                        "slot": "BODY",
                        "name": "Test Coat",
                        "name_en": "Test Coat",
                        "source_item_url": "https://example.test/item/1",
                        "occupied_slots": ["BODY", "HANDS"],
                        "blocked_slots": ["HANDS"],
                    }
                ],
            )
            write_jsonl(
                images_path,
                [
                    image_row("https://example.test/body.png", "BODY"),
                    image_row("https://example.test/hands.png", "HANDS"),
                    image_row("https://example.test/front.png", None),
                ],
            )

            summary = build_eorzea_sqlite(
                db_path=db_path,
                gearsets_path=gearsets_path,
                gear_pieces_path=gear_pieces_path,
                images_path=images_path,
                summary_path=summary_path,
            )

            self.assertEqual(summary["gearsets"], 1)
            self.assertEqual(summary["gear_pieces"], 1)
            self.assertEqual(summary["images"], 3)
            self.assertEqual(summary["gear_piece_image_links"], 2)
            conn = sqlite3.connect(db_path)
            try:
                link_types = [
                    row[0]
                    for row in conn.execute("SELECT link_type FROM gear_piece_images ORDER BY link_type").fetchall()
                ]
            finally:
                conn.close()
            self.assertEqual(link_types, ["exact_slot", "occupied_slot"])

    def test_build_sqlite_normalizes_gearset_slug_before_linking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gearsets_path = root / "gearsets.jsonl"
            gear_pieces_path = root / "pieces.jsonl"
            images_path = root / "images.jsonl"
            db_path = root / "dataset.sqlite"
            summary_path = root / "summary.json"

            write_jsonl(
                gearsets_path,
                [
                    {
                        "slug": "no-2-type-b",
                        "title": "No.2 Type B",
                        "present_slots": ["HANDS"],
                        "missing_slots": [],
                    }
                ],
            )
            write_jsonl(
                gear_pieces_path,
                [
                    {
                        "gearset_slug": "no-2-type-b",
                        "slot": "HANDS",
                        "name": "No.2 Type B Gloves",
                        "source_item_url": "https://example.test/item/gloves",
                        "occupied_slots": ["HANDS"],
                    }
                ],
            )
            image = image_row("https://example.test/gloves.png", "HANDS")
            image["gearset_slug"] = "no.2-type-b"
            write_jsonl(images_path, [image])

            summary = build_eorzea_sqlite(
                db_path=db_path,
                gearsets_path=gearsets_path,
                gear_pieces_path=gear_pieces_path,
                images_path=images_path,
                summary_path=summary_path,
            )

            self.assertEqual(summary["gear_piece_image_links"], 1)
            self.assertEqual(summary["slot_images_without_gearset"], 0)
            self.assertEqual(summary["unlinked_slot_images"], 0)


def image_row(url: str, slot: str | None) -> dict[str, object]:
    return {
        "gearset_slug": "test-set",
        "source_image_url": url,
        "source_page_url": "https://example.test/gearset/test-set",
        "local_path": f"C:/tmp/{Path(url).name}",
        "sha256": url,
        "byte_size": 100,
        "actual_byte_size": 100,
        "width": 340,
        "height": 690,
        "image_format": "png",
        "image_kind": "gearset_static",
        "image_part": slot.lower() if slot else "front",
        "slot": slot,
        "model_gender": "female",
        "race": "hyur",
        "dye_state": "normal",
        "qa_status": "ready",
        "quality_flags": [],
        "file_exists": True,
        "readable_image": True,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
