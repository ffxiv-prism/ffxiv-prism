from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ffxiv_prism.boilmaster import fetch_sheet_page, next_after
from ffxiv_prism.jsonl import write_json, write_jsonl_atomic
from scripts import build_eorzea_image_url_manifest_from_list_cache as image_manifest
from scripts import download_cafemaker_equip_slot_categories as equip_slots
from scripts import download_cafemaker_item_names as item_names


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class GameDataSourceTests(unittest.TestCase):
    @patch("ffxiv_prism.boilmaster.urlopen")
    def test_fetch_sheet_page_uses_after_cursor(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = FakeResponse({"rows": []})

        payload = fetch_sheet_page(
            "https://example.test/api/sheet/Item",
            fields=("Name@lang(chs)", "Name@lang(en)"),
            after=42,
            limit=100,
            timeout=5,
            retries=1,
        )

        self.assertEqual(payload, {"rows": []})
        request = mock_urlopen.call_args.args[0]
        self.assertIn("after=42", request.full_url)
        self.assertIn("limit=100", request.full_url)
        self.assertIn("Name%40lang%28chs%29", request.full_url)

    def test_next_after_rejects_stalled_pagination(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not advance"):
            next_after([{"row_id": 42, "fields": {}}], previous_after=42)

    def test_item_normalization_preserves_downstream_contract(self) -> None:
        row = {
            "row_id": 37362,
            "fields": {
                "Name@lang(chs)": "加隆德工具套装",
                "Name@lang(en)": "Ironworks Tool Set",
                "EquipSlotCategory@as(raw)": 7,
                "ItemUICategory@lang(en)": {"fields": {"Name": "Wall-mounted"}},
            },
        }

        self.assertEqual(
            item_names.normalize_item_row(row, source="https://example.test/Item"),
            {
                "game_item_id": 37362,
                "name_zh": "加隆德工具套装",
                "name_en": "Ironworks Tool Set",
                "equip_slot_category_id": 7,
                "item_ui_category_en": "Wall-mounted",
                "source": "https://example.test/Item",
            },
        )

    def test_equip_slot_normalization_preserves_downstream_contract(self) -> None:
        fields = {field: 0 for field in equip_slots.EQUIP_SLOT_FIELDS}
        fields["Body"] = 1
        row = {"row_id": 12, "fields": fields}

        normalized = equip_slots.normalize_row(row, source="https://example.test/EquipSlotCategory")

        self.assertEqual(normalized["equip_slot_category_id"], 12)
        self.assertEqual(normalized["Body"], 1)
        self.assertEqual(normalized["Head"], 0)
        self.assertEqual(normalized["source"], "https://example.test/EquipSlotCategory")

    def test_atomic_jsonl_write_preserves_existing_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.jsonl"
            output.write_text("original\n", encoding="utf-8")

            def broken_rows():
                yield {"row": 1}
                raise RuntimeError("boom")

            with self.assertRaisesRegex(RuntimeError, "boom"):
                write_jsonl_atomic(output, broken_rows())

            self.assertEqual(output.read_text(encoding="utf-8"), "original\n")

    def test_json_write_uses_stable_lf_newlines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "payload.json"

            write_json(output, {"rows": [1, 2]})

            self.assertNotIn(b"\r\n", output.read_bytes())

    def test_item_download_preserves_existing_output_on_empty_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "items.jsonl"
            summary = Path(directory) / "summary.json"
            output.write_text("existing\n", encoding="utf-8")
            argv = [
                "download_cafemaker_item_names.py",
                "--output",
                str(output),
                "--summary",
                str(summary),
                "--retries",
                "1",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(item_names, "fetch_sheet_page", return_value={"rows": []}),
                self.assertRaisesRegex(RuntimeError, "refusing to replace"),
            ):
                item_names.main()

            self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")
            self.assertFalse(summary.exists())

    def test_image_manifest_preserves_output_without_seeds_or_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata"
            metadata.mkdir()
            manifest = metadata / "image_url_manifest_raw.jsonl"
            manifest.write_text("existing\n", encoding="utf-8")
            argv = [
                "build_eorzea_image_url_manifest_from_list_cache.py",
                "--list-cache-dir",
                str(root / "missing-cache"),
                "--resume",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(image_manifest, "EORZEA_METADATA_DIR", metadata),
                self.assertRaisesRegex(RuntimeError, "Refusing to replace"),
            ):
                image_manifest.main()

            self.assertEqual(manifest.read_text(encoding="utf-8"), "existing\n")

    def test_image_candidates_can_be_rebuilt_from_cached_checks(self) -> None:
        url = (
            "https://gearsets.eorzeacollection.com/achievement/certificates/"
            "black-summer-female/hyur-female-body.png"
        )

        candidates = image_manifest.candidate_urls_from_checks(
            [{"source_image_url": url}, {"source_image_url": url}, {"source_image_url": "invalid"}]
        )

        self.assertEqual(candidates, [url])


if __name__ == "__main__":
    unittest.main()
