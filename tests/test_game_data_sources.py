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
from scripts import download_cafemaker_item_models as item_models
from scripts.filter_cafemaker_item_names_for_gear_pieces import filter_mapping
from ffxiv_prism.matching.item_dyes import load_item_dye_counts


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
                "dye_count": None,
                "source": "https://example.test/Item",
            },
        )

    def test_dye_count_normalization_preserves_zero_and_unknowns(self) -> None:
        for raw, expected in [(0, 0), (1, 1), (2, 2), ("2", 2),
                              (None, None), (-1, None), ("bad", None),
                              (True, None), (1.5, None)]:
            with self.subTest(raw=raw):
                row = item_names.normalize_item_row({"row_id": 1, "fields": {"DyeCount": raw}})
                self.assertEqual(row["dye_count"], expected)

    def test_dye_download_reaches_report_and_filtered_mapping_without_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cafemaker_item_names_raw.jsonl"
            summary = Path(directory) / "summary.json"
            rows = [{"row_id": number, "fields": {
                "Name@lang(en)": name, "Name@lang(chs)": name,
                "EquipSlotCategory@as(raw)": 3, "DyeCount": count,
            }} for number, name, count in [(1, "Fixed", 0), (2, "Dual", 2), (3, "Unknown", None)]]
            pages = [{"rows": rows[:2], "schema": "schema-test", "version": "version-test"},
                     {"rows": rows[2:]}, {"rows": []}]
            with patch.object(sys, "argv", ["download", "--output", str(output), "--summary", str(summary)]), \
                 patch.object(item_names, "fetch_sheet_page", side_effect=pages) as fetch:
                self.assertEqual(item_names.main(), 0)
            self.assertIn("DyeCount", fetch.call_args_list[0].kwargs["fields"])
            self.assertEqual([call.kwargs["after"] for call in fetch.call_args_list], [None, 2, 3])
            with patch("ffxiv_prism.matching.item_dyes.GAME_DATA_RAW_DIR", Path(directory)):
                self.assertEqual(load_item_dye_counts(), ({1: 0, 2: 2}, {"Fixed": 0, "Dual": 2}))
            filtered, _ = filter_mapping(item_names.read_jsonl(output), {"fixed", "dual", "unknown"})
            self.assertEqual({r["game_item_id"]: r["dye_count"] for r in filtered}, {1: 0, 2: 2, 3: None})
            metadata = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(metadata["wearable_rows_with_dye_count"], 2)
            self.assertEqual(metadata["wearable_rows_missing_dye_count"], 1)
            self.assertEqual(metadata["dyeable_wearable_rows"], 1)
            self.assertEqual(metadata["game_version"], "version-test")

    def test_missing_dye_schema_or_network_failure_preserves_old_output(self) -> None:
        for failure in ("schema", "network"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "items.jsonl"
                summary = Path(directory) / "summary.json"
                output.write_text("original", encoding="utf-8")
                summary.write_text("original-summary", encoding="utf-8")
                row = {"row_id": 1, "fields": {"EquipSlotCategory@as(raw)": 3}}
                responses = [{"rows": [row]}, {"rows": []} if failure == "schema" else RuntimeError("offline")]
                with patch.object(sys, "argv", ["download", "--output", str(output), "--summary", str(summary)]), \
                     patch.object(item_names, "fetch_sheet_page", side_effect=responses), \
                     self.assertRaises(RuntimeError):
                    item_names.main()
                self.assertEqual(output.read_text(encoding="utf-8"), "original")
                self.assertEqual(summary.read_text(encoding="utf-8"), "original-summary")

    def test_resume_rejects_pre_dye_partial_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "items.jsonl"
            output.with_suffix(".jsonl.tmp").write_text('{"game_item_id": 1}\n', encoding="utf-8")
            with patch.object(sys, "argv", ["download", "--output", str(output), "--resume"]), \
                 patch.object(item_names, "fetch_sheet_page") as fetch, \
                 self.assertRaisesRegex(ValueError, "without --resume"):
                item_names.main()
            fetch.assert_not_called()

    def test_equip_slot_normalization_preserves_downstream_contract(self) -> None:
        fields = {field: 0 for field in equip_slots.EQUIP_SLOT_FIELDS}
        fields["Body"] = 1
        row = {"row_id": 12, "fields": fields}

        normalized = equip_slots.normalize_row(row, source="https://example.test/EquipSlotCategory")

        self.assertEqual(normalized["equip_slot_category_id"], 12)
        self.assertEqual(normalized["Body"], 1)
        self.assertEqual(normalized["Head"], 0)
        self.assertEqual(normalized["source"], "https://example.test/EquipSlotCategory")

    def test_item_model_normalization_unpacks_xivapi_uint64_fields(self) -> None:
        row = {
            "row_id": 42589,
            "fields": {
                "Name@lang(chs)": "天使画笔",
                "Name@lang(en)": "Angel Brush",
                "ModelMain": 4295035733,
                "ModelSub": 4295035783,
                "EquipSlotCategory@as(raw)": 1,
                "ItemUICategory@as(raw)": 110,
            },
        }

        normalized = item_models.normalize_item_row(
            row,
            source="https://example.test/Item",
        )

        self.assertEqual(normalized["game_item_id"], 42589)
        self.assertEqual(normalized["model_main"], [2901, 1, 1, 0])
        self.assertEqual(normalized["model_sub"], [2951, 1, 1, 0])
        self.assertEqual(normalized["equip_slot_category_id"], 1)
        self.assertEqual(normalized["item_ui_category_id"], 110)
        self.assertEqual(normalized["source"], "https://example.test/Item")

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
