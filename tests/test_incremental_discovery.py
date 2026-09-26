from __future__ import annotations

import copy
import io
import json
import sys
import sqlite3
from contextlib import closing
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from ffxiv_prism.collectors.eorzea_collection.discovery import scan_model
from ffxiv_prism.jsonl import read_jsonl, write_jsonl_atomic
from scripts import discover_eorzea_gearsets_by_model as discover
from scripts import build_eorzea_image_url_manifest_from_list_cache as manifest
from scripts import run_lifecycle_pipeline as lifecycle
from scripts import run_data_pipeline as pipeline

BASE = "https://ffxiv.eorzeacollection.com"
IMAGE = "https://gearsets.eorzeacollection.com/raid/current/current/hyur-female-body.png"


def page(number, *, image=None):
    return '<title>Gearsets | Eorzea Collection</title>' + (
        f'<a href="/gearset/set-{number}">set</a>' if number else ""
    ) + (f'<img src="{image}">' if image else "")


def run_scan(state, *, full=False, known=None, fetch=None, checkpoint=None, **kwargs):
    return scan_model(model="female", state=state,
        known_links=known if known is not None else {f"{BASE}/gearset/set-{i}" for i in range(1, 101)},
        known_seeds=set(), fetch=fetch or (lambda p: (page(p), False)),
        checkpoint=checkpoint or (lambda *args: None), full=full, **kwargs)


class IncrementalDiscoveryTests(unittest.TestCase):
    def test_known_library_uses_six_pages_and_rotates_history(self):
        state = {}
        first = run_scan(state)
        second = run_scan(state)
        self.assertEqual(first["pages_checked"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(second["pages_checked"], [1, 2, 3, 7, 8, 9])
        self.assertEqual(first["remote_pages"], 6)

    def test_many_new_pages_have_hard_request_budget(self):
        state = {}
        result = run_scan(state, known=set())
        self.assertEqual(result["pages_checked"], list(range(1, 12)))
        self.assertEqual(state["tail_cursor"], 12)

    def test_existing_gearset_new_image_extends_head_scan(self):
        result = run_scan({}, fetch=lambda p: (page(p, image=IMAGE if p == 3 else None), False))
        self.assertEqual(result["pages_checked"], [1, 2, 3, 4, 5, 6, 7, 8])

    def test_rotation_finds_changes_outside_head(self):
        state = {"tail_cursor": 20}
        known = {f"{BASE}/gearset/set-{i}" for i in range(1, 101)}
        run_scan(state, known=known, fetch=lambda p: (page(999 if p == 21 else p), False))
        self.assertIn(f"{BASE}/gearset/set-999", known)
        self.assertEqual(state["tail_cursor"], 23)

    def test_terminal_pages_wrap_rotation(self):
        state = {"tail_cursor": 100}
        run_scan(state, fetch=lambda p: (page(p if p < 100 else None), False))
        self.assertEqual(state["tail_cursor"], 4)
        self.assertEqual(state["rotation_cycles"], 1)

    def test_initial_full_scan_and_resume_do_not_refetch_passed_pages(self):
        saved = {}
        calls = []
        def checkpoint(*args):
            saved.clear()
            saved.update(copy.deepcopy(state))
        def fetch(p):
            calls.append(p)
            if p == 4:
                raise OSError("interrupted")
            return page(p), False
        state = {}
        with self.assertRaises(OSError):
            run_scan(state, full=True, known=set(), fetch=fetch, checkpoint=checkpoint)
        self.assertEqual(saved["full_next_page"], 4)
        result = run_scan(saved, full=True, fetch=lambda p: (page(p if p < 6 else None), False))
        self.assertEqual(result["pages_checked"], [4, 5, 6, 7])
        self.assertTrue(result["full_complete"])

    def test_challenge_page_does_not_advance_checkpoint(self):
        state = {"tail_cursor": 20}
        checkpoint_calls = []
        with self.assertRaisesRegex(ValueError, "verification"):
            run_scan(state, fetch=lambda p: ('<title>Just a moment...</title>', False),
                     checkpoint=lambda *args: checkpoint_calls.append(args))
        self.assertEqual(checkpoint_calls, [])
        self.assertEqual(state["tail_cursor"], 20)

    def test_empty_initial_scan_is_not_success(self):
        with self.assertRaisesRegex(ValueError, "empty baseline"):
            run_scan({}, full=True, known=set(), fetch=lambda p: (page(None), False))

    def test_full_scan_reaching_cap_stays_incomplete(self):
        state = {}
        result = run_scan(state, full=True, max_pages=8)
        self.assertFalse(result["full_complete"])
        with self.assertRaisesRegex(ValueError, "max-pages"):
            run_scan(state, full=True, max_pages=8)

    def test_existing_snapshot_without_state_or_html_stays_incremental_and_merges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "data/raw/eorzea_collection/metadata"
            write_jsonl_atomic(metadata / "gearset_details_raw.jsonl",
                              [{"source_page_url": f"{BASE}/gearset/set-{i}"} for i in range(1, 101)])
            state_path = root / "state.json"
            requested = []
            def fetch(url, *args, **kwargs):
                from urllib.parse import parse_qs, urlparse
                n = int(parse_qs(urlparse(url).query)["page"][0])
                requested.append(n)
                return page(999 if n == 2 else n), False
            with patch.object(discover, "PROJECT_ROOT", root), \
                 patch.object(discover, "INTERIM_DIR", root / "interim"), \
                 patch.object(discover.EorzeaCollectionCollector, "_fetch_html", side_effect=fetch), \
                 patch.object(sys, "argv", ["discover", "--models", "female", "--refresh", "--state-file", str(state_path)]), \
                 patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(discover.main(), 0)
            summary = json.loads(state_path.with_name("last_scan_summary.json").read_text())
            self.assertEqual(summary["mode"], "incremental")
            self.assertLessEqual(len(requested), 11)
            self.assertEqual(len(read_jsonl(metadata / "gearset_discovery_by_model_raw.jsonl")), 101)
            self.assertEqual(len(read_jsonl(root / "interim/gearset_detail_queue.jsonl")), 101)

    def test_sqlite_only_library_does_not_trigger_full_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "data/processed/eorzea.sqlite"
            database.parent.mkdir(parents=True)
            with closing(sqlite3.connect(database)) as conn:
                conn.execute("CREATE TABLE gearsets (source_page_url TEXT)")
                conn.execute("INSERT INTO gearsets VALUES (?)", (f"{BASE}/gearset/set-1",))
                conn.commit()
            with patch.object(discover, "PROJECT_ROOT", root), \
                 patch.object(discover, "INTERIM_DIR", root / "interim"), \
                 patch.object(discover.EorzeaCollectionCollector, "_fetch_html", return_value=(page(1), False)), \
                 patch.object(sys, "argv", ["discover", "--models", "female", "--state-file", str(root / "state.json")]), \
                 patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(discover.main(), 0)
            self.assertEqual(json.loads((root / "last_scan_summary.json").read_text())["mode"], "incremental")

    def test_unified_and_compatibility_entry_points_forward_scan_controls(self):
        args = lifecycle.parse_args(["--discovery-mode", "full", "--discovery-history-pages", "5"])
        stage = lifecycle.sample_stages(args, "test")[0]
        self.assertIn("full", stage.command)
        self.assertIn("--discovery-history-pages", stage.command)
        args.refresh_source = True
        command = pipeline.commands_for_stage("discover", args)[0]
        self.assertIn("--history-pages", command)
        self.assertEqual(command[command.index("--history-pages") + 1], "5")
        self.assertEqual(command[command.index("--discovery-mode") + 1], "full")


class RetryBudgetTests(unittest.TestCase):
    def test_cooldowns_oldest_first_budget_and_success_reuse(self):
        now = datetime(2026, 9, 24, tzinfo=UTC)
        def row(age, status=404, accessible=False):
            return {"checked_at": (now - timedelta(hours=age)).isoformat(),
                    "status": status, "accessible": accessible}
        rows = {"old404": row(200), "new404": row(20), "error": row(7, 503),
                "new_error": row(1, 503), "ok": row(900, 200, True), "oldest": row(400)}
        self.assertEqual(manifest.select_retry_urls(rows, enabled=True, limit=2, now=now), {"oldest", "old404"})
        self.assertEqual(manifest.select_retry_urls(rows, enabled=True, now=now), {"oldest", "old404", "error"})
        self.assertEqual(manifest.select_retry_urls(rows, enabled=False, now=now), set())
        self.assertEqual(manifest.select_retry_urls(rows, enabled=True, limit=0, now=now), set())

    def test_browser_retries_are_also_bounded_and_legacy_timestamps_eligible(self):
        rows = {str(i): {"status": 403, "content_type": "text/html", "accessible": False}
                for i in range(500)}
        self.assertEqual(len(manifest.select_retry_urls(rows, enabled=False, browser=True)), 200)

    def test_small_list_cache_preserves_old_failures_and_does_not_check_them_all(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = IMAGE.replace("/current/current/", "/old/old/")
            rows = [{"source_image_url": old, "accessible": True, "status": 200, "image_variant": "normal"},
                    {"source_image_url": IMAGE, "accessible": False, "status": 404,
                     "checked_at": datetime.now(UTC).isoformat()}]
            write_jsonl_atomic(root / "image_url_checks_raw.jsonl", rows)
            with patch.object(manifest, "EORZEA_METADATA_DIR", root), \
                 patch.object(manifest, "discover_seed_urls", return_value=[IMAGE]), \
                 patch.object(manifest, "generate_candidate_urls", return_value=[IMAGE]), \
                 patch.object(manifest, "check_image_url") as check, \
                 patch.object(sys, "argv", ["manifest", "--resume", "--retry-unavailable"]), \
                 patch.object(sys, "stdout", io.StringIO()), patch.object(sys, "stderr", io.StringIO()):
                self.assertEqual(manifest.main(), 0)
            check.assert_not_called()
            self.assertEqual([r["source_image_url"] for r in read_jsonl(root / "image_url_manifest_raw.jsonl")], [old])

    def test_new_image_is_checked_immediately_even_with_zero_historical_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {"source_image_url": IMAGE, "status": 200, "content_type": "image/png",
                      "content_length": 10, "error": None, "checked_at": datetime.now(UTC).isoformat()}
            with patch.object(manifest, "EORZEA_METADATA_DIR", root), \
                 patch.object(manifest, "discover_seed_urls", return_value=[IMAGE]), \
                 patch.object(manifest, "generate_candidate_urls", return_value=[IMAGE]), \
                 patch.object(manifest, "check_image_url", return_value=result) as check, \
                 patch.object(sys, "argv", ["manifest", "--resume", "--retry-unavailable", "--retry-limit", "0"]), \
                 patch.object(sys, "stdout", io.StringIO()), patch.object(sys, "stderr", io.StringIO()):
                self.assertEqual(manifest.main(), 0)
            check.assert_called_once()
            self.assertEqual(read_jsonl(root / "image_url_manifest_raw.jsonl")[0]["source_image_url"], IMAGE)


if __name__ == "__main__":
    unittest.main()
