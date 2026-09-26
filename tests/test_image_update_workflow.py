from __future__ import annotations

import io
import sys
from contextlib import closing
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ffxiv_prism.dataset.update_audit import sample_rows, update_result
from ffxiv_prism.dataset.image_qa import annotate_image_row
from ffxiv_prism.dataset.sqlite_store import build_eorzea_sqlite
from ffxiv_prism.jsonl import write_jsonl_atomic
from scripts import run_lifecycle_pipeline as lifecycle
from scripts import update_active_candidate_indexes as indexes
from scripts import web_app
from ffxiv_prism.matching import inference


class ImageUpdateWorkflowTests(unittest.TestCase):
    def test_default_is_image_update_without_model_release(self):
        args = lifecycle.parse_args([])
        self.assertEqual((args.workflow, args.source_mode, args.candidate_mode, args.device),
                         ("sample-library", "refresh", "extend", "auto"))
        stages = lifecycle.sample_stages(args, "test")
        names = [s.name for s in stages]
        self.assertEqual(names[:4], ["discover", "build-image-manifest", "download-images", "collect-details"])
        self.assertIn("--refresh-source", stages[1].command)
        self.assertTrue(all(s.advisory for s in stages[names.index("candidate-indexes") + 1:]))

    def test_web_modes_share_the_image_lifecycle(self):
        for mode, source in [("standard", "refresh"), ("full", "refresh"), ("local-index", "local")]:
            command = web_app.build_dataset_command(python_executable="python", mode=mode)
            self.assertTrue(command[1].endswith("run_lifecycle_pipeline.py"))
            self.assertEqual(command[command.index("--source-mode") + 1], source)
            self.assertEqual(command[command.index("--candidate-mode") + 1], "extend")
            self.assertNotIn("--release-mode", command)

    def test_metadata_only_is_not_reported_as_new_samples_and_web_keeps_result(self):
        result = update_result({}, {}, candidate={}, download={}, warnings=1, failed=False, planned=False)
        self.assertEqual(result["outcome"], "no_new_samples")
        self.assertIsNone(result["training_admitted_images"])
        job = web_app.Job("test", "dataset", "update")
        web_app._append_job_log(job, "DATASET_RESULT " + json.dumps(result))
        self.assertEqual(job.public()["dataset_result"], result)
        changed = update_result({}, {"eligible": ["new"], "discovered": ["new", "missing"], "downloaded": ["new"]},
                                candidate={"activated": True, "new_indexed_images": 1}, download={"images_downloaded": 1},
                                warnings=0, failed=False, planned=False)
        self.assertEqual(changed["outcome"], "images_indexed")
        self.assertEqual(changed["missing_download_images"], 1)
        self.assertIn("待处理", changed["message"])

    def test_eligibility_needs_image_and_label_and_deduplicates_by_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "image.png"
            Image.new("RGB", (80, 80)).save(path)
            database = root / "test.sqlite"
            with closing(sqlite3.connect(database)) as conn:
                conn.executescript("CREATE TABLE images (image_row_id INTEGER, source_image_url TEXT, local_path TEXT, sha256 TEXT, slot TEXT, dye_state TEXT, qa_status TEXT); CREATE TABLE gear_piece_images (image_row_id INTEGER, piece_id INTEGER, link_type TEXT);")
                rows = [(1, "valid", str(path), "same", "BODY", "normal", "ready"),
                        (2, "duplicate", str(path), "same", "BODY", "normal", "ready"),
                        (3, "unlabelled", str(path), "other", "BODY", "normal", "ready"),
                        (4, "invalid", str(path), "bad", "BODY", "normal", "invalid"),
                        (5, "missing", str(root / "missing.png"), "missing", "BODY", "normal", "ready"),
                        (6, "different-label", str(path), "same", "BODY", "normal", "ready")]
                conn.executemany("INSERT INTO images VALUES (?,?,?,?,?,?,?)", rows)
                conn.executemany("INSERT INTO gear_piece_images VALUES (?,?,?)", [(1, 10, "exact_slot"), (2, 10, "exact_slot"), (6, 20, "exact_slot")])
                conn.commit()
            states = {r["source_image_url"]: r["eligibility_status"] for r in sample_rows(database)}
            self.assertEqual(states, {"valid": "eligible", "duplicate": "duplicate", "unlabelled": "pending_labels",
                                     "invalid": "quality_rejected", "missing": "quality_rejected", "different-label": "eligible"})

    def test_truncated_pixels_are_rejected_even_with_valid_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.png"
            Image.new("RGB", (80, 80)).save(path)
            path.write_bytes(path.read_bytes()[:45])
            self.assertEqual(annotate_image_row({"local_path": str(path)})["qa_status"], "invalid")

    def test_database_failure_keeps_previous_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "library.sqlite"
            database.write_bytes(b"previous library")
            with patch("ffxiv_prism.dataset.sqlite_store.create_schema", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    build_eorzea_sqlite(database, root / "g", root / "p", root / "i", root / "summary.json")
            self.assertEqual(database.read_bytes(), b"previous library")
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_active_sources_and_atomic_activation_preserve_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = root / "selection.json"
            baseline = {"candidate_suffix": "20260829", "checkpoints": {"model": "unchanged"}}
            selection_path.write_text(json.dumps(baseline), encoding="utf-8")
            planned = indexes.target_plan("20260829", "20260923-123456")[:2]
            targets = [replace(t, source=root / t.source.name, output=root / t.output.name) for t in planned]
            results = []
            for target in targets:
                target.source.mkdir()
                write_jsonl_atomic(target.source / "metadata.jsonl", [{"source_image_url": "old"}])
                np.save(target.source / "embeddings.npy", np.zeros((1, 3), dtype=np.float32))
                target.output.mkdir()
                write_jsonl_atomic(target.output / "metadata.jsonl", [{"source_image_url": "old"}, {"source_image_url": "new"}])
                np.save(target.output / "embeddings.npy", np.zeros((2, 3), dtype=np.float32))
                (target.output / "summary.json").write_text('{"completed": true}', encoding="utf-8")
                results.append({"target": target.name, "output": str(target.output)})
            np.save(targets[1].output / "embeddings.npy", np.zeros((1, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "shape"):
                indexes.activate_targets(targets, results, baseline, selection_path)
            self.assertEqual(json.loads(selection_path.read_text()), baseline)
            np.save(targets[1].output / "embeddings.npy", np.zeros((2, 3), dtype=np.float32))
            self.assertTrue(indexes.activate_targets(targets, results, baseline, selection_path))
            updated = json.loads(selection_path.read_text())
            self.assertEqual(updated["checkpoints"], baseline["checkpoints"])
            next_plan = indexes.active_target_plan(updated, "20260924")
            self.assertEqual(next_plan[0].source, targets[0].output)
            self.assertTrue(indexes.activate_targets(targets, results, baseline, selection_path))
            current = inference.load_current_best_selection()
            current["candidate_indexes"] = updated["candidate_indexes"]
            with patch.object(inference, "load_current_best_selection", return_value=current):
                self.assertEqual(inference.current_best_configs()["BODY"].index_dir, targets[0].output)

    def test_metadata_failure_does_not_block_indexing_and_core_failure_does(self):
        for failed_stage, expected in [("cafemaker-zh", "warning"), ("validate-images", "failed")]:
            with self.subTest(stage=failed_stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "data").mkdir()
                (root / "data/current_best_selection.json").write_text('{"candidate_suffix": "20260829"}', encoding="utf-8")
                ran = []
                def fake_run(stage, *, dry_run):
                    ran.append(stage.name)
                    status = ("warning" if stage.advisory else "failed") if stage.name == failed_stage else "passed"
                    if stage.name == "candidate-indexes":
                        output = Path(stage.command[stage.command.index("--summary") + 1])
                        output.write_text('{"activated": true, "new_indexed_images": 2}', encoding="utf-8")
                    return lifecycle.StageResult(stage.name, stage.label, stage.phase, status,
                                                 1 if stage.name == failed_stage else 0, 0, list(stage.command), [])
                with patch.object(lifecycle, "PROJECT_ROOT", root), \
                     patch.object(lifecycle, "run_stage", side_effect=fake_run), \
                     patch.object(lifecycle, "collect_summary", return_value={}), \
                     patch.object(lifecycle, "image_snapshot", return_value={}), \
                     patch.object(sys, "stdout", io.StringIO()):
                    code = lifecycle.main(["--report-root", str(root / "reports")])
                report = next(path for path in (root / "reports").glob("*.json") if path.name.count(".") == 1)
                payload = json.loads(report.read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], expected)
                if failed_stage == "cafemaker-zh":
                    self.assertEqual(code, 0)
                    self.assertLess(ran.index("candidate-indexes"), ran.index(failed_stage))
                    self.assertEqual(payload["image_update"]["new_indexed_images"], 2)
                else:
                    self.assertEqual(code, 2)
                    self.assertNotIn("candidate-indexes", ran)
                    self.assertEqual(payload["image_update"]["outcome"], "failed")

    def test_interrupted_index_write_can_retry_without_publishing_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = replace(indexes.target_plan("20260829", "20260923")[0], source=root / "source", output=root / "output")
            target.source.mkdir()
            write_jsonl_atomic(target.source / "metadata.jsonl", [{"source_image_url": "old"}])
            np.save(target.source / "embeddings.npy", np.ones((1, 3), dtype=np.float32))
            additions = [{"source_image_url": "new"}]
            with patch.object(indexes, "write_jsonl", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    indexes.extend_index(target, additions, np.ones((1, 3), dtype=np.float32))
            self.assertFalse(target.output.exists())
            result = indexes.extend_index(target, additions, np.ones((1, 3), dtype=np.float32))
            self.assertEqual(result["candidate_count"], 2)
            self.assertTrue(indexes.load_completed_target(target, additions)["reused"])

    def test_resume_ignores_only_output_summary_path_not_source_version(self):
        stage = lifecycle.Stage("candidate-indexes", "index", ("python", "script.py", "--summary", "new.json", "--source-selection", "v1.json"), "sample-library")
        old = {"status": "passed", "command": ["python", "script.py", "--summary", "old.json", "--source-selection", "v1.json"]}
        self.assertIsNotNone(lifecycle.reused_stage_result(stage, old))
        old["command"][-1] = "v0.json"
        self.assertIsNone(lifecycle.reused_stage_result(stage, old))


if __name__ == "__main__":
    unittest.main()
