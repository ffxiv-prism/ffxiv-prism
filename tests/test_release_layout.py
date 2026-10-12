import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.release_layout import compact_resources, inference_sources, resource_name, index_names

ROOT = Path(__file__).resolve().parents[1]


class InferenceReleaseTests(unittest.TestCase):
    def test_entry_point_closure_excludes_training_and_build_files(self):
        files = inference_sources(ROOT)
        self.assertIn("src/ffxiv_prism/matching/runtime/query.py", files)
        self.assertIn("src/ffxiv_prism/matching/report_export.py", files)
        self.assertIn("src/ffxiv_prism/collectors/eorzea_collection/visible_browser.py", files)
        self.assertFalse(any("evaluation" in p or "adaptation" in p or "egg-info" in p for p in files))
        self.assertNotIn("src/ffxiv_prism/collectors/eorzea_collection/collector.py", files)

    def test_compaction_updates_json_sqlite_and_selection_without_touching_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = "data/raw/eorzea_collection/images/gearset_static/hat/raid-hat-head.png"
            old_index = "data/indexes/facebook-dinov2-base-head-multiscale-v4-20261009-230000"
            selection = {"candidate_indexes": {"facebook-dinov2-base-head-multiscale-v4": old_index}}
            metadata = root / "metadata.jsonl"
            metadata.write_text(json.dumps({"local_path": image}) + "\n")
            selected = root / "selection.json"
            selected.write_text(json.dumps(selection))
            dbpath = root / "build.sqlite"
            with closing(sqlite3.connect(dbpath)) as db, db:
                db.execute("CREATE TABLE images (local_path TEXT)")
                db.execute("INSERT INTO images VALUES (?)", (image,))
            result = compact_resources({old_index + "/metadata.jsonl": metadata,
                                        "data/current_best_selection.json": selected,
                                        "data/processed/eorzea.sqlite": dbpath}, selection, root / "compact")
            new_image = "data/images/raid-hat-head.png"
            self.assertEqual(json.loads(result["data/indexes/head-dinov2/metadata.jsonl"].read_text())["local_path"], new_image)
            self.assertEqual(json.loads(metadata.read_text())["local_path"], image)
            with closing(sqlite3.connect(dbpath)) as db:
                self.assertEqual(db.execute("SELECT local_path FROM images").fetchone()[0], new_image)
            self.assertEqual(json.loads(result["data/current_best_selection.json"].read_text())["candidate_indexes"],
                             {"facebook-dinov2-base-head-multiscale-v4": "data/indexes/head-dinov2"})

    def test_flattening_rejects_case_insensitive_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Duplicate compact"):
                compact_resources({"data/raw/eorzea_collection/images/a/hat.png": Path("a.png"),
                                   "data/raw/eorzea_collection/images/b/HAT.png": Path("b.png")},
                                  {"candidate_indexes": {}}, Path(tmp))

    def test_active_gallery_paths_fit_64_character_root(self):
        selection = json.loads((ROOT / "data/current_best_selection.json").read_text("utf-8"))
        names = index_names(selection)
        for folder in selection["candidate_indexes"].values():
            for line in (ROOT / folder / "metadata.jsonl").read_text("utf-8").splitlines():
                path = resource_name(json.loads(line)["local_path"], names)
                self.assertLessEqual(64 + 1 + len("resources/gallery-20261009/" + path), 249)


if __name__ == "__main__":
    unittest.main()
