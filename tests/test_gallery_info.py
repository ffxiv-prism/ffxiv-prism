import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ffxiv_prism.gallery_info import gallery_info


class GalleryInfoTests(unittest.TestCase):
    def test_installed_and_development_dates_match_and_ignore_mtime(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests/galleries").mkdir(parents=True)
            (root / "data").mkdir()
            snapshot = {"id": "gallery", "updated_at": "2026-09-26", "candidate_indexes": {"head": "index"}}
            path = root / "manifests/galleries/gallery.json"
            path.write_text(json.dumps(snapshot))
            (root / "data/current_best_selection.json").write_text(json.dumps(snapshot))
            self.assertEqual(gallery_info(root, root), snapshot)
            os.utime(path, (1, 1))
            (root / "gallery.json").write_text('{"id":"gallery","updated_at":"wrong index date"}')
            self.assertEqual(gallery_info(root, root), snapshot)

    def test_different_snapshot_never_inherits_old_date(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gallery.json").write_text('{"id":"other","updated_at":"wrong index date","image_count":2}')
            info = gallery_info(root, root)
            self.assertIsNone(info["updated_at"])
            self.assertEqual(info["image_count"], 2)
