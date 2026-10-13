import json
from pathlib import Path
import tempfile
import unittest

from scripts.migrate_gallery_layout import migrate
from ffxiv_prism.paths import EORZEA_IMAGE_DIR, resolve_project_path


class GalleryMigrationTests(unittest.TestCase):
    def fixture(self, root):
        index = root / "data/indexes/old-head"
        index.mkdir(parents=True)
        image = root / "data/raw/eorzea_collection/images/a/hat.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        (index / "embeddings.npy").write_bytes(b"unchanged vectors")
        (index / "metadata.jsonl").write_text(json.dumps({"local_path": image.relative_to(root).as_posix()}) + "\n")
        (index / "summary.json").write_text('{}')
        (root / "data/current_best_selection.json").write_text(json.dumps({
            "candidate_indexes": {"facebook-dinov2-base-head-v1": "data/indexes/old-head"}}))
        return image, index

    def test_migration_preserves_vectors_and_history_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, index = self.fixture(root)
            self.assertFalse(migrate(root)["applied"])
            self.assertFalse((root / "data/images").exists())
            migrate(root, apply=True)
            self.assertEqual((root / "data/images/hat.png").read_bytes(), image.read_bytes())
            compact = root / "data/indexes/head-dinov2"
            self.assertEqual((compact / "embeddings.npy").read_bytes(), (index / "embeddings.npy").read_bytes())
            self.assertEqual(json.loads((compact / "metadata.jsonl").read_text())["local_path"], "data/images/hat.png")
            self.assertIn("data/raw/", (index / "metadata.jsonl").read_text())
            before = (root / "data/current_best_selection.json").read_bytes()
            migrate(root, apply=True)
            self.assertEqual(before, (root / "data/current_best_selection.json").read_bytes())

    def test_conflicting_flat_names_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, _ = self.fixture(root)
            other = image.parent.parent / "b/HAT.png"
            other.parent.mkdir()
            other.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "Conflicting flat image"):
                migrate(root, apply=True)
            self.assertFalse((root / "data/images").exists())

    def test_historical_image_references_resolve_to_shared_gallery(self):
        self.assertEqual(resolve_project_path("data/raw/eorzea_collection/images/gearset_static/a/hat.png"),
                         EORZEA_IMAGE_DIR / "hat.png")
