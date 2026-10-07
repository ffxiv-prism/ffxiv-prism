import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bootstrap"))
from scripts import portable_upgrade as upgrade


class UpgradeTests(unittest.TestCase):
    def fixture(self, root, version):
        home, bundle = root / "home", root / "bundle"
        for name, content in {"bootstrap/obsolete.py": "old bootstrap", "启动.bat": "old bat",
                              "runtime/python/python312._pth": "old paths",
                              "manifests/active.json": json.dumps({"app": version, "resources": "gallery"}),
                              "manifests/previous.json": "{}", "user/report.txt": "keep",
                              "work/pending-update.json": "stale", "work/proposed.json": "stale"}.items():
            path = home / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        files = []
        for name in ("bootstrap/launch.py", "启动.bat"):
            path = bundle / "payload" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("new")
            files.append({"path": "payload/" + name})
        manifest = {"version": "v3", "supported_from": ["v1", "v2"], "app_package": "app.zip", "files": files}
        return home, bundle, manifest

    def test_upgrades_both_versions_removes_obsolete_files_preserves_user_data(self):
        for version in ("v1", "v2"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                home, bundle, manifest = self.fixture(Path(directory), version)
                def install(target, archive, active):
                    self.assertEqual(active["app"], version)
                    (target / "manifests/active.json").write_text('{"app":"v3"}')
                upgrade.apply_upgrade(home, bundle, manifest, install)
                self.assertFalse((home / "bootstrap/obsolete.py").exists())
                self.assertEqual((home / "bootstrap/launch.py").read_text(), "new")
                self.assertFalse((home / "work/pending-update.json").exists())
                self.assertFalse((home / "work/proposed.json").exists())
                self.assertEqual((home / "user/report.txt").read_text(), "keep")
                self.assertEqual(len(list((home / "work/upgrade-backups").glob("*/bootstrap/obsolete.py"))), 1)

    def test_install_failure_restores_files_activation_and_python_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            home, bundle, manifest = self.fixture(Path(directory), "v1")
            before = {name: (home / name).read_bytes() for name in upgrade.STATE}
            def fail(target, archive, active):
                for name in upgrade.STATE:
                    (target / name).write_text("changed")
                raise RuntimeError("selfcheck failed")
            with self.assertRaisesRegex(RuntimeError, "selfcheck failed"):
                upgrade.apply_upgrade(home, bundle, manifest, fail)
            self.assertEqual({name: (home / name).read_bytes() for name in upgrade.STATE}, before)
            self.assertTrue((home / "bootstrap/obsolete.py").is_file())
            self.assertFalse((home / "bootstrap/launch.py").exists())
            self.assertEqual((home / "启动.bat").read_text(), "old bat")

    def test_unknown_source_version_rejected_before_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            home, bundle, manifest = self.fixture(Path(directory), "v0")
            with self.assertRaises(ValueError):
                upgrade.apply_upgrade(home, bundle, manifest)
            self.assertTrue((home / "bootstrap/obsolete.py").exists())
            self.assertFalse((home / "work/upgrade-backups").exists())
