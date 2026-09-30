"""Package trust boundary, atomic activation and portable cache invariants."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bootstrap"))
import portable
from ffxiv_prism.matching import asset_identity

class PortableTests(unittest.TestCase):
    def test_paths_reject_traversal_and_windows_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../escape", "/absolute", "C:/escape", "a\\..\\x", "x:stream", "CON.txt", "a/../b", "a./b", "a//b"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    portable.safe_path(directory, name)
            self.assertEqual(portable.safe_path(directory, "data/中文 文件.png"), Path(directory) / "data/中文 文件.png")

    def test_content_signature_survives_move_and_timestamp_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "a", root / "b"
            a.mkdir()
            (a / "model.safetensors").write_bytes(b"weights")
            (a / "config.json").write_text('{}')
            with patch.object(asset_identity, "model_weight_files", return_value=[a / "model.safetensors"]):
                asset_identity.asset_signature.cache_clear()
                before = asset_identity.asset_signature("model")
            shutil.copytree(a, b)
            os.utime(b / "model.safetensors", (1, 1))
            (b / "model.safetensors.verified.json").write_text('{}')
            with patch.object(asset_identity, "model_weight_files", return_value=[b / "model.safetensors"]):
                asset_identity.asset_signature.cache_clear()
                self.assertEqual(before, asset_identity.asset_signature("model"))
                (b / "model.safetensors").write_bytes(b"changed")
                asset_identity.asset_signature.cache_clear()
                self.assertNotEqual(before, asset_identity.asset_signature("model"))
            asset_identity.asset_signature.cache_clear()

    @unittest.skipUnless(os.name == "nt", "Uses Windows signing tool")
    def test_signed_install_tamper_rejected_without_switching_active(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "bootstrap").mkdir()
            content = b"test equipment resource"
            manifest = {"schema": 1, "kind": "resources", "id": "r1", "compatibility": "test", "files": [
                {"path": "data/image.bin", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}]}
            source = home / "package.json"
            source.write_text(json.dumps(manifest), "utf-8")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/sign_portable.ps1"),
                            "-Manifest", str(source), "-KeyPath", str(home / "private.xml"),
                            "-PublicPath", str(home / "bootstrap/public-key.json")], check=True, capture_output=True)
            archive = home / "good.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.write(source, "package.json")
                output.write(source.with_suffix(".sig"), "package.sig")
                output.writestr("payload/data/image.bin", content)
            portable.atomic_json(home / "manifests/active.json", {"app": "old", "resources": "old"})
            portable.install_package(home, archive)
            self.assertEqual((home / "resources/r1/data/image.bin").read_bytes(), content)
            self.assertEqual(portable.read_json(home / "manifests/active.json")["resources"], "old")
            portable.verify_signature(source.read_bytes(), source.with_suffix(".sig").read_bytes(), portable.read_json(home / "bootstrap/public-key.json"))
            with self.assertRaises(ValueError):
                portable.verify_signature(source.read_bytes() + b" ", source.with_suffix(".sig").read_bytes(), portable.read_json(home / "bootstrap/public-key.json"))
            bad = home / "bad.zip"
            with zipfile.ZipFile(bad, "w") as output:
                output.write(source, "package.json")
                output.write(source.with_suffix(".sig"), "package.sig")
                output.writestr("payload/data/image.bin", b"X" * len(content))
            shutil.rmtree(home / "resources/r1")
            with self.assertRaises(ValueError):
                portable.install_package(home, bad)
            self.assertFalse((home / "resources/r1").exists())
            self.assertEqual(portable.read_json(home / "manifests/active.json")["resources"], "old")

if __name__ == "__main__":
    unittest.main()
