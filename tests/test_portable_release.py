"""Package trust boundary, atomic activation and portable cache invariants."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bootstrap"))
import portable
from ffxiv_prism.matching import asset_identity

class DownloadResponse(io.BytesIO):
    def __init__(self, data, status=200, headers=None, url="https://cdn.example/file"):
        super().__init__(data)
        self.status, self.headers, self.url = status, headers or {}, url

    def geturl(self):
        return self.url

class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.content = b"verified payload"
        self.row = {"size": len(self.content), "sha256": hashlib.sha256(self.content).hexdigest(),
                    "urls": ["https://official.example/file", "https://mirror.example/file"]}

    def test_failed_source_falls_back_with_concise_output_for_small_files(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output):
            path = Path(directory) / "file.whl"
            responses = [urllib.error.HTTPError(self.row['urls'][0], 404, "missing", {}, None),
                         DownloadResponse(self.content)]
            with patch.object(portable.urllib.request, 'urlopen', side_effect=responses):
                portable.download(self.row, path)
            self.assertEqual(path.read_bytes(), self.content)
            self.assertFalse(path.with_suffix('.whl.part').exists())
        for expected in ('official.example', 'mirror.example', 'Download failed'):
            self.assertIn(expected, output.getvalue())
        self.assertNotIn('https://', output.getvalue())
        self.assertNotIn('MiB/s', output.getvalue())
        self.assertLessEqual(len(output.getvalue().splitlines()), 3)

    def test_resume_uses_range_and_preserves_downloaded_bytes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / 'model.bin'
            path.with_suffix('.bin.part').write_bytes(self.content[:5])
            response = DownloadResponse(self.content[5:], 206, {'Content-Range': f'bytes 5-{len(self.content)-1}/{len(self.content)}'})
            with patch.object(portable.urllib.request, 'urlopen', return_value=response) as fetch:
                portable.download(self.row, path)
            self.assertEqual(fetch.call_args.args[0].get_header('Range'), 'bytes=5-')
            self.assertEqual(path.read_bytes(), self.content)

    def test_ignored_range_restarts_instead_of_appending(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / 'model.bin'
            path.with_suffix('.bin.part').write_bytes(b'wrong prefix')
            with patch.object(portable.urllib.request, 'urlopen', return_value=DownloadResponse(self.content)):
                portable.download(self.row, path)
            self.assertEqual(path.read_bytes(), self.content)

    def test_truncated_response_retries_from_partial_file(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / 'file'
            responses = [DownloadResponse(self.content[:5]),
                         DownloadResponse(self.content[5:], 206, {'Content-Range': f'bytes 5-{len(self.content)-1}/{len(self.content)}'})]
            with patch.object(portable.urllib.request, 'urlopen', side_effect=responses) as fetch, patch.object(portable.time, 'sleep'):
                portable.download(self.row, path)
            self.assertEqual(fetch.call_args.args[0].get_header('Range'), 'bytes=5-')
            self.assertEqual(path.read_bytes(), self.content)

    def test_wrong_hash_never_replaces_destination(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / 'file'
            path.write_bytes(b'old file')
            with patch.object(portable.urllib.request, 'urlopen', side_effect=lambda *a, **k: DownloadResponse(b'X' * len(self.content))), patch.object(portable.time, 'sleep'):
                with self.assertRaises(RuntimeError):
                    portable.download(self.row, path)
            self.assertEqual(path.read_bytes(), b'old file')
            self.assertFalse(path.with_suffix('.part').exists())

    def test_complete_partial_or_valid_cache_needs_no_network(self):
        for partial in (True, False):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                path = Path(directory) / 'file'
                (path.with_suffix('.part') if partial else path).write_bytes(self.content)
                with patch.object(portable.urllib.request, 'urlopen') as fetch:
                    portable.download(self.row, path)
                fetch.assert_not_called()
                self.assertEqual(path.read_bytes(), self.content)

    def test_invalid_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / 'file'
            path.with_suffix('.part').write_bytes(self.content[:5])
            with patch.object(portable.urllib.request, 'urlopen', side_effect=lambda *a, **k: DownloadResponse(self.content, 206, {'Content-Range': 'bytes 0-15/16'})), patch.object(portable.time, 'sleep'):
                with self.assertRaises(RuntimeError):
                    portable.download(self.row, path)
            self.assertFalse(path.exists())

    def test_locked_mirrors_keep_original_file_identity(self):
        from scripts.lock_portable_runtime import download_urls
        lock = portable.read_json(ROOT / 'manifests/runtime.json')
        for row in lock['wheels']:
            self.assertEqual(row['urls'], download_urls(row['urls'][-1]))
            self.assertEqual(len(row['sha256']), 64)
            if 'files.pythonhosted.org' in row['urls'][-1]:
                self.assertEqual(len(row['urls']), 3)
            else:
                self.assertIn('mirror.sjtu.edu.cn', row['urls'][0])

class PortableTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Uses Windows signing tool")
    def test_explicit_rotation_retains_old_signatures_and_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'package.json'
            manifest.write_bytes(b'{"version":1}')
            public = root / 'public.json'
            def sign(key, *flags, trust=public, ok=True):
                result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                         str(ROOT / 'scripts/sign_portable.ps1'), '-Manifest', str(manifest),
                                         '-KeyPath', str(root / key), '-PublicPath', str(trust), *flags], capture_output=True)
                self.assertEqual(result.returncode == 0, ok, result.stderr.decode(errors='replace'))
                return manifest.with_suffix('.sig').read_bytes() if ok else None
            old_sig = sign('old.xml', '-InitializeKey')
            old_public = public.read_bytes()
            sign('new.xml', ok=False)
            sign('new.xml', '-InitializeKey', ok=False)
            self.assertEqual(public.read_bytes(), old_public)
            self.assertFalse((root / 'new.xml').exists())
            new_sig = sign('new.xml', '-InitializeKey', '-RotateKey')
            trusted = portable.read_json(public)
            self.assertEqual(len(trusted['previous_keys']), 1)
            portable.verify_signature(manifest.read_bytes(), old_sig, trusted)
            portable.verify_signature(manifest.read_bytes(), new_sig, trusted)
            for sig in (old_sig, new_sig):
                with self.assertRaises(ValueError):
                    portable.verify_signature(b'tampered', sig, trusted)
            with self.assertRaises(ValueError):
                portable.verify_signature(manifest.read_bytes(), new_sig, json.loads(old_public))
            unknown = sign('unknown.xml', '-InitializeKey', trust=root / 'unknown.json')
            with self.assertRaises(ValueError):
                portable.verify_signature(manifest.read_bytes(), unknown, trusted)
            sign('old.xml', ok=False)
            before = public.read_bytes()
            historical_sig = sign('old.xml', '-AllowTrustedPreviousKey')
            self.assertEqual(public.read_bytes(), before)
            portable.verify_signature(manifest.read_bytes(), historical_sig, trusted)
            sign('unknown.xml', '-AllowTrustedPreviousKey', ok=False)
            self.assertEqual(public.read_bytes(), before)
            sign('new.xml', '-RotateKey', ok=False)
            sign('new.xml')
            self.assertEqual(portable.read_json(public), trusted)

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
                            "-PublicPath", str(home / "bootstrap/public-key.json"), "-InitializeKey"], check=True, capture_output=True)
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
