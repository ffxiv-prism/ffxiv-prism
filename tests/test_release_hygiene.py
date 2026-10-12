import json
import hashlib
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bootstrap'))
import portable
import ffxiv_prism.version_info as versions


class ReleaseHygieneTests(unittest.TestCase):
    def test_damaged_gallery_repair_uses_short_backup_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            manifest = {'kind': 'resources', 'id': 'gallery-' + 'a' * 24,
                        'files': [{'path': 'image.bin', 'size': 3,
                                   'sha256': hashlib.sha256(b'new').hexdigest()}]}
            raw = json.dumps(manifest).encode()
            installed = home / 'resources' / manifest['id']
            installed.mkdir(parents=True)
            (installed / 'package.json').write_bytes(raw)
            (installed / 'image.bin').write_bytes(b'old')
            archive = home / 'renamed.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('package.json', raw)
                z.writestr('package.sig', b'test')
                z.writestr('payload/image.bin', b'new')
            with patch.object(portable, 'package_manifest', return_value=manifest):
                portable.install_package(home, archive)
            backup = next((home / 'work/rollback').iterdir())
            self.assertRegex(backup.name, r'^r-[0-9a-f]+$')
            self.assertEqual((backup / 'image.bin').read_bytes(), b'old')
            self.assertEqual((installed / 'image.bin').read_bytes(), b'new')

    def test_path_budget_boundary_counts_drive_and_separators(self):
        with patch.object(portable.os, 'name', 'nt'):
            with patch.object(portable.Path, 'resolve', return_value='D:\\' + 'a' * 61):
                portable.check_install_root('unused')
            with patch.object(portable.Path, 'resolve', return_value='D:\\' + 'a' * 62):
                with self.assertRaisesRegex(ValueError, '缩短 1'):
                    portable.check_install_root('unused')

    def test_cleanup_only_after_success_and_only_inside_imports(self):
        launch = runpy.run_path(str(ROOT / 'bootstrap/launch.py'), run_name='test_launch')
        fn = launch['install_inputs']
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            inside = home / 'imports/renamed-gallery.zip'
            outside = home / 'original-gallery.zip'
            inside.parent.mkdir()
            inside.write_bytes(b'keep until selfcheck succeeds')
            outside.write_bytes(b'outside imports')
            active = {'app': 'a', 'resources': 'r'}
            globals_ = fn.__globals__
            with patch.dict(globals_, HOME=home, install_package=lambda *args: {'kind': 'resources', 'id': 'r'},
                            check_gallery_inputs=lambda *args: None,
                            activate_paths=lambda *args: (home, home), prepare_models=lambda *args, **kwargs: None,
                            prepare_optional=lambda *args: None, validate_combination=lambda *args, **kwargs: None):
                with patch.object(subprocess, 'run', side_effect=RuntimeError('probe failed')):
                    with self.assertRaisesRegex(RuntimeError, 'probe failed'):
                        fn(active, [inside])
                self.assertTrue(inside.exists())
                with patch.object(subprocess, 'run'):
                    fn(active, [inside])
                self.assertFalse(inside.exists())
                portable.remove_managed_files(home, [outside], 'imports')
                self.assertTrue(outside.exists())

    def test_source_version_ignores_stale_built_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / '.git').mkdir()
            (root / 'manifests').mkdir()
            (root / 'manifests/release.json').write_text(json.dumps({'version': 'old', 'code_commit': 'stale'}))
            (root / 'pyproject.toml').write_text('[project]\nversion="0.2.0"')
            with patch.object(versions, 'APP_ROOT', root), patch.object(versions.subprocess, 'check_output', return_value='actual\n'):
                self.assertEqual(versions.version_info()['code_commit'], 'actual')
                self.assertEqual(versions.version_info()['release'], 'v0.2.0')


if __name__ == '__main__':
    unittest.main()
