"""First installation, later GPU addition, and cleanup boundaries."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = runpy.run_path(str(ROOT / 'bootstrap/launch.py'), run_name='test_install')
import portable


class InstallationTests(unittest.TestCase):
    def test_missing_or_multiple_galleries_never_start_dependencies(self):
        install = LAUNCH['install_inputs']
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            runtime = Mock()
            with patch.dict(install.__globals__, HOME=Path(directory), activate_paths=runtime):
                for files in ([], [Path('one.zip'), Path('two.zip')]):
                    with self.assertRaises(ValueError):
                        install({'app': 'a', 'resources': None}, files, 'cpu')
            runtime.assert_not_called()

    def test_wrong_kind_or_incompatible_gallery_rejected_before_extraction(self):
        install = LAUNCH['install_inputs']
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            home = Path(directory)
            archive = home / 'input.zip'
            with zipfile.ZipFile(archive, 'w'):
                pass
            extract = Mock()
            runtime = Mock()
            for manifest in ({'kind': 'app'}, {'kind': 'resources', 'compatibility': 'other'}):
                with patch.dict(install.__globals__, HOME=home, install_package=extract,
                                activate_paths=runtime, package_manifest=lambda *args: manifest,
                                installed_manifest=lambda *args: {'resource_compatibility': ['test']}):
                    with self.assertRaises(ValueError):
                        install({'app': 'a', 'resources': None}, [archive], 'cpu')
            extract.assert_not_called()
            runtime.assert_not_called()

    def test_import_precedes_environment_and_cleanup_follows_selfcheck(self):
        install = LAUNCH['install_inputs']
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            home = Path(directory)
            events = []
            def extract(*args):
                events.append('import')
                return {'kind': 'resources', 'id': 'r'}
            def paths(*args):
                events.append('cpu')
                return home, home
            def optional(*args):
                events.append(args[-1])
            def probe(*args, **kwargs):
                events.append('probe')
                self.assertFalse((home / 'manifests/active.json').exists())
            def cleanup(*args):
                events.append('cleanup')
                self.assertEqual(portable.read_json(home / 'manifests/active.json')['resources'], 'r')
            with patch.dict(install.__globals__, HOME=home, check_gallery_inputs=lambda *args: events.append('check'),
                            install_package=extract, validate_combination=lambda *args, **kwargs: None,
                            activate_paths=paths, prepare_optional=optional,
                            prepare_models=lambda *args: events.append('models'), cleanup_installation=cleanup):
                with patch.object(subprocess, 'run', side_effect=probe):
                    self.assertEqual(install({'app': 'a', 'resources': None}, [Path('gallery.zip')], 'gpu'), 0)
            self.assertEqual(events, ['check', 'import', 'cpu', 'gpu', 'models', 'probe', 'cleanup'])

    def test_later_gpu_install_needs_no_zip_and_gpu_failure_keeps_downloads(self):
        install = LAUNCH['install_inputs']
        for success in (True, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                home = Path(directory)
                extract, cleanup = Mock(), Mock()
                optional = Mock(return_value=success)
                with patch.dict(install.__globals__, HOME=home, install_package=extract,
                                validate_combination=lambda *args, **kwargs: None,
                                activate_paths=lambda *args: (home, home), prepare_optional=optional,
                                prepare_models=lambda *args: None, cleanup_installation=cleanup):
                    with patch.object(subprocess, 'run'):
                        code = install({'app': 'a', 'resources': 'r'}, [], 'gpu')
                self.assertEqual(code, 0 if success else 1)
                extract.assert_not_called()
                optional.assert_called_once_with(home, home, 'gpu')
                self.assertEqual(cleanup.call_count, 1 if success else 0)
                self.assertEqual(portable.read_json(home / 'manifests/active.json')['resources'], 'r')

    def test_cleanup_preserves_unrelated_downloads_models_reports_and_cache(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            home = Path(directory)
            app = home / 'app/a'
            manifest = app / 'manifests/runtime.json'
            portable.atomic_json(manifest, {'id': 'cpu', 'wheels': [{'filename': 'known.whl'}]})
            portable.atomic_json(home / 'runtime/cpu/ready.json', {'lock_sha256': portable.digest(manifest)})
            remove = ['work/downloads/known.whl', 'work/downloads/known.whl.part',
                      'imports/gallery.zip', 'work/proposed.json']
            preserve = ['work/downloads/unrelated.whl', 'work/cache/keep', 'work/temp/user-file',
                        'user/reports/keep', 'models/keep', 'original.zip', 'imports/other.zip']
            for name in remove + preserve:
                path = home / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'keep')
            portable.cleanup_installation(home, app, [home / 'imports/gallery.zip', home / 'original.zip'])
            for name in remove:
                self.assertFalse((home / name).exists(), name)
            for name in preserve:
                self.assertEqual((home / name).read_bytes(), b'keep', name)

    def test_redirected_progress_is_throttled(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(portable.time, 'monotonic', return_value=100) as clock:
            meter = portable.Progress('test', 32 * 1024**2)
            for second in range(1, 31):
                clock.return_value = 100 + second
                meter.update(second * 1024**2)
            meter.update(32 * 1024**2, force=True)
            meter.close()
        self.assertEqual(len(output.getvalue().splitlines()), 4)
        self.assertIn('100.0%', output.getvalue())


if __name__ == '__main__':
    unittest.main()
