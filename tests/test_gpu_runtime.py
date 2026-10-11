import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bootstrap'))
import gpu_runtime as gpu
import portable
from scripts import portable_upgrade, build_portable_release, web_app


class OptionalGpuTests(unittest.TestCase):
    def fixture(self, home):
        app = home / 'app/v6'
        (app / 'manifests').mkdir(parents=True)
        (home / 'runtime/python').mkdir(parents=True)
        (home / 'runtime/python/python.exe').write_bytes(b'embedded')
        (home / 'runtime/python/python312._pth').write_text('CPU original')
        (home / 'work/downloads').mkdir(parents=True)
        for kind, filename in [('cpu', 'runtime.json'), ('cu130', 'runtime-cu130.json')]:
            wheel = home / f'work/downloads/{kind}.whl'
            with zipfile.ZipFile(wheel, 'w') as z:
                z.writestr('torch/version.py', kind)
            row = dict(filename=wheel.name, size=wheel.stat().st_size,
                       sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), urls=[])
            (app / 'manifests' / filename).write_text(json.dumps(dict(id=kind, wheels=[row])))
        return app

    def test_append_gpu_reuses_cpu_and_repeat_install_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            app = self.fixture(home)
            cpu = portable.prepare_runtime(home, app)
            before = (cpu / 'torch/version.py').stat().st_mtime_ns
            gpu.prepare_optional(home, app, 'cpu')
            with patch.object(gpu, 'detect_variant', return_value='cu130'), patch.object(gpu, 'probe'):
                self.assertTrue(gpu.prepare_optional(home, app, 'gpu'))
                with patch.object(portable, 'download', side_effect=AssertionError('redownload')):
                    portable.prepare_runtime(home, app)
                    self.assertTrue(gpu.prepare_optional(home, app, 'gpu'))
            self.assertEqual((cpu / 'torch/version.py').read_text(), 'cpu')
            self.assertEqual((cpu / 'torch/version.py').stat().st_mtime_ns, before)
            self.assertEqual((home / 'runtime/python/python312._pth').read_text(), 'CPU original')
            self.assertEqual(gpu.settings(home)['gpu'], 'cu130')
            python = gpu.worker_python(home, app, 'cu130')
            self.assertTrue(python.is_relative_to(home))
            paths = python.with_name('python312._pth').read_text().splitlines()
            self.assertIn('../site-packages', paths)
            gpu.prepare_optional(home, app, 'cpu')
            self.assertEqual(gpu.settings(home)['gpu'], 'cu130')

    def test_gpu_failure_keeps_cpu_and_does_not_activate_failed_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            app = self.fixture(home)
            cpu = portable.prepare_runtime(home, app)
            with patch.object(gpu, 'detect_variant', return_value='cu130'), patch.object(gpu, 'probe', side_effect=RuntimeError('DLL')):
                self.assertFalse(gpu.prepare_optional(home, app, 'gpu'))
            self.assertEqual(gpu.settings(home), {})
            self.assertEqual((cpu / 'torch/version.py').read_text(), 'cpu')

    def test_selection_hardware_and_driver(self):
        self.assertEqual(gpu.choose_variant('8.6, 580.88'), 'cu130')
        self.assertEqual(gpu.choose_variant('6.1, 580.88'), 'cu126')
        self.assertEqual(gpu.choose_variant('8.6, 560.76'), 'cu126')
        self.assertEqual(gpu.choose_variant('12.0, 580.88'), 'cu130')
        for value in ('12.0, 560.76', '3.5, 580.88', '8.6, 550.0'):
            with self.assertRaises(ValueError):
                gpu.choose_variant(value)

    def test_gpu_probe_failure_does_not_disable_cpu(self):
        web_app.execution_runtime_status.cache_clear()
        with patch.dict(os.environ, {'PRISM_GPU_PYTHON': 'gpu-python'}), patch.object(web_app, '_probe_execution_runtime', return_value={'cpu': {'available': True}}), patch.object(gpu, 'probe', side_effect=RuntimeError('DLL')):
            result = web_app.execution_runtime_status('cpu-python')
        web_app.execution_runtime_status.cache_clear()
        self.assertTrue(result['cpu']['available'])
        self.assertFalse(result['gpu']['available'])

    def test_recognition_routes_to_separate_interpreters(self):
        with patch.dict(os.environ, {'PRISM_GPU_PYTHON': 'gpu-python'}):
            for device, expected in [('cpu', 'cpu-python'), ('cuda', 'gpu-python')]:
                command = web_app.build_recognition_command(python_executable='cpu-python', image_path=Path('input.png'), output_dir=Path('out'), slots=['HEAD'], top_k=10, gender=None, profile='current-best', device=device)
                self.assertEqual(command[0], expected)

    def test_update_discovers_parent_and_bat_never_prompts_for_path(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.fixture(home)
            (home / 'manifests').mkdir()
            (home / 'manifests/active.json').write_text('{}')
            bundle = home / 'FFXIV-Prism-Update'
            bundle.mkdir()
            self.assertEqual(portable_upgrade.discover_home(bundle), home.resolve())
            with self.assertRaisesRegex(ValueError, '根目录'):
                portable_upgrade.discover_home(home)
        bat = build_portable_release.upgrade_bat()
        self.assertNotIn('set /p', bat)
        self.assertIn('%~dp0..', bat)


if __name__ == '__main__':
    unittest.main()
