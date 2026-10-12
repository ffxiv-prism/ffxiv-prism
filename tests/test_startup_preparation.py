"""Selfcheck must not repeat preparation or mutate the prepared runtime."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class StartupPreparationTests(unittest.TestCase):
    def setUp(self):
        with patch.object(sys, 'path', [str(ROOT / 'bootstrap'), *sys.path]):
            self.launch = runpy.run_path(str(ROOT / 'bootstrap/launch.py'))

    def test_probe_configures_process_without_rewriting_runtime(self):
        fn = self.launch['activate_paths']
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            app = home / 'app/v-test'
            (app / 'manifests').mkdir(parents=True)
            (app / 'manifests/runtime.json').write_text(json.dumps({'id': 'cpu-test'}))
            pth = home / 'runtime/python/python312._pth'
            pth.parent.mkdir(parents=True)
            pth.write_bytes(b'prepared paths\n')
            before = pth.stat().st_mtime_ns
            with patch.dict(fn.__globals__, HOME=home), patch.dict(os.environ), \
                 patch.object(sys, 'path', list(sys.path)), \
                 patch.dict(fn.__globals__, prepare_runtime=unittest.mock.Mock(),
                            expose_optional=unittest.mock.Mock(), status=unittest.mock.Mock()):
                result = fn({'app': 'v-test', 'resources': 'gallery-test'},
                            prepare=False, announce=False, persist=False)
                self.assertEqual(result, (app, home / 'resources/gallery-test'))
                self.assertIn(str(app / 'src'), sys.path)
                self.assertEqual(os.environ['PRISM_RESOURCE_ROOT'], str(result[1].resolve()))
                for name in ('prepare_runtime', 'expose_optional', 'status'):
                    fn.__globals__[name].assert_not_called()
            self.assertEqual(pth.read_bytes(), b'prepared paths\n')
            self.assertEqual(pth.stat().st_mtime_ns, before)

    def test_selfcheck_preserves_failure_and_uses_prepared_app(self):
        fn = self.launch['selfcheck']
        app = ROOT / 'work/prepared-app'
        failure = subprocess.CalledProcessError(1, 'probe')
        with patch.dict(fn.__globals__, activate_paths=unittest.mock.Mock()) as globals_, \
             patch.object(subprocess, 'run', side_effect=failure) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                fn(app)
            globals_['activate_paths'].assert_not_called()
            self.assertEqual(run.call_args.kwargs['cwd'], app)
            self.assertTrue(run.call_args.kwargs['check'])
            self.assertEqual(run.call_args.kwargs['timeout'], 180)


if __name__ == '__main__':
    unittest.main()
