"""Worker imports must work without Python's implicit script-directory path."""
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkerEntrypointTests(unittest.TestCase):
    def test_worker_imports_sibling_in_isolated_python(self):
        with tempfile.TemporaryDirectory() as directory:
            scripts = Path(directory) / 'scripts'
            scripts.mkdir()
            worker = scripts / 'recognition_worker.py'
            shutil.copyfile(ROOT / 'scripts/recognition_worker.py', worker)
            # Keep this regression independent of installed ML dependencies.
            (scripts / 'recognize_glamour.py').write_text(
                'def main(*args, **kwargs):\n    raise AssertionError("EOF must not run inference")\n')
            result = subprocess.run([sys.executable, '-I', '-B', str(worker)],
                                    input='', capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_selfcheck_rejects_broken_worker(self):
        sys.path.insert(0, str(ROOT / 'bootstrap'))
        try:
            launch = runpy.run_path(str(ROOT / 'bootstrap/launch.py'))
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            (app / 'scripts').mkdir()
            (app / 'scripts/recognition_worker.py').write_text('raise SystemExit(23)\n')
            with self.assertRaises(subprocess.CalledProcessError) as error:
                launch['check_worker_entrypoint'](app)
            self.assertEqual(error.exception.returncode, 23)

if __name__ == '__main__':
    unittest.main()
