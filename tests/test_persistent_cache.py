from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ffxiv_prism.matching.runtime.session import InferenceSession, frozen_resource
from ffxiv_prism.matching.runtime.feature_store import read_feature_batch


class PersistentCacheTests(unittest.TestCase):
    def test_models_survive_success_but_not_failure(self):
        models = OrderedDict()
        with InferenceSession(shared_models=models) as first:
            value = frozen_resource(('encoder', 'model', 'cpu'), object)
        with InferenceSession(shared_models=models) as second:
            self.assertIs(value, frozen_resource(('encoder', 'model', 'cpu'), object))
            self.assertEqual(second.stats, {'model_hits': 1})
        with InferenceSession(enabled=False, shared_models=models):
            self.assertIsNot(value, frozen_resource(('encoder', 'model', 'cpu'), object))
        self.assertTrue(models)
        with self.assertRaises(RuntimeError):
            with InferenceSession(shared_models=models):
                raise RuntimeError('failed')
        self.assertFalse(models)

    def test_pack_hits_changes_deletions_and_disabled_mode(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            source.mkdir()
            a, b = source / 'a.json', source / 'b.json'
            a.write_bytes(b'first')
            b.write_bytes(b'second')
            with patch('ffxiv_prism.paths.CACHE_DIR', root / 'cache'):
                with InferenceSession() as session:
                    self.assertEqual(read_feature_batch(source, ['b','a','a'], '.json'), {'a':b'first','b':b'second'})
                    self.assertEqual(session.stats['feature_file_reads'], 2)
                with InferenceSession() as session:
                    self.assertEqual(read_feature_batch(source, ['a','b'], '.json')['a'], b'first')
                    self.assertEqual(session.stats['feature_pack_hits'], 2)
                a.write_bytes(b'changed')
                b.unlink()
                with InferenceSession() as session:
                    self.assertEqual(read_feature_batch(source, ['a','b'], '.json'), {'a':b'changed'})
                    self.assertEqual(session.stats['feature_file_reads'], 1)
                with InferenceSession(enabled=False) as session:
                    self.assertEqual(read_feature_batch(source, ['a'], '.json'), {'a':b'changed'})
                    self.assertNotIn('feature_pack_hits', session.stats)

    def test_corrupt_pack_falls_back_to_source(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'a.json').write_bytes(b'valid')
            with patch('ffxiv_prism.paths.CACHE_DIR',root/'cache'):
                read_feature_batch(root,['a'],'.json')
                pack=next((root/'cache/feature-packs-v1').glob('*.sqlite3'))
                pack.write_bytes(b'broken')
                self.assertEqual(read_feature_batch(root,['a'],'.json'), {'a':b'valid'})


class WorkerRegistryTests(unittest.TestCase):
    def test_success_reuses_process_and_shutdown_releases_it(self):
        import io
        from unittest.mock import Mock
        from scripts import web_app
        with TemporaryDirectory() as folder, patch.object(web_app, 'HOME_ROOT', Path(folder)), patch.dict('os.environ', {'PRISM_ONESHOT':'0'}):
            process = Mock()
            process.poll.return_value = None
            process.stdin = io.StringIO()
            process.stdout = io.StringIO('PERFORMANCE {"elapsed_seconds":1}\nWORKER_RESULT {"id":"first","return_code":0}\nWORKER_RESULT {"id":"second","return_code":0}\n')
            registry = web_app.JobRegistry()
            with patch.object(web_app.subprocess, 'Popen', return_value=process) as popen:
                for name in ('first','second'):
                    job = registry.reserve(kind='recognition',title=name,job_id=name,device='cpu')
                    registry._run(job.id,['python','recognize_glamour.py','image.png'],{})
                    self.assertEqual(job.status,'succeeded')
                self.assertEqual(popen.call_count,1)
                self.assertFalse(process.stdout.closed)
                self.assertIn('image.png',process.stdin.getvalue())
                registry.close_worker()
                process.terminate.assert_called_once()
                self.assertTrue(process.stdout.closed)

    def test_unexpected_clean_eof_is_failure_and_worker_is_discarded(self):
        import io
        from unittest.mock import Mock
        from scripts import web_app
        with TemporaryDirectory() as folder, patch.object(web_app, 'HOME_ROOT', Path(folder)), patch.dict('os.environ', {'PRISM_ONESHOT':'0'}):
            process = Mock()
            process.poll.return_value = 0
            process.wait.return_value = 0
            process.stdin = io.StringIO()
            process.stdout = io.StringIO('incomplete log\n')
            registry = web_app.JobRegistry()
            with patch.object(web_app.subprocess, 'Popen', return_value=process):
                job = registry.reserve(kind='recognition',title='test',device='cpu')
                registry._run(job.id,['python','recognize_glamour.py','image.png'],{})
                self.assertEqual(job.status,'failed')
                self.assertIsNone(registry._worker)


if __name__ == '__main__':
    unittest.main()
