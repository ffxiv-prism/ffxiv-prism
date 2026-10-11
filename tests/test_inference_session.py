from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ffxiv_prism.matching.runtime.session import (
    InferenceSession, current_session, frozen_resource, read_image_bytes,
    read_metadata, load_index_vectors, timed,
)
from ffxiv_prism.matching.runtime.encoders import search_crops


class InferenceSessionTests(unittest.TestCase):
    def test_models_are_bounded_isolated_and_cleared_on_failure(self):
        calls = []
        def factory():
            calls.append(1)
            return object()
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with InferenceSession(model_limit=1) as session:
                cpu = frozen_resource(("style", "m", "cpu"), factory)
                self.assertIs(cpu, frozen_resource(("style", "m", "cpu"), factory))
                self.assertIsNot(cpu, frozen_resource(("style", "m", "cuda"), factory))
                self.assertIsNot(cpu, frozen_resource(("style", "m", "cpu"), factory))
                self.assertEqual(len(session.models), 1)
                raise RuntimeError("stop")
        self.assertEqual(len(calls), 3)
        self.assertFalse(session.models)
        self.assertIsNone(current_session())
        with InferenceSession():
            self.assertIsNot(cpu, frozen_resource(("style", "m", "cpu"), factory))

    def test_disabled_cache_and_nested_session(self):
        with InferenceSession() as outer:
            with InferenceSession(enabled=False) as inner:
                a = frozen_resource("x", object)
                self.assertIsNot(a, frozen_resource("x", object))
                self.assertFalse(inner.models)
            self.assertIs(current_session(), outer)

    def test_large_sequential_scans_do_not_thrash_admitted_files(self):
        with tempfile.TemporaryDirectory() as folder, InferenceSession(file_bytes=4) as session:
            paths = [Path(folder) / str(i) for i in range(3)]
            for path in paths:
                path.write_bytes(b"data")
            for _ in range(2):
                for path in paths:
                    read_image_bytes(path)
            self.assertEqual(session.stats["image_hits"], 1)
            self.assertEqual(session.stats["image_reads"], 5)

    def test_files_invalidate_on_change_and_respect_byte_limit(self):
        with tempfile.TemporaryDirectory() as folder, InferenceSession(file_bytes=4) as session:
            a, b = Path(folder) / "a", Path(folder) / "b"
            a.write_bytes(b"1234")
            b.write_bytes(b"5678")
            self.assertEqual(read_image_bytes(a), b"1234")
            self.assertEqual(read_image_bytes(a), b"1234")
            self.assertEqual(session.stats["image_hits"], 1)
            read_image_bytes(b)
            self.assertEqual(session.file_bytes, 4)
            self.assertEqual(len(session.files), 1)
            a.write_bytes(b"updated")
            self.assertEqual(read_image_bytes(a), b"updated")
            self.assertLessEqual(session.file_bytes, 4)

    def test_metadata_and_mappings_are_reused_but_replacement_invalidates(self):
        with tempfile.TemporaryDirectory() as folder, InferenceSession():
            metadata = Path(folder) / "metadata.jsonl"
            metadata.write_text('{"piece_id":1}\n', encoding="utf-8")
            rows = read_metadata(metadata)
            self.assertIs(rows, read_metadata(metadata))
            metadata.write_text('{"piece_id":123}\n', encoding="utf-8")
            self.assertEqual(read_metadata(metadata), [{"piece_id": 123}])
            path = Path(folder) / "embeddings.npy"
            np.save(path, np.eye(2, dtype=np.float32))
            vectors = load_index_vectors(path)
            self.assertIs(vectors, load_index_vectors(path))
            self.assertFalse(vectors.flags.writeable)
            del vectors  # Release the caller's mapping before Windows removes the temp directory.

    def test_nested_timings_do_not_double_count_self_time(self):
        @timed("child")
        def child():
            return 7
        with patch("ffxiv_prism.matching.runtime.session.perf_counter", side_effect=[0, 1, 2, 4, 8, 9]):
            with InferenceSession() as session:
                with session.stage("parent"):
                    self.assertEqual(child(), 7)
                result = session.snapshot()
        self.assertEqual(result["stages"]["parent"]["seconds"], 7)
        self.assertEqual(result["stages"]["parent"]["self_seconds"], 5)
        self.assertEqual(result["stages"]["child"]["seconds"], 2)

    def test_multiple_query_crops_copy_each_candidate_route_only_once(self):
        class CountingArray(np.ndarray):
            reads = 0
            def __getitem__(self, key):
                if isinstance(key, np.ndarray):
                    self.reads += 1
                return super().__getitem__(key)
        vectors = np.eye(2, dtype=np.float32).view(CountingArray)
        metadata = [dict(slot="HEAD", pieces=[dict(piece_id=i)], image_row_id=i,
                         local_path=str(i), source_image_url=str(i)) for i in range(2)]
        crops = {"query": [dict(path="a", slot="HEAD"), dict(path="b", slot="HEAD")]}
        result = search_crops(metadata, vectors, crops,
                              {"a": np.array([1., 0.], dtype=np.float32),
                               "b": np.array([0., 1.], dtype=np.float32)}, 2)
        self.assertEqual(vectors.reads, 1)
        self.assertEqual([r["piece_id"] for r in result["a"]], [0, 1])
        self.assertEqual([r["piece_id"] for r in result["b"]], [1, 0])


if __name__ == "__main__":
    unittest.main()
