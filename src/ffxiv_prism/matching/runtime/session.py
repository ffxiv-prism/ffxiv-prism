"""Bounded, request-local reuse and wall-clock diagnostics for inference.

An optional worker-owned model pool survives successful requests; file caches
remain request-local. Checkpoint-mutated models must never enter the frozen-model
cache. File keys include identity and modification data;
existing content-derived feature cache keys are unchanged.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path
from time import perf_counter

_ACTIVE = ContextVar("prism_inference_session", default=None)


class InferenceSession:
    def __init__(self, *, enabled=True, model_limit=2, file_bytes=256 * 1024**2, shared_models=None):
        if model_limit < 1 or file_bytes < 0:
            raise ValueError("Cache limits must be non-negative and allow at least one model")
        self.enabled = enabled
        self.model_limit = model_limit
        self.file_limit = file_bytes
        self.shared_models = shared_models if enabled else None
        self.models = self.shared_models if self.shared_models is not None else OrderedDict()
        self.files = OrderedDict()
        self.file_bytes = 0
        self.stats = {}
        self.timings = {}
        self.stack = []
        self.started = perf_counter()

    def __enter__(self):
        self.token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc):
        if self.shared_models is None or (exc and exc[0] is not None):
            self.models.clear()
        self.files.clear()
        self.file_bytes = 0
        _ACTIVE.reset(self.token)

    def count(self, name):
        self.stats[name] = self.stats.get(name, 0) + 1

    @contextmanager
    def stage(self, name):
        frame = [perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = perf_counter() - frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            row = self.timings.setdefault(name, dict(calls=0, seconds=0.0, self_seconds=0.0))
            row["calls"] += 1
            row["seconds"] += elapsed
            row["self_seconds"] += elapsed - frame[1]
            if name.startswith("slot.") or name in {"pose_estimate", "report_export"}:
                print(f"TIMING stage={name} seconds={elapsed:.3f}", flush=True)

    def model(self, key, factory):
        if self.enabled and key in self.models:
            self.count("model_hits")
            self.models.move_to_end(key)
            return self.models[key]
        self.count("model_loads")
        # Evict before allocating, rather than transiently retaining an extra model.
        if self.enabled and len(self.models) >= self.model_limit:
            self.models.popitem(last=False)
            self.count("model_evictions")
        with self.stage("model_load"):
            value = factory()
        if self.enabled:
            self.models[key] = value
        return value

    def file(self, kind, path, loader, size=None):
        path = Path(path)
        stat = path.stat()
        # resolve() performs extra filesystem queries for every ancestor on
        # Windows; lexical normalization is sufficient for a request-local key.
        key = (kind, os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns,
               stat.st_ctime_ns, stat.st_ino)
        if self.enabled and key in self.files:
            self.count(kind + "_hits")
            self.files.move_to_end(key)
            return self.files[key][0]
        self.count(kind + "_reads")
        with self.stage("cache_read." + kind):
            value = loader(path)
        cost = size(value) if size else stat.st_size
        # Keep an admitted working set. Repeated full catalogue scans larger
        # than the budget would otherwise evict every entry just before reuse.
        if self.enabled and self.file_bytes + cost <= self.file_limit and len(self.files) < 8192:
            self.files[key] = (value, cost)
            self.file_bytes += cost
        return value

    def snapshot(self):
        return dict(
            version=1, cache_enabled=self.enabled,
            elapsed_seconds=perf_counter() - self.started,
            timing_semantics="wall-clock; seconds includes children; self_seconds excludes them; Python startup excluded",
            stages={name: dict(row) for name, row in self.timings.items()},
            cache=dict(self.stats),
            limits=dict(frozen_models=self.model_limit, file_bytes=self.file_limit),
        )


def current_session():
    return _ACTIVE.get()


def timed(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            session = current_session()
            if session is None:
                return function(*args, **kwargs)
            with session.stage(name):
                return function(*args, **kwargs)
        return call
    return decorate


def frozen_resource(key, factory):
    session = current_session()
    return session.model(key, factory) if session else factory()


def read_image_bytes(path):
    session = current_session()
    return session.file("image", path, Path.read_bytes) if session else Path(path).read_bytes()


def read_metadata(path):
    from ffxiv_prism.jsonl import read_jsonl
    session = current_session()
    # JSON objects occupy substantially more memory than their serialized bytes.
    return session.file("metadata", path, read_jsonl, size=lambda rows: Path(path).stat().st_size * 10) if session else read_jsonl(path)


def load_index_vectors(path, *, mmap_mode="r"):
    import numpy as np
    def load(path):
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    session = current_session()
    # Mappings do not allocate the whole index on the Python heap.
    return session.file("index", path, load, size=lambda value: 4096) if session else load(path)
