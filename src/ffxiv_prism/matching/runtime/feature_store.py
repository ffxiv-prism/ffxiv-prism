"""Disposable SQLite read-through pack for small content-addressed feature files.

Original files remain authoritative. A batch checks their stat identities before
using packed bytes; missing/changed files cannot be hidden by this accelerator.
"""
import hashlib
import os
from pathlib import Path
import sqlite3

from .session import current_session, timed


@timed("feature_batch_read")
def read_feature_batch(directory, keys, suffix):
    from ffxiv_prism.paths import CACHE_DIR
    directory = Path(directory)
    session = current_session()
    identities = {}
    for key in dict.fromkeys(keys):
        try:
            stat = (directory / (key + suffix)).stat()
            identities[key] = str((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
        except OSError:
            pass
    found = {}
    connection = None
    try:
        if session is None or session.enabled:
            root = CACHE_DIR / "feature-packs-v1"
            root.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha256((os.path.normcase(os.path.abspath(directory)) + suffix).encode()).hexdigest()
            connection = sqlite3.connect(root / (name + ".sqlite3"), timeout=1)
            connection.execute("CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, identity TEXT, data BLOB)")
            unique = list(identities)
            for start in range(0, len(unique), 400):
                batch = unique[start:start + 400]
                rows = connection.execute("SELECT key, identity, data FROM features WHERE key IN (" + ",".join("?" for _ in batch) + ")", batch)
                for key, identity, data in rows:
                    if identity == identities[key]:
                        found[key] = data
                        if session:
                            session.count("feature_pack_hits")
    except (OSError, sqlite3.Error):
        if connection:
            connection.close()
            connection = None
    updates = []
    try:
        for key, identity in identities.items():
            if key in found:
                continue
            try:
                data = (directory / (key + suffix)).read_bytes()
            except OSError:
                continue
            found[key] = data
            updates.append((key, identity, data))
            if session:
                session.count("feature_file_reads")
        if connection and updates:
            try:
                with connection:
                    connection.executemany("INSERT OR REPLACE INTO features VALUES (?, ?, ?)", updates)
            except sqlite3.Error:
                pass  # Read-only/full/busy accelerator never blocks inference.
    finally:
        if connection:
            connection.close()
    return found
