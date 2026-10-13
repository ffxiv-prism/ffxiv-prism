"""Unify source gallery paths with the portable layout; keep recoverable backups.

Run without --apply to inspect the plan. Historical experiment snapshots are
left intact; the path resolver understands their old equipment image paths.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.release_layout import index_names, rewrite_paths


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def migrate(root: Path, *, apply: bool = False, indexes_only: bool = False):
    root = root.resolve()
    selection_path = root / "data/current_best_selection.json"
    selection = json.loads(selection_path.read_text("utf-8"))
    indexes = index_names(selection)
    # Resolve and validate every path before writing anything.
    def inside(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Path escapes project: {relative}")
        return path

    images = {}
    if not indexes_only:
        old_images = inside("data/raw/eorzea_collection/images")
        for source in sorted(old_images.rglob("*")):
            if not source.is_file() or source.name == ".gitkeep":
                continue
            if not source.resolve().is_relative_to(old_images):
                raise ValueError(f"Image escapes gallery: {source}")
            destination = inside("data/images/" + source.name)
            previous = images.get(source.name.casefold())
            if previous and digest(previous[0]) != digest(source):
                raise ValueError(f"Conflicting flat image name: {source.name}")
            if destination.exists() and digest(destination) != digest(source):
                raise ValueError(f"Existing image differs: {destination}")
            images[source.name.casefold()] = (source, destination)
    for old, new in indexes.items():
        source, destination = inside(old), inside(new)
        for name in ("embeddings.npy", "metadata.jsonl", "summary.json"):
            if not (source / name).is_file():
                raise FileNotFoundError(source / name)
    plan = {"images": len(images), "indexes": indexes, "applied": apply}
    if not apply:
        return plan

    backup = inside("work/gallery-layout-backup/" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    backup.mkdir(parents=True)
    def save(path, contents):
        if path.exists() and path.read_bytes() == contents:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            original = backup / path.relative_to(root)
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, original)
        temporary = path.with_name(path.name + ".layout-tmp")
        temporary.write_bytes(contents)
        temporary.replace(path)

    def rewrite(source, destination):
        if source.suffix == ".jsonl":
            contents = "".join(json.dumps(rewrite_paths(json.loads(line), indexes), ensure_ascii=False) + "\n"
                               for line in source.read_text("utf-8").splitlines() if line.strip())
        else:
            contents = json.dumps(rewrite_paths(json.loads(source.read_text("utf-8")), indexes),
                                  ensure_ascii=False, indent=2) + "\n"
        save(destination, contents.encode("utf-8"))

    for source, destination in images.values():
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for old, new in indexes.items():
        source, destination = inside(old), inside(new)
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("embeddings.npy", "metadata.jsonl", "summary.json"):
            path = source / name
            if path.suffix in (".json", ".jsonl"):
                rewrite(path, destination / name)
            elif source != destination:
                save(destination / name, path.read_bytes())

    if not indexes_only:
        metadata = list(inside("data/raw/eorzea_collection/metadata").glob("*.json*"))
        metadata += list(inside("data/processed").glob("eorzea*.json*"))
        metadata += [inside("data/local_features/catalogue.json")]
        for path in metadata:
            if path.is_file() and path.suffix in (".json", ".jsonl"):
                rewrite(path, path)
        database = inside("data/processed/eorzea.sqlite")
        if database.exists():
            original = backup / database.relative_to(root)
            original.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(database)) as db, db:
                with closing(sqlite3.connect(original)) as copy:
                    db.backup(copy)
                db.create_function("canonical_image", 1, lambda value: rewrite_paths(value, indexes))
                for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                    quoted = '"' + table.replace('"', '""') + '"'
                    columns = {row[1] for row in db.execute(f"PRAGMA table_info({quoted})")}
                    if "local_path" in columns:
                        db.execute(f"UPDATE {quoted} SET local_path=canonical_image(local_path)")
    # Publish the selection only after all active branches have been copied.
    rewrite(selection_path, selection_path)
    for path in inside("manifests/galleries").glob("*.json"):
        # Published historical snapshots retain their own directory mappings.
        if json.loads(path.read_text("utf-8")).get("candidate_indexes") == selection["candidate_indexes"]:
            rewrite(path, path)
    manifest_path = inside("data/current_best_assets.json")
    if manifest_path.exists():
        manifest = rewrite_paths(json.loads(manifest_path.read_text("utf-8")), indexes)
        manifest["user_downloaded_data"] = "data/images"
        for row in manifest["files"]:
            path = inside(row["path"])
            row.update(size=path.stat().st_size, sha256=digest(path))
        save(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    plan["backup"] = backup.relative_to(root).as_posix()
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--indexes-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(migrate(ROOT, apply=args.apply, indexes_only=args.indexes_only), ensure_ascii=False, indent=2))
