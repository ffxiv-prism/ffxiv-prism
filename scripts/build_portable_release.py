"""Build a small portable app and a separate, signed equipment catalogue ZIP."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "bootstrap")]
from portable import atomic_json, digest, verify_signature
from ffxiv_prism.jsonl import read_jsonl

VERSION = "v0.2.0-beta.1"
RESOURCE_ID = "gallery-20260926-head-v2"
CONTRACT = "prism-current-best-eye-v2"

def sign(path):
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(ROOT / "scripts/sign_portable.ps1"), "-Manifest", str(path),
                    "-KeyPath", str(ROOT / "work/signing/portable-private.xml"),
                    "-PublicPath", str(ROOT / "bootstrap/public-key.json")], check=True, capture_output=True)

def pack(files, destination, manifest):
    manifest["schema"] = 1
    manifest["files"] = [{"path": name, "size": path.stat().st_size, "sha256": digest(path)}
                         for name, path in sorted(files.items())]
    staging = ROOT / "work/build" / manifest["id"]
    staging.mkdir(parents=True, exist_ok=True)
    metadata = staging / "package.json"
    atomic_json(metadata, manifest)
    sign(metadata)
    verify_signature(metadata.read_bytes(), metadata.with_suffix(".sig").read_bytes(), json.loads((ROOT / "bootstrap/public-key.json").read_text("utf-8")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".partial")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as archive:
        archive.write(metadata, "package.json")
        archive.write(metadata.with_suffix(".sig"), "package.sig")
        for number, (name, path) in enumerate(sorted(files.items()), 1):
            archive.write(path, "payload/" + name)
            if number % 5000 == 0:
                print(f"{manifest['id']}: {number}/{len(files)}", flush=True)
    for attempt in range(10):
        try:
            temp.replace(destination)
            break
        except PermissionError:
            if attempt == 9:
                # Windows scanners may allow reads while briefly denying rename.
                # The release is not uploaded until its complete hash is recorded.
                shutil.copyfile(temp, destination)
                break
            time.sleep(.5)
    (destination.with_suffix(".zip.sha256")).write_text(digest(destination) + "  " + destination.name + "\n", "utf-8")
    print(f"Built {destination.name}: {destination.stat().st_size/1024**2:.1f} MiB; {len(files)} files", flush=True)
    return manifest

def gallery():
    selection = json.loads((ROOT / "data/current_best_selection.json").read_text("utf-8"))
    files = {}
    image_names, piece_ids = set(), set()
    for relative in set(selection["candidate_indexes"].values()):
        folder = ROOT / relative
        for path in folder.iterdir():
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = path
        for row in read_jsonl(folder / "metadata.jsonl"):
            name = row["local_path"].replace("\\", "/")
            if not name.startswith("data/raw/eorzea_collection/images/"):
                raise ValueError(f"Non-equipment image in catalogue: {name}")
            image_names.add(name)
            piece_ids.update(int(p["piece_id"]) for p in row["pieces"])
    for name in image_names:
        files[name] = ROOT / name
    asset_manifest = json.loads((ROOT / "data/current_best_assets.json").read_text("utf-8"))
    for row in asset_manifest["files"]:
        if row["path"].startswith(("data/models/", "data/processed/")):
            files[row["path"]] = ROOT / row["path"]
    for relative in ("data/current_best_selection.json", "data/raw/eorzea_collection/metadata/gear_pieces_raw.jsonl",
                     "data/raw/game_data/cafemaker_item_names_for_gear_pieces_raw.jsonl", "data/raw/game_data/cafemaker_item_names_raw.jsonl"):
        files[relative] = ROOT / relative
    for path in (ROOT / "data/local_features").rglob("*"):
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
    # Read-only SQLite backup avoids copying an inconsistent live database/WAL.
    build = ROOT / "work/build/gallery"
    build.mkdir(parents=True, exist_ok=True)
    database = build / "eorzea.sqlite"
    with sqlite3.connect(f"file:{(ROOT / 'data/processed/eorzea.sqlite').as_posix()}?mode=ro", uri=True) as source, sqlite3.connect(database) as target:
        tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if any("glamour" in name or "player" in name for name in tables):
            raise ValueError("Player tables must not be included in the equipment package")
        source.backup(target)
    files["data/processed/eorzea.sqlite"] = database
    info = {"schema": 1, "id": RESOURCE_ID, "updated_at": selection["candidate_indexes_updated_at"],
            "date_basis": "active searchable candidate snapshot; excludes later unindexed images",
            "image_count": len(image_names), "piece_count": len(piece_ids), "compatibility": CONTRACT}
    atomic_json(build / "gallery.json", info)
    files["gallery.json"] = build / "gallery.json"
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing catalogue assets: {missing[:20]}")
    return pack(files, ROOT / f"FFXIV-Prism-{RESOURCE_ID}.zip", {"kind": "resources", "id": RESOURCE_ID, "compatibility": CONTRACT, "gallery": info})

def app():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    atomic_json(ROOT / "manifests/release.json", {"version": VERSION, "code_commit": commit, "prerelease": True})
    files = {}
    for folder in ("src", "web", "manifests"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc"}:
                files[path.relative_to(ROOT).as_posix()] = path
    for name in ("scripts/recognize_glamour.py", "scripts/web_app.py", "scripts/open_report_directory.ps1", "README.md", "docs/portable_release_zh.md"):
        files[name] = ROOT / name
    manifest = pack(files, ROOT / "dist" / f"FFXIV-Prism-{VERSION}-app.zip", {"kind": "app", "id": VERSION, "resource_compatibility": [CONTRACT]})
    source = ROOT / "dist" / f"FFXIV-Prism-{VERSION}-app.zip"
    output = ROOT / "dist" / f"FFXIV-Prism-{VERSION}-windows.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        prefix = "FFXIV-Prism/"
        for file in (ROOT / "bootstrap").glob("*"):
            if file.is_file():
                archive.write(file, prefix + "bootstrap/" + file.name)
        with zipfile.ZipFile(source) as package:
            for name in package.namelist():
                target = name.removeprefix("payload/")
                archive.writestr(prefix + "app/" + VERSION + "/" + target, package.read(name))
        with zipfile.ZipFile(ROOT / "work/python-3.12.10-embed-amd64.zip") as python:
            for name in python.namelist():
                archive.writestr(prefix + "runtime/python/" + name, python.read(name))
        archive.writestr(prefix + "manifests/active.json", json.dumps({"app": VERSION, "resources": None}))
        for name, action in (("启动", "start"), ("导入更新", "import"), ("检查修复", "repair"), ("回退", "rollback")):
            bat = '@echo off\r\nsetlocal\r\nchcp 65001 >nul\r\ncd /d "%~dp0"\r\n"%~dp0runtime\\python\\python.exe" -X utf8 "%~dp0bootstrap\\launch.py" ' + action + ' %*\r\nif errorlevel 1 echo 操作未完成，请查看上方提示。\r\npause\r\n'
            archive.writestr(prefix + name + ".bat", bat.encode("utf-8"))
        archive.write(ROOT / "docs/portable_release_zh.md", prefix + "使用说明.md")
        archive.writestr(prefix + "imports/放置维护者提供的ZIP.txt", "将图库或程序更新包原样放在此目录，不用自行解压。")
    output.with_suffix(".zip.sha256").write_text(digest(output) + "  " + output.name + "\n", "utf-8")
    print(f"Built lightweight portable ZIP: {output.stat().st_size/1024**2:.1f} MiB", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-only", action="store_true")
    parser.add_argument("--gallery-only", action="store_true")
    args = parser.parse_args()
    if not args.app_only:
        gallery()
    if not args.gallery_only:
        app()
