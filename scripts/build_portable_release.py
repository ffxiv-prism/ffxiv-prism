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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "bootstrap"), str(ROOT / "scripts")]
from portable import atomic_json, digest, verify_signature
from ffxiv_prism.jsonl import read_jsonl
from release_layout import compact_resources, inference_sources

VERSION = "v0.2.0"
RESOURCE_FILENAME = "FFXIV-Prism-gallery-20261009.zip"
RESOURCE_ID = "gallery-20261009"
SOURCE_GALLERY_ID = "gallery-20261009-visor-feet-v3"
CONTRACT = "prism-inference-v1"
ALLOW_TRUSTED_PREVIOUS_KEY = False

def sign(path):
    if not (ROOT / "work/signing/portable-private.xml").is_file():
        raise FileNotFoundError("Missing release signing key: work/signing/portable-private.xml; restore it or explicitly rotate keys with sign_portable.ps1")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(ROOT / "scripts/sign_portable.ps1"), "-Manifest", str(path),
                    "-KeyPath", str(ROOT / "work/signing/portable-private.xml"),
                    "-PublicPath", str(ROOT / "bootstrap/public-key.json"),
                    *(["-AllowTrustedPreviousKey"] if ALLOW_TRUSTED_PREVIOUS_KEY else [])], check=True, capture_output=True)

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
        for name in ("embeddings.npy", "metadata.jsonl", "summary.json"):
            path = folder / name
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
    feature_root = ROOT / "data/local_features"
    feature_manifest = json.loads((feature_root / "catalogue.json").read_text("utf-8"))
    head_index = selection["candidate_indexes"]["facebook-dinov2-base-head-multiscale-v4"]
    head_images = {r["local_path"] for r in read_jsonl(ROOT / head_index / "metadata.jsonl")
                   if r.get("candidate_crop") == "full"}
    if feature_manifest["head_index"] != head_index or set(feature_manifest["images"]) != head_images:
        raise ValueError("Rebuild catalogue features for the active HEAD index before packaging")
    for name in feature_manifest["files"]:
        path = feature_root / name
        if not path.resolve().is_relative_to(feature_root.resolve()):
            raise ValueError("Invalid catalogue feature path")
        files[path.relative_to(ROOT).as_posix()] = path
    files["data/local_features/catalogue.json"] = feature_root / "catalogue.json"
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
    info = json.loads((ROOT / f"manifests/galleries/{SOURCE_GALLERY_ID}.json").read_text("utf-8"))
    if (info["candidate_indexes"] != selection["candidate_indexes"]
            or info["image_count"] != len(image_names) or info["piece_count"] != len(piece_ids)):
        raise ValueError("Refresh published gallery metadata from collection records before building")
    info["samples_updated_at"] = info["updated_at"]
    info.update(id=RESOURCE_ID, compatibility=CONTRACT)
    atomic_json(build / "gallery.json", info)
    files["gallery.json"] = build / "gallery.json"
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing catalogue assets: {missing[:20]}")
    files = compact_resources(files, selection, build / "compact")
    info = json.loads(files["gallery.json"].read_text("utf-8"))
    longest = max(len("resources/" + RESOURCE_ID + "/" + name) for name in files)
    if longest + 65 > 249:
        raise ValueError(f"Gallery exceeds the 64-character install-root budget: {longest}")
    print(f"Gallery longest installed relative path: {longest}", flush=True)
    return pack(files, ROOT / RESOURCE_FILENAME, {"kind": "resources", "id": RESOURCE_ID, "compatibility": CONTRACT, "gallery": info})

def app():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    release_metadata = ROOT / "work/build/release.json"
    atomic_json(release_metadata, {"version": VERSION, "code_commit": commit,
                                               "working_tree_modified": dirty, "prerelease": False})
    files = inference_sources(ROOT)
    for path in (ROOT / "web").rglob("*"):
        if path.is_file() and path.suffix in {".html", ".css", ".js", ".svg", ".png", ".ico"}:
            files[path.relative_to(ROOT).as_posix()] = path
    for name in ("runtime.json", "runtime-cu126.json", "runtime-cu130.json", "models.json"):
        files["manifests/" + name] = ROOT / "manifests" / name
    files["manifests/release.json"] = release_metadata
    files["scripts/open_report_directory.ps1"] = ROOT / "scripts/open_report_directory.ps1"
    # User instructions live once at the ZIP root; development docs stay in Git.
    manifest = pack(files, ROOT / "work/build" / f"FFXIV-Prism-{VERSION}-app.zip",
                    {"kind": "app", "id": VERSION, "resource_compatibility": [CONTRACT]})
    source = ROOT / "work/build" / f"FFXIV-Prism-{VERSION}-app.zip"
    output = ROOT / "dist" / f"FFXIV-Prism-{VERSION}-windows.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        prefix = "FFXIV-Prism/"
        for name in ("launch.py", "portable.py", "gpu_runtime.py", "public-key.json"):
            archive.write(ROOT / "bootstrap" / name, prefix + "bootstrap/" + name)
        with zipfile.ZipFile(source) as package:
            for name in package.namelist():
                target = name.removeprefix("payload/")
                archive.writestr(prefix + "app/" + VERSION + "/" + target, package.read(name))
        with zipfile.ZipFile(ROOT / "work/python-3.12.10-embed-amd64.zip") as python:
            for name in python.namelist():
                archive.writestr(prefix + "runtime/python/" + name, python.read(name))
        archive.writestr(prefix + "manifests/active.json", json.dumps({"app": VERSION, "resources": None}))
        for name, action in (("安装环境", "install"), ("启动", "start"), ("导入图库", "import"), ("检查修复", "repair")):
            bat = launcher_bat(name, action)
            archive.writestr(prefix + name + ".bat", bat.encode("utf-8"))
        archive.write(ROOT / "docs/portable_release_zh.md", prefix + "使用说明.md")
        archive.writestr(prefix + "imports/放置图库ZIP.txt", "将维护者提供的图库包原样放在此目录，不用自行解压。")
    temporary.replace(output)
    output.with_suffix(".zip.sha256").write_text(digest(output) + "  " + output.name + "\n", "utf-8")
    # The first public version has no separate app/update download. Remove
    # only our exact obsolete outputs, after the replacement ZIP is complete.
    for kind in ("app", "update"):
        obsolete = ROOT / "dist" / f"FFXIV-Prism-{VERSION}-{kind}.zip"
        obsolete.unlink(missing_ok=True)
        obsolete.with_suffix(".zip.sha256").unlink(missing_ok=True)
    print(f"Built lightweight portable ZIP: {output.stat().st_size/1024**2:.1f} MiB", flush=True)

def launcher_bat(name, action):
    return ('@echo off\r\nsetlocal\r\nchcp 65001 >nul\r\n'
            f'title FFXIV Prism - {name}\r\necho [FFXIV Prism] {name}：正在初始化，请保留此窗口……\r\n'
            'cd /d "%~dp0"\r\nset PYTHONUNBUFFERED=1\r\n'
            'if not exist "%~dp0runtime\\python\\python.exe" (\r\n'
            'echo 缺少内置 Python。请完整解压 windows.zip 后重试。\r\npause\r\nexit /b 1\r\n)\r\n'
            '"%~dp0runtime\\python\\python.exe" -u -X utf8 "%~dp0bootstrap\\launch.py" '
            + action + ' %*\r\nset "PRISM_EXIT=%errorlevel%"\r\n'
            'if not "%PRISM_EXIT%"=="0" echo 操作未完成，请查看上方提示。\r\n'
            'pause\r\nexit /b %PRISM_EXIT%\r\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-only", action="store_true")
    parser.add_argument("--gallery-only", action="store_true")
    parser.add_argument("--allow-trusted-previous-key", action="store_true",
                        help="Explicitly authorize signing this build with an already trusted historical key; never changes public trust.")
    args = parser.parse_args()
    ALLOW_TRUSTED_PREVIOUS_KEY = args.allow_trusted_previous_key
    if not args.app_only:
        gallery()
    if not args.gallery_only:
        app()
