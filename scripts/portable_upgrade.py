"""Run from an extracted release update bundle using the installed embedded Python."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import zipfile

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE / "support"))
from portable import (package_manifest, read_json, safe_path, status,
                      verify_files, verify_signature)

MANAGED = ("bootstrap", "启动.bat", "导入更新.bat", "检查修复.bat", "回退.bat", "使用说明.md")
STATE = ("manifests/active.json", "manifests/previous.json", "runtime/python/python312._pth")


@contextmanager
def installation_lock(home):
    import msvcrt
    with safe_path(home, "work/app.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("请先关闭旧版服务，再运行更新脚本。") from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def validate_bundle(bundle):
    raw = (bundle / "update.json").read_bytes()
    verify_signature(raw, (bundle / "update.sig").read_bytes(),
                     read_json(bundle / "support/public-key.json"))
    manifest = json.loads(raw)
    if manifest.get("schema") != 1 or manifest.get("kind") != "portable-upgrade":
        raise ValueError("不支持的升级包。")
    paths = [row["path"] for row in manifest["files"]]
    if len(set(p.casefold() for p in paths)) != len(paths):
        raise ValueError("重复升级文件。")
    verify_files(bundle, manifest, full=True)
    for row in manifest["files"]:
        name = row["path"]
        if name.startswith("payload/"):
            relative = name.removeprefix("payload/")
            if relative.split("/")[0] not in MANAGED:
                raise ValueError("升级包包含非启动器目标。")
    archive = safe_path(bundle, manifest["app_package"])
    with zipfile.ZipFile(archive) as package:
        app = package_manifest(package, bundle / "payload")
        if app["kind"] != "app" or app["id"] != manifest["version"]:
            raise ValueError("程序包版本不符。")
    return manifest


def install_app(home, archive, active):
    launch = runpy.run_path(str(home / "bootstrap/launch.py"), run_name="upgrade_launcher")
    try:
        return launch["install_inputs"](active, [archive])
    except Exception:
        launch["activate_paths"](active, prepare=False)
        raise


def apply_upgrade(home, bundle, manifest, installer=install_app):
    """Caller holds app.lock. Keep backups even after success for manual recovery."""
    home, bundle = Path(home).resolve(), Path(bundle).resolve()
    active = read_json(safe_path(home, "manifests/active.json"))
    if active["app"] not in manifest["supported_from"]:
        raise ValueError(f"不支持从 {active['app']} 升级，请使用完整安装包。")
    # Resolve every managed destination before moving or deleting any file.
    for name in (*MANAGED, *STATE, "work", "imports"):
        path = home / name
        for part in (path, *path.parents):
            if part == home:
                break
            if part.is_symlink() or part.is_junction():
                raise ValueError(f"升级目标不能是符号链接或目录联接：{part}")
    targets = {name: safe_path(home, name) for name in (*MANAGED, *STATE)}
    backup_root = safe_path(home, "work/upgrade-backups")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=manifest["version"] + "-", dir=backup_root))
    moved = []
    saved_state = {name: path.read_bytes() if path.is_file() else None
                   for name, path in targets.items() if name in STATE}
    for name, content in saved_state.items():
        if content is not None:
            saved = safe_path(backup, name)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(content)
    status(f"备份旧启动器：{backup}")
    try:
        for name in MANAGED:
            target = targets[name]
            if target.exists():
                target.rename(safe_path(backup, name))
            moved.append(name)
        # Copy only the signed files; never include unlisted leftovers in the bundle.
        for row in manifest["files"]:
            if row["path"].startswith("payload/"):
                name = row["path"].removeprefix("payload/")
                target = safe_path(home, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(safe_path(bundle, row["path"]), target)
        status("启动器已更新，正在校验和安装程序……")
        installer(home, safe_path(bundle, manifest["app_package"]), active)
    except Exception:
        status("更新失败，恢复旧启动器与活动版本……")
        for name in reversed(moved):
            target = safe_path(home, name)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            old = safe_path(backup, name)
            if old.exists():
                old.rename(target)
        for name, content in saved_state.items():
            target = safe_path(home, name)
            if content is None:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        raise
    # Cleanup only known transient state and signed old app packages. Keep galleries.
    for name in ("work/pending-update.json", "work/proposed.json"):
        try:
            safe_path(home, name).unlink(missing_ok=True)
        except OSError as exc:
            status(f"程序已更新，临时文件待清理 {name}：{exc}")
    imports = safe_path(home, "imports")
    for archive in imports.glob("*.zip"):
        try:
            archive = safe_path(home, archive.relative_to(home).as_posix())
            with zipfile.ZipFile(archive) as package:
                old = package_manifest(package, home)
            if old["kind"] == "app":
                archived = safe_path(backup, "imports/" + archive.name)
                archived.parent.mkdir(parents=True, exist_ok=True)
                archive.rename(archived)
                status(f"已归档程序更新包：{archive.name}")
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            status(f"保留未归档的 ZIP {archive.name}：{exc}")
    status("升级完成。请在原安装目录双击“启动.bat”。旧版本、图库、模型和用户文件均已保留。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    args = parser.parse_args()
    home = args.home.resolve()
    if not (home / "runtime/python/python.exe").is_file() or not (home / "manifests/active.json").is_file():
        raise ValueError("请选择旧版 FFXIV-Prism 安装根目录。")
    if not str(home).isascii():
        raise ValueError("安装目录必须为短英文路径。")
    if BUNDLE.is_relative_to(home / "bootstrap"):
        raise ValueError("请将更新包解压到安装目录之外。")
    safe_path(home, "work").mkdir(exist_ok=True)
    with installation_lock(home):
        manifest = validate_bundle(BUNDLE)
        apply_upgrade(home, BUNDLE, manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"更新未完成：{exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
