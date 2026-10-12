"""Portable package operations. Standard library only; no update discovery."""
from __future__ import annotations
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

SCHEMA = 1
MAX_INSTALL_ROOT_CHARS = 64  # Checked against compact gallery and installed wheel paths.


def check_install_root(home):
    home = str(Path(home).resolve())
    if not home.isascii():
        raise ValueError('请使用不含中文的短英文安装路径，例如 D:\\FFXIV-Prism。')
    if os.name == 'nt' and len(home) > MAX_INSTALL_ROOT_CHARS:
        raise ValueError(f'安装根目录长 {len(home)} 个字符，当前资源允许最多 {MAX_INSTALL_ROOT_CHARS} 个（含盘符和分隔符）。请缩短 {len(home) - MAX_INSTALL_ROOT_CHARS} 个字符，例如 D:\\FFXIV-Prism。')


def check_windows_paths(root, names):
    if os.name != 'nt':
        return
    longest = max((len(str(Path(root) / name).encode('utf-16-le')) // 2 for name in names), default=0)
    if longest >= 250:
        raise ValueError(f'安装路径最长 {longest} 个字符，保守上限为 249；请将根目录至少缩短 {longest - 249} 个字符。')


def remove_managed_files(home, paths, folder):
    """Only delete completed managed downloads; never follow links outside home."""
    home = Path(home).resolve()
    managed = safe_path(home, folder)
    removed = 0
    for path in paths:
        path = Path(path)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(managed):
            continue
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError as exc:
            status(f'安装成功，下载文件暂未清理：{path.name}：{exc}')
    if removed:
        status(f'已清理 {removed} 个安装文件。')


def cleanup_installation(home, app, packages=()):
    """Clean only known installer outputs after the application selfcheck succeeds."""
    home, app = Path(home), Path(app)
    downloads = []
    for name in ('runtime.json', 'runtime-cu126.json', 'runtime-cu130.json'):
        manifest = app / 'manifests' / name
        if not manifest.is_file():
            continue
        lock = read_json(manifest)
        marker = safe_path(home, 'runtime/' + lock['id'] + '/ready.json')
        if not marker.is_file() or read_json(marker).get('lock_sha256') != digest(manifest):
            continue
        for row in lock['wheels']:
            path = safe_path(home, 'work/downloads/' + row['filename'])
            downloads.extend((path, path.with_suffix(path.suffix + '.part')))
    remove_managed_files(home, downloads, 'work/downloads')
    remove_managed_files(home, packages, 'imports')
    remove_managed_files(home, [home / 'work/proposed.json'], 'work')
    # These directories should already be empty. Never remove arbitrary temp/cache content.
    for name in ('work/downloads', 'work/staging', 'work/temp'):
        try:
            safe_path(home, name).rmdir()
        except OSError:
            pass

def status(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

class Progress:
    """Single-line console progress, with readable periodic lines in redirected logs."""
    def __init__(self, label, total, initial=0):
        self.label, self.total, self.initial = label, total, initial
        self.started = time.monotonic()
        self.last = 0.0
        self.terminal = sys.stdout.isatty()
        self.active = False
        self.enabled = total >= 16 * 1024**2
        self.update(initial, force=True)

    def update(self, current, force=False):
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last < (1 if self.terminal else 15):
            return
        self.last = now
        fraction = min(current / self.total, 1) if self.total else 1
        filled = int(fraction * 24)
        speed = max(0, current - self.initial) / max(now - self.started, .001)
        eta = f"{(self.total - current) / speed:.0f}s" if speed and current < self.total else "--"
        line = (f"{self.label} [{'#' * filled}{'-' * (24 - filled)}] {fraction:6.1%} "
                f"{current / 1024**2:.2f}/{self.total / 1024**2:.2f} MiB "
                f"{speed / 1024**2:.2f} MiB/s ETA {eta}")
        print(("\r" if self.terminal else "") + line, end="" if self.terminal else "\n", flush=True)
        self.active = True

    def close(self):
        if self.terminal and self.active:
            print(flush=True)
        self.active = False

def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temp.replace(path)

def digest(path, progress=False):
    with Path(path).open("rb") as stream:
        if not progress:
            return hashlib.file_digest(stream, "sha256").hexdigest()
        meter = Progress("SHA-256", Path(path).stat().st_size)
        result = hashlib.sha256()
        try:
            while chunk := stream.read(1024 * 1024):
                result.update(chunk)
                meter.update(stream.tell())
            meter.update(stream.tell(), force=True)
            return result.hexdigest()
        finally:
            meter.close()

def safe_path(root, name):
    # Reject Windows ADS, drive names, traversal, duplicate separator aliases and links.
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError(f"Invalid package path: {name}")
    parts = name.split("/")
    if not parts or any(p in {"", ".", ".."} or p.endswith((" ", ".")) or re.match(r"(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", p) for p in parts):
        raise ValueError(f"Invalid package path: {name}")
    target = Path(root).joinpath(*parts).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError("Package path escapes destination")
    return target

def verify_signature(raw, signature, key):
    """Trust only keys bundled in the launcher, including retained release keys."""
    for trusted in (key, *key.get("previous_keys", [])):
        try:
            _verify_signature(raw, signature, trusted)
            return
        except ValueError:
            pass
    raise ValueError("更新包签名无效，请重新下载维护者提供的原始包。")

def _verify_signature(raw, signature, key):
    """Strict RSASSA-PKCS1-v1_5/SHA-256 verification of signed manifest bytes."""
    n, e = int(key["n"], 16), int(key["e"], 16)
    size = (n.bit_length() + 7) // 8
    sig = base64.b64decode(signature, validate=True)
    if size < 256 or len(sig) != size or int.from_bytes(sig, "big") >= n:
        raise ValueError("Invalid release signature")
    value = pow(int.from_bytes(sig, "big"), e, n).to_bytes(size, "big")
    payload = bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(raw).digest()
    expected = b"\x00\x01" + b"\xff" * (size - len(payload) - 3) + b"\x00" + payload
    if value != expected:
        raise ValueError("更新包签名无效，请重新下载维护者提供的原始包。")

def package_manifest(archive, home):
    names = archive.namelist()
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("Duplicate archive paths")
    raw = archive.read("package.json")
    verify_signature(raw, archive.read("package.sig"), read_json(Path(home) / "bootstrap/public-key.json"))
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA or manifest.get("kind") not in {"app", "resources"}:
        raise ValueError("Unsupported package format")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", manifest.get("id", "")):
        raise ValueError("Invalid package id")
    expected = {"package.json", "package.sig"}
    for row in manifest["files"]:
        safe_path(home, row["path"])
        expected.add("payload/" + row["path"])
    if set(names) != expected or len(expected) != len(manifest["files"]) + 2:
        raise ValueError("Package contains missing, duplicate or unlisted files")
    for item in archive.infolist():
        if (item.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("Archive links are not allowed")
    return manifest

def verify_files(root, manifest, full=True):
    status(f"检查文件：{root}（{len(manifest['files'])} 个，{'完整校验' if full else '大小检查'}）")
    last = time.monotonic()
    for number, row in enumerate(manifest["files"], 1):
        path = safe_path(root, row["path"])
        if not path.is_file() or path.stat().st_size != row["size"]:
            raise ValueError(f"文件缺失或大小不符：{row['path']}")
        if full and digest(path, progress=row["size"] >= 64 * 1024**2) != row["sha256"]:
            raise ValueError(f"文件校验失败：{row['path']}")
        if time.monotonic() - last >= 10:
            status(f"已检查 {number}/{len(manifest['files'])} 个文件")
            last = time.monotonic()

def install_package(home, filename):
    home = Path(home)
    status(f"验证图库签名：{Path(filename).name}")
    with zipfile.ZipFile(filename) as archive:
        manifest = package_manifest(archive, home)
        category = "app" if manifest["kind"] == "app" else "resources"
        destination = home / category / manifest["id"]
        damaged = False
        if destination.exists():
            existing = destination / "package.json"
            if existing.is_file() and existing.read_bytes() != archive.read("package.json"):
                raise ValueError("同一版本编号对应不同内容；请使用新的发布编号。")
            try:
                verify_files(destination, manifest)
                return manifest
            except ValueError:
                damaged = True
        required = sum(row["size"] for row in manifest["files"])
        if shutil.disk_usage(home).free < required + 256 * 1024**2:
            raise ValueError("磁盘空间不足，无法暂存并保留上一版本。")

        check_windows_paths(destination, [row['path'] for row in manifest['files']])
        staging_root = safe_path(home, "work/staging")
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="app-" if category == "app" else "res-", dir=staging_root) as directory:
            staging = Path(directory)
            check_windows_paths(staging, [row['path'] for row in manifest['files']])
            status(f"解压安装 {manifest['id']}：共 {len(manifest['files'])} 个文件")
            last = time.monotonic()
            for number, row in enumerate(manifest["files"], 1):
                info = archive.getinfo("payload/" + row["path"])
                if info.file_size != row["size"]:
                    raise ValueError("Archive size differs from signed manifest")
                path = safe_path(staging, row["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                if digest(path) != row["sha256"]:
                    raise ValueError(f"解压校验失败：{row['path']}")
                if time.monotonic() - last >= 10 or number == len(manifest["files"]):
                    status(f"安装 {manifest['id']}：{number}/{len(manifest['files'])}")
                    last = time.monotonic()
            (staging / "package.json").write_bytes(archive.read("package.json"))
            (staging / "package.sig").write_bytes(archive.read("package.sig"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if damaged:
                # Keep repaired-resource backups shorter than their installed path.
                backup = home / "work/rollback" / f"{category[0]}-{time.time_ns():x}"
                check_windows_paths(backup, [row['path'] for row in manifest['files']])
                backup.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(backup)
            # Windows antivirus may briefly hold freshly extracted DLLs/files open.
            for attempt in range(10):
                try:
                    staging.rename(destination)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    status("安装文件暂时被占用，等待后重试……")
                    time.sleep(.2)
            return manifest

def installed_manifest(home, kind, identifier):
    root = Path(home) / kind / identifier
    raw = (root / "package.json").read_bytes()
    verify_signature(raw, (root / "package.sig").read_bytes(), read_json(Path(home) / "bootstrap/public-key.json"))
    return json.loads(raw)

def validate_combination(home, active, full=False):
    app = installed_manifest(home, "app", active["app"])
    verify_files(Path(home) / "app" / active["app"], app, full)
    resource_id = active.get("resources")
    if not resource_id:
        raise ValueError("尚未导入图库资源包。请从维护者的网盘下载 ZIP，放入 imports，双击“安装环境.bat”。")
    resources = installed_manifest(home, "resources", resource_id)
    if resources["compatibility"] not in app["resource_compatibility"]:
        raise ValueError("程序与图库不兼容，请导入配套程序包和图库包。")
    verify_files(Path(home) / "resources" / resource_id, resources, full)
    return app, resources

def download(row, destination, timeout=20):
    """Download only an explicitly pinned component; resume and validate before install."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == row["size"]:
        if digest(destination, progress=True) == row["sha256"]:
            return
    temporary = destination.with_suffix(destination.suffix + ".part")
    errors = []
    urls = row.get("urls", [])
    for source_number, url in enumerate(urls, 1):
        if not url.startswith("https://"):
            raise ValueError("Download sources must use HTTPS")
        for attempt in range(2):
            meter = None
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                if offset >= row["size"]:
                    if offset == row["size"] and digest(temporary, progress=True) == row["sha256"]:
                        temporary.replace(destination)
                        status(f"Recovered completed download: {destination.name}")
                        return
                    temporary.unlink()
                    offset = 0
                status(f"Downloading: {urllib.parse.urlsplit(url).netloc} · {row['size'] / 1024**2:.1f} MiB"
                       + (f" · resuming from {offset / 1024**2:.1f} MiB" if offset else ""))
                request = urllib.request.Request(url, headers={"User-Agent": "FFXIVPrism/portable", **({"Range": f"bytes={offset}-"} if offset else {})})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    resumed = offset and response.status == 206
                    if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Invalid resume response")
                    if offset and not resumed:
                        status("Source does not support resume; restarting download.")
                    meter = Progress("Download", row["size"], offset if resumed else 0)
                    with temporary.open("ab" if resumed else "wb") as stream:
                        while chunk := response.read(64 * 1024):
                            stream.write(chunk)
                            if stream.tell() > row["size"]:
                                raise ValueError("Download exceeds expected size")
                            meter.update(stream.tell())
                        meter.update(stream.tell(), force=True)
                    meter.close()
                if temporary.stat().st_size != row["size"]:
                    raise OSError("Download incomplete; partial file retained for retry")
                if digest(temporary, progress=True) != row["sha256"]:
                    temporary.unlink(missing_ok=True)
                    raise ValueError("Download checksum mismatch")
                temporary.replace(destination)
                return
            except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
                if meter:
                    meter.close()
                errors.append(f"{url}: {exc}")
                status(f"Download failed: {exc}; retrying or trying the next source if available.")
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                    break
                time.sleep(.2)
    raise RuntimeError(f"Could not prepare {destination}; you can download the file manually to this path.\n" + "\n".join(errors))

def prepare_runtime(home, app, manifest_name="runtime.json"):
    home, app = Path(home), Path(app)
    manifest_path = app / "manifests" / manifest_name
    lock = read_json(manifest_path)
    status(f"Checking Python runtime: {len(lock['wheels'])} pinned packages")
    target = home / "runtime" / lock["id"] / "site-packages"
    marker = target.parent / "ready.json"
    if marker.is_file() and read_json(marker).get("lock_sha256") == digest(manifest_path):
        status("Python runtime is ready.")
        return target
    target.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(home).free < sum(row["size"] for row in lock["wheels"]) * 4 + 256 * 1024**2:
        raise ValueError("Not enough free disk space to install the runtime.")
    for number, row in enumerate(lock["wheels"], 1):
        path = home / "work/downloads" / row["filename"]
        status(f"Installing dependency [{number}/{len(lock['wheels'])}]: {row['filename']}")
        download(row, path)
        with zipfile.ZipFile(path) as archive:
            check_windows_paths(target, archive.namelist())
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                parts = name.split("/")
                if len(parts) > 2 and parts[0].endswith(".data"):
                    if parts[1] not in {"purelib", "platlib"}:
                        continue
                    name = "/".join(parts[2:])
                output = safe_path(target, name)
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, output.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
    atomic_json(marker, {"lock_sha256": digest(manifest_path)})
    status("Python runtime installation complete.")
    return target

def prepare_models(home, app, full=False):
    models = read_json(Path(app) / "manifests/models.json")["models"]
    status(f"Checking base models: {len(models)} models")
    for number, model in enumerate(models, 1):
        status(f"Model [{number}/{len(models)}]: {model['id']}")
        directory = safe_path(home, model["directory"])
        for row in model["files"]:
            path = safe_path(directory, row["path"])
            receipt = path.with_suffix(path.suffix + ".verified.json")
            verified = receipt.is_file() and read_json(receipt).get("sha256") == row["sha256"]
            if path.is_file() and path.stat().st_size == row["size"] and (not full and verified or digest(path, progress=True) == row["sha256"]):
                atomic_json(receipt, {"sha256": row["sha256"]})
                continue
            print(f"Preparing model: {model['id']} / {row['path']}", flush=True)
            download(row, path)
            atomic_json(receipt, {"sha256": row["sha256"]})
    status("Base models are ready.")

def configure_environment(home, resource_root=None):
    home = Path(home).resolve()
    for name in ("work/temp", "work/cache", "work/logs", "imports", "user/reports"):
        (home / name).mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "PRISM_HOME": str(home), "PRISM_PORTABLE": "1", "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "TEMP": str(home / "work/temp"), "TMP": str(home / "work/temp"),
        "HF_HOME": str(home / "work/cache/huggingface"),
        "HF_HUB_CACHE": str(home / "work/cache/huggingface/hub"),
        "MODELSCOPE_CACHE": str(home / "work/cache/modelscope"),
        "TORCH_HOME": str(home / "work/cache/torch"),
        "XDG_CACHE_HOME": str(home / "work/cache"),
        "CUDA_CACHE_PATH": str(home / "work/cache/cuda"),
        "PIP_CACHE_DIR": str(home / "work/cache/pip"),
        "TORCHINDUCTOR_CACHE_DIR": str(home / "work/cache/torchinductor"),
        "MPLCONFIGDIR": str(home / "work/cache/matplotlib"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
    })
    if resource_root:
        os.environ["PRISM_RESOURCE_ROOT"] = str(Path(resource_root).resolve())
