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
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

SCHEMA = 1

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
        self.update(initial, force=True)

    def update(self, current, force=False):
        now = time.monotonic()
        if not force and now - self.last < (0.25 if self.terminal else 2):
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
        if time.monotonic() - last >= 2 or number == len(manifest["files"]):
            status(f"已检查 {number}/{len(manifest['files'])}：{row['path']}")
            last = time.monotonic()

def install_package(home, filename):
    home = Path(home)
    status(f"读取并验证更新包签名：{filename}")
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
        staging = home / "work/staging" / ("a" if category == "app" else "r")
        longest = max(len(str(destination / row["path"])) for row in manifest["files"])
        if os.name == "nt" and longest >= 250:
            raise ValueError("安装目录过深，装备图片文件名可能超过 Windows 限制。请将整个项目移到短英文路径，例如 D:\\FFXIV-Prism。")
        staging.mkdir(parents=True, exist_ok=True)
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
            if time.monotonic() - last >= 2 or number == len(manifest["files"]):
                status(f"安装 {manifest['id']}：{number}/{len(manifest['files'])}")
                last = time.monotonic()
        (staging / "package.json").write_bytes(archive.read("package.json"))
        (staging / "package.sig").write_bytes(archive.read("package.sig"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if damaged:
            backup = home / "work/rollback" / (category + "-" + manifest["id"] + "-" + str(time.time_ns()))
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
        raise ValueError("尚未导入图库资源包。请从维护者的网盘下载 ZIP，放入 imports，双击“导入更新.bat”。")
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
        status(f"检查本地缓存：{destination.name}")
        if digest(destination, progress=True) == row["sha256"]:
            status("本地缓存校验通过，无需下载。")
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
                        status(f"已恢复完整下载：{destination.name}")
                        return
                    temporary.unlink()
                    offset = 0
                status(f"下载 {destination.name} | 来源 {source_number}/{len(urls)} | 尝试 {attempt + 1}/2")
                status(f"网站：{urllib.parse.urlsplit(url).netloc}\n下载地址：{url}")
                status(f"连接中（超时 {timeout}s）" + (f"，从 {offset / 1024**2:.2f} MiB 断点续传" if offset else ""))
                request = urllib.request.Request(url, headers={"User-Agent": "FFXIVPrism/portable", **({"Range": f"bytes={offset}-"} if offset else {})})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    redirected = response.geturl()
                    if redirected != url:
                        # CDN query strings may contain temporary credentials.
                        parts = urllib.parse.urlsplit(redirected)
                        status(f"实际下载网站：{parts.netloc}（重定向）")
                    resumed = offset and response.status == 206
                    if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Invalid resume response")
                    if offset and not resumed:
                        status("来源不支持此断点，从头下载。")
                    meter = Progress("下载", row["size"], offset if resumed else 0)
                    with temporary.open("ab" if resumed else "wb") as stream:
                        while chunk := response.read(64 * 1024):
                            stream.write(chunk)
                            if stream.tell() > row["size"]:
                                raise ValueError("Download exceeds expected size")
                            meter.update(stream.tell())
                        meter.update(stream.tell(), force=True)
                    meter.close()
                if temporary.stat().st_size != row["size"]:
                    raise OSError("下载未完成，保留断点后重试")
                status(f"下载完成，正在校验 SHA-256：{destination.name}")
                if digest(temporary, progress=True) != row["sha256"]:
                    temporary.unlink(missing_ok=True)
                    raise ValueError("下载校验失败")
                temporary.replace(destination)
                status(f"校验通过：{destination.name}")
                return
            except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
                if meter:
                    meter.close()
                errors.append(f"{url}: {exc}")
                status(f"下载失败：{exc}；将重试或切换下一个来源（如有）。")
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                    break
                time.sleep(.2)
    raise RuntimeError(f"无法准备 {destination}；可手动下载到此路径。\n" + "\n".join(errors))

def prepare_runtime(home, app):
    home, app = Path(home), Path(app)
    lock = read_json(app / "manifests/runtime.json")
    status(f"检查 Python 运行依赖：{len(lock['wheels'])} 个固定版本包")
    target = home / "runtime/envs" / lock["id"] / "site-packages"
    marker = target.parent / "ready.json"
    if marker.is_file() and read_json(marker).get("lock_sha256") == digest(app / "manifests/runtime.json"):
        status("Python 运行依赖已就绪。")
        return target
    target.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(home).free < sum(row["size"] for row in lock["wheels"]) * 4 + 256 * 1024**2:
        raise ValueError("磁盘空间不足以准备运行环境。")
    for number, row in enumerate(lock["wheels"], 1):
        path = home / "work/downloads" / row["filename"]
        status(f"准备依赖 [{number}/{len(lock['wheels'])}]：{row['filename']}")
        download(row, path)
        status(f"解压安装依赖：{row['filename']}")
        with zipfile.ZipFile(path) as archive:
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
    atomic_json(marker, {"lock_sha256": digest(app / "manifests/runtime.json")})
    status("Python 运行依赖准备完成。")
    return target

def prepare_models(home, app, full=False):
    models = read_json(Path(app) / "manifests/models.json")["models"]
    status(f"检查基础模型：共 {len(models)} 个模型")
    for number, model in enumerate(models, 1):
        status(f"模型 [{number}/{len(models)}]：{model['id']}")
        directory = safe_path(home, model["directory"])
        for row in model["files"]:
            path = safe_path(directory, row["path"])
            receipt = path.with_suffix(path.suffix + ".verified.json")
            verified = receipt.is_file() and read_json(receipt).get("sha256") == row["sha256"]
            status(f"检查模型文件：{row['path']}")
            if path.is_file() and path.stat().st_size == row["size"] and (not full and verified or digest(path, progress=True) == row["sha256"]):
                atomic_json(receipt, {"sha256": row["sha256"]})
                continue
            print(f"准备模型：{model['id']} / {row['path']}", flush=True)
            download(row, path)
            atomic_json(receipt, {"sha256": row["sha256"]})
    status("基础模型已就绪。")

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
        "MPLCONFIGDIR": str(home / "work/cache/matplotlib"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
    })
    if resource_root:
        os.environ["PRISM_RESOURCE_ROOT"] = str(Path(resource_root).resolve())
