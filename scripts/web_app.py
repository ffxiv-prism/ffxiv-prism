from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from email import policy
from email.parser import BytesParser
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "outputs" / "web"
DATA_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "eorzea_collection" / "images"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
EXECUTION_DEVICES = {"cpu", "cuda"}
RUNNING_STATES = {"queued", "running"}
LOADING_WEIGHTS_PROGRESS = re.compile(r"\bLoading weights:\s*(\d{1,3})%")
DATASET_MODES: dict[str, list[str]] = {
    "standard": [],
    "full": ["--all"],
    "local-index": [
        "--stage",
        "build-index",
        "--stage",
        "validate-images",
        "--stage",
        "build-sqlite",
    ],
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: str
    title: str
    status: str = "queued"
    stage: str = "等待启动"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error: str | None = None
    output_dir: str | None = None
    device: str | None = None
    logs: list[str] = field(default_factory=list)

    def public(self, *, include_logs: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if include_logs:
            payload["logs"] = self.logs[-160:]
        else:
            payload.pop("logs", None)
        if self.kind == "recognition" and self.status == "succeeded":
            payload["result_url"] = f"/api/jobs/{self.id}/result"
            payload["report_url"] = f"/api/jobs/{self.id}/report"
        return payload


class JobRegistry:
    def __init__(self, python_executable: str | Path | None = None) -> None:
        self.python_executable = str(python_executable or sys.executable)
        self._jobs: dict[str, Job] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def reserve(
        self,
        *,
        kind: str,
        title: str,
        job_id: str | None = None,
        output_dir: Path | None = None,
        device: str | None = None,
    ) -> Job:
        with self._lock:
            active = next(
                (job for job in self._jobs.values() if job.status in RUNNING_STATES),
                None,
            )
            if active is not None:
                raise RuntimeError(f"当前已有任务在运行：{active.title}")
            job = Job(
                id=job_id or uuid.uuid4().hex[:12],
                kind=kind,
                title=title,
                output_dir=(
                    str(output_dir.relative_to(PROJECT_ROOT)).replace("\\", "/")
                    if output_dir is not None
                    else None
                ),
                device=device,
            )
            self._jobs[job.id] = job
            return job

    def start(
        self,
        job: Job,
        command: list[str],
        *,
        environment_overrides: dict[str, str] | None = None,
    ) -> None:
        thread = threading.Thread(
            target=self._run,
            args=(job.id, command, environment_overrides or {}),
            name=f"ffxiv-prism-{job.kind}-{job.id}",
            daemon=True,
        )
        thread.start()

    def fail_reserved(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.error = message
            job.finished_at = _now()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
            return [job.public(include_logs=False) for job in jobs[:20]]

    def _run(
        self,
        job_id: str,
        command: list[str],
        environment_overrides: dict[str, str],
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.stage = "正在启动"
            job.started_at = _now()

        process: subprocess.Popen[str] | None = None
        try:
            environment = os.environ.copy()
            environment.setdefault("PYTHONIOENCODING", "utf-8")
            environment.setdefault("PYTHONUTF8", "1")
            environment.update(environment_overrides)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            )
            with self._lock:
                self._processes[job_id] = process

            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                with self._lock:
                    job = self._jobs[job_id]
                    _append_job_log(job, line)
                    job.stage = _stage_from_log(line, job.kind)

            return_code = process.wait()
            with self._lock:
                job = self._jobs[job_id]
                job.return_code = return_code
                job.finished_at = _now()
                if return_code == 0:
                    job.status = "succeeded"
                    job.stage = "已完成"
                else:
                    job.status = "failed"
                    job.stage = "运行失败"
                    job.error = _last_useful_log(job.logs) or f"进程退出码：{return_code}"
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.stage = "运行失败"
                job.error = str(exc)
                job.finished_at = _now()
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            with self._lock:
                self._processes.pop(job_id, None)


def _loading_weights_percent(line: str) -> int | None:
    match = LOADING_WEIGHTS_PROGRESS.search(line)
    if match is None:
        return None
    percent = int(match.group(1))
    return percent if 0 <= percent <= 100 else None


def _append_job_log(job: Job, line: str) -> None:
    current_percent = _loading_weights_percent(line)
    previous_percent = (
        _loading_weights_percent(job.logs[-1]) if job.logs else None
    )
    if (
        current_percent is not None
        and previous_percent is not None
        and current_percent >= previous_percent
    ):
        job.logs[-1] = line
    else:
        job.logs.append(line)
    if len(job.logs) > 1000:
        del job.logs[:-600]


def _stage_from_log(line: str, kind: str) -> str:
    if kind == "recognition":
        match = re.search(r"\bstage=([^\s]+)(?:\s+slot=([^\s]+))?", line)
        if match:
            stage, slot = match.groups()
            labels = {
                "pose": "定位人物与装备区域",
                "retrieve": f"检索 {slot or ''} 候选".strip(),
            }
            return labels.get(stage, stage)
    normalized = line.replace("\\", "/")
    dataset_labels = {
        "discover_eorzea_gearsets": "发现装备套装",
        "collect_eorzea_gearset_details": "采集装备详情",
        "build_eorzea_image_url_manifest": "整理图片清单",
        "download_cafemaker": "更新中文名称",
        "search_huiji_item_zh_name": "补充中文名称",
        "build_eorzea_dataset_index": "重建数据索引",
        "validate_eorzea_images": "校验本地图片",
        "build_eorzea_sqlite": "生成本地数据库",
    }
    for marker, label in dataset_labels.items():
        if marker in normalized:
            return label
    return line[:96]


def _last_useful_log(lines: list[str]) -> str | None:
    for line in reversed(lines):
        if line.strip():
            return line[-500:]
    return None


def build_recognition_command(
    *,
    python_executable: str,
    image_path: Path,
    output_dir: Path,
    slots: list[str],
    top_k: int,
    gender: str | None,
    profile: str,
    device: str,
) -> list[str]:
    if not slots or any(slot not in SLOTS for slot in slots):
        raise ValueError("至少选择一个有效槽位。")
    if not 1 <= top_k <= 20:
        raise ValueError("候选数量必须在 1 到 20 之间。")
    if gender not in {None, "female", "male"}:
        raise ValueError("角色性别参数无效。")
    if profile not in {"current-best", "base"}:
        raise ValueError("识别配置无效。")
    if device not in EXECUTION_DEVICES:
        raise ValueError("运行设备必须是 CPU 或 GPU。")

    command = [
        python_executable,
        str(PROJECT_ROOT / "scripts" / "recognize_glamour.py"),
        str(image_path),
        "--slots",
        *slots,
        "--top-k",
        str(top_k),
        "--profile",
        profile,
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]
    if gender:
        command.extend(["--gender", gender])
    return command


def _probe_execution_runtime(python_executable: str) -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        return {
            "python_executable": python_executable,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "logical_processors": os.cpu_count(),
            "cpu": {
                "available": True,
                "label": "CPU",
            },
            "gpu": {
                "available": cuda_available,
                "label": "GPU",
                "device_name": torch.cuda.get_device_name(0)
                if cuda_available
                else None,
                "reason": None
                if cuda_available
                else "当前 Python/PyTorch 环境未检测到可用 CUDA GPU。",
            },
        }
    except (ImportError, OSError, RuntimeError) as exc:
        return {
            "python_executable": python_executable,
            "python_version": None,
            "torch_version": None,
            "torch_threads": None,
            "logical_processors": os.cpu_count(),
            "cpu": {
                "available": True,
                "label": "CPU",
            },
            "gpu": {
                "available": False,
                "label": "GPU",
                "device_name": None,
                "reason": f"无法确认 GPU 可用性：{exc}",
            },
        }


@lru_cache(maxsize=4)
def execution_runtime_status(python_executable: str) -> dict[str, Any]:
    return _probe_execution_runtime(python_executable)


def validate_execution_device(device: str, runtime: dict[str, Any]) -> str:
    if device not in EXECUTION_DEVICES:
        raise ValueError("运行设备必须是 CPU 或 GPU。")
    if device == "cuda" and not bool(runtime.get("gpu", {}).get("available")):
        reason = runtime.get("gpu", {}).get("reason") or "当前环境没有可用 CUDA GPU。"
        raise ValueError(
            f"无法使用 GPU：{reason} 请按 environment-gpu.yml 配置环境并重启 WebUI，"
            "或改选 CPU。"
        )
    return device


def execution_environment_overrides(device: str) -> dict[str, str]:
    if device not in EXECUTION_DEVICES:
        raise ValueError("运行设备必须是 CPU 或 GPU。")
    return {"CUDA_VISIBLE_DEVICES": "-1"} if device == "cpu" else {}


def build_dataset_command(*, python_executable: str, mode: str) -> list[str]:
    if mode not in DATASET_MODES:
        raise ValueError("数据集更新模式无效。")
    return [
        python_executable,
        str(PROJECT_ROOT / "scripts" / "run_data_pipeline.py"),
        *DATASET_MODES[mode],
    ]


def parse_multipart_form(
    content_type: str,
    body: bytes,
) -> tuple[dict[str, str], dict[str, tuple[str, str, bytes]]]:
    if "multipart/form-data" not in content_type.lower():
        raise ValueError("请求必须使用 multipart/form-data。")
    message = BytesParser(policy=policy.default).parsebytes(
        (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8")
        + body
    )
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, str, bytes]] = {}
    if not message.is_multipart():
        raise ValueError("无法解析上传内容。")
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            files[name] = (
                filename,
                part.get_content_type() or "application/octet-stream",
                payload,
            )
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = payload.decode(charset, errors="replace")
    return fields, files


def detect_image_extension(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    signature_extension: str | None = None
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        signature_extension = ".png"
    elif content.startswith(b"\xff\xd8\xff"):
        signature_extension = ".jpg"
    elif len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        signature_extension = ".webp"
    if signature_extension is None:
        raise ValueError("仅支持 PNG、JPG/JPEG 或 WebP 图片。")
    if extension not in {".png", ".jpg", ".jpeg", ".webp"}:
        return signature_extension
    if extension == ".jpeg":
        extension = ".jpg"
    if extension != signature_extension:
        raise ValueError("图片扩展名与实际格式不一致。")
    return signature_extension


def resolve_served_file(raw_path: str) -> Path:
    decoded = unquote(raw_path).strip()
    if not decoded:
        raise ValueError("文件路径不能为空。")
    candidate = Path(decoded)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    allowed_roots = (OUTPUT_ROOT.resolve(), DATA_IMAGE_ROOT.resolve())
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise PermissionError("该文件不在网页可访问范围内。")
    if not resolved.is_file():
        raise FileNotFoundError("文件不存在。")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_status() -> dict[str, Any]:
    summary_paths = {
        "dataset": PROJECT_ROOT / "data" / "processed" / "eorzea_dataset_summary.json",
        "images": PROJECT_ROOT / "data" / "processed" / "eorzea_image_qa_summary.json",
        "sqlite": PROJECT_ROOT / "data" / "processed" / "eorzea_sqlite_summary.json",
    }
    summaries: dict[str, Any] = {}
    timestamps: list[float] = []
    for key, path in summary_paths.items():
        if path.is_file():
            try:
                summaries[key] = _read_json(path)
                timestamps.append(path.stat().st_mtime)
            except (OSError, json.JSONDecodeError):
                summaries[key] = {}
        else:
            summaries[key] = {}
    last_updated = (
        datetime.fromtimestamp(max(timestamps), UTC).isoformat(timespec="seconds")
        if timestamps
        else None
    )
    dataset = summaries["dataset"]
    images = summaries["images"]
    sqlite = summaries["sqlite"]
    return {
        "ready": bool(sqlite.get("database") and (PROJECT_ROOT / sqlite["database"]).is_file()),
        "image_count": dataset.get("image_count") or sqlite.get("images"),
        "gear_pieces": sqlite.get("gear_pieces"),
        "gearsets": sqlite.get("gearsets"),
        "ready_images": images.get("ready_rows", sqlite.get("ready_images")),
        "missing_images": images.get("missing_rows", sqlite.get("missing_or_invalid_images")),
        "last_updated": last_updated,
    }


class PrismRequestHandler(BaseHTTPRequestHandler):
    server_version = "FFXIVPrism/1.0"

    @property
    def registry(self) -> JobRegistry:
        return self.server.registry  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_file(WEB_ROOT / "index.html", cache=False)
            elif parsed.path.startswith("/static/"):
                filename = parsed.path.removeprefix("/static/")
                if "/" in filename or "\\" in filename or filename.startswith("."):
                    self._send_error(HTTPStatus.NOT_FOUND, "资源不存在。")
                    return
                self._send_file(WEB_ROOT / filename, cache=True)
            elif parsed.path == "/api/status":
                self._send_json(
                    {
                        "app": "FFXIV Prism",
                        "local": True,
                        "dataset": dataset_status(),
                        "execution": execution_runtime_status(
                            self.registry.python_executable
                        ),
                        "jobs": self.registry.list(),
                    }
                )
            elif parsed.path == "/api/jobs":
                self._send_json({"jobs": self.registry.list()})
            elif parsed.path == "/api/file":
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                self._send_file(resolve_served_file(raw_path), cache=True)
            else:
                job_route = _parse_job_route(parsed.path)
                if job_route is None:
                    self._send_error(HTTPStatus.NOT_FOUND, "页面不存在。")
                    return
                job_id, resource = job_route
                job = self.registry.get(job_id)
                if job is None:
                    self._send_error(HTTPStatus.NOT_FOUND, "任务不存在。")
                    return
                if resource is None:
                    self._send_json(job.public())
                elif resource == "result":
                    self._send_job_result(job)
                elif resource == "report":
                    self._send_job_report(job)
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "任务资源不存在。")
        except PermissionError as exc:
            self._send_error(HTTPStatus.FORBIDDEN, str(exc))
        except FileNotFoundError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"读取失败：{exc}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/recognize":
                self._start_recognition()
            elif parsed.path == "/api/dataset/update":
                self._start_dataset_update()
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "接口不存在。")
        except RuntimeError as exc:
            self._send_error(HTTPStatus.CONFLICT, str(exc))
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"启动失败：{exc}")

    def log_message(self, format: str, *args: Any) -> None:
        print(
            f"{self.address_string()} - [{self.log_date_time_string()}] {format % args}",
            flush=True,
        )

    def _start_recognition(self) -> None:
        content_length = self._content_length(MAX_UPLOAD_BYTES + MAX_JSON_BYTES)
        fields, files = parse_multipart_form(
            self.headers.get("Content-Type", ""),
            self.rfile.read(content_length),
        )
        upload = files.get("image")
        if upload is None:
            raise ValueError("请选择一张图片。")
        original_name, _content_type, image_bytes = upload
        if not image_bytes:
            raise ValueError("上传图片为空。")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError("图片不能超过 25 MB。")

        extension = detect_image_extension(original_name, image_bytes)
        slots = [item for item in fields.get("slots", "").split(",") if item]
        top_k = int(fields.get("top_k", "10"))
        gender = fields.get("gender") or None
        profile = fields.get("profile", "current-best")
        runtime = execution_runtime_status(self.registry.python_executable)
        default_device = "cuda" if runtime["gpu"]["available"] else "cpu"
        device = validate_execution_device(
            fields.get("device", default_device),
            runtime,
        )

        job_id = uuid.uuid4().hex[:12]
        output_dir = OUTPUT_ROOT / job_id
        job = self.registry.reserve(
            kind="recognition",
            title=f"识别 {Path(original_name).name}",
            job_id=job_id,
            output_dir=output_dir,
            device=device,
        )
        try:
            upload_dir = output_dir / "input"
            upload_dir.mkdir(parents=True, exist_ok=False)
            image_path = upload_dir / f"source{extension}"
            image_path.write_bytes(image_bytes)
            command = build_recognition_command(
                python_executable=self.registry.python_executable,
                image_path=image_path,
                output_dir=output_dir,
                slots=slots,
                top_k=top_k,
                gender=gender,
                profile=profile,
                device=device,
            )
        except Exception as exc:
            self.registry.fail_reserved(job.id, str(exc))
            raise
        self.registry.start(
            job,
            command,
            environment_overrides=execution_environment_overrides(device),
        )
        self._send_json({"job": job.public()}, status=HTTPStatus.ACCEPTED)

    def _start_dataset_update(self) -> None:
        payload = self._read_request_json()
        mode = str(payload.get("mode", "standard"))
        labels = {
            "standard": "标准数据集更新",
            "full": "完整数据集更新",
            "local-index": "重建本地索引",
        }
        if mode not in labels:
            raise ValueError("数据集更新模式无效。")
        command = build_dataset_command(
            python_executable=self.registry.python_executable,
            mode=mode,
        )
        job = self.registry.reserve(kind="dataset", title=labels[mode])
        self.registry.start(job, command)
        self._send_json({"job": job.public()}, status=HTTPStatus.ACCEPTED)

    def _send_job_result(self, job: Job) -> None:
        if job.status != "succeeded" or not job.output_dir:
            self._send_error(HTTPStatus.CONFLICT, "识别结果尚未就绪。")
            return
        result_path = PROJECT_ROOT / job.output_dir / "results.json"
        self._send_json(_read_json(result_path))

    def _send_job_report(self, job: Job) -> None:
        if job.status != "succeeded" or not job.output_dir:
            self._send_error(HTTPStatus.CONFLICT, "识别报告尚未就绪。")
            return
        report_path = PROJECT_ROOT / job.output_dir / "report.html"
        html = report_path.read_text(encoding="utf-8")
        html = re.sub(
            r"file:///([^\"']+)",
            lambda match: "/api/file?path="
            + quote(unquote(match.group(1)).replace("/", "\\"), safe=""),
            html,
        )
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_request_json(self) -> dict[str, Any]:
        content_length = self._content_length(MAX_JSON_BYTES)
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求内容不是有效 JSON。") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求内容必须是 JSON 对象。")
        return payload

    def _content_length(self, maximum: int) -> int:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length 无效。") from exc
        if length <= 0:
            raise ValueError("请求内容为空。")
        if length > maximum:
            raise ValueError("请求内容过大。")
        return length

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_file(self, path: Path, *, cache: bool) -> None:
        if not path.is_file():
            raise FileNotFoundError("文件不存在。")
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
        }:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control",
            "public, max-age=3600" if cache else "no-store",
        )
        self.end_headers()
        self.wfile.write(body)


class PrismServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        registry: JobRegistry | None = None,
    ) -> None:
        self.registry = registry or JobRegistry()
        # Load PyTorch and cache device capabilities on the server's main thread.
        # Importing Conda extension modules for the first time in a request worker
        # can fail on Windows before the environment DLL search path is settled.
        execution_runtime_status(self.registry.python_executable)
        super().__init__(server_address, PrismRequestHandler)


def _parse_job_route(path: str) -> tuple[str, str | None] | None:
    match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})(?:/(result|report))?", path)
    if not match:
        return None
    return match.group(1), match.group(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 FFXIV Prism 本地网页。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="启动后不自动打开浏览器。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not WEB_ROOT.joinpath("index.html").is_file():
        raise SystemExit(f"网页资源不存在：{WEB_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = PrismServer((args.host, args.port))
    host, port = server.server_address[:2]
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}"
    print(f"FFXIV Prism 本地网页已启动：{url}", flush=True)
    print("按 Ctrl+C 停止。", flush=True)
    if not args.no_open:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n正在停止本地网页……", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
