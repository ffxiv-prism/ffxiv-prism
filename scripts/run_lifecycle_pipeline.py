from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ffxiv_prism.dataset.update_audit import SAMPLE_PATH, image_snapshot, update_result
from ffxiv_prism.jsonl import write_json_atomic
from ffxiv_prism.paths import portable_path
DEFAULT_REPORT_ROOT = Path("data/outputs/lifecycle-reports")
METRICS_PATH = Path("data/processed/current_recognition_metrics.json")
ASSET_MANIFEST = Path("data/current_best_assets.json")
SAME_MODEL_OUTPUT = Path(
    "data/processed/piece_appearance_equivalence_huiji_exact_models.json"
)
CAFEMAKER_MODELS = Path("data/raw/game_data/cafemaker_item_models_raw.jsonl")


@dataclass(frozen=True)
class Stage:
    name: str
    label: str
    command: tuple[str, ...]
    phase: str
    advisory: bool = False


@dataclass
class StageResult:
    name: str
    label: str
    phase: str
    status: str
    return_code: int | None
    duration_seconds: float
    command: list[str]
    output_tail: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the versioned sample-library update and gated current-best "
            "model release lifecycle."
        )
    )
    parser.add_argument(
        "--workflow",
        choices=("sample-library", "model-release", "all"),
        default="sample-library",
    )
    parser.add_argument(
        "--source-mode",
        choices=("local", "cached", "refresh"),
        default="refresh",
        help=(
            "local rebuilds derived assets only; cached prefers existing web "
            "caches; refresh bypasses cached Eorzea list pages."
        ),
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("skip", "check", "extend"),
        default="extend",
        help=(
            "Check or version-extend active visual indexes after updating the "
            "sample library."
        ),
    )
    parser.add_argument("--candidate-source-suffix", default=None)
    parser.add_argument(
        "--candidate-suffix",
        default=None,
        help="Version suffix for newly extended indexes; defaults to the run ID.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--huiji-proxy", default=None)
    parser.add_argument("--huiji-user-data-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--include-full-previews", action="store_true",
        help="Opt in to downloading full-set front/back previews for experiments.",
    )
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--download-delay", type=float, default=0.2)
    parser.add_argument(
        "--browser-verification",
        action="store_true",
        help=(
            "Open a visible persistent Chrome during refreshed Eorzea source "
            "collection and wait for manual Cloudflare verification."
        ),
    )
    parser.add_argument(
        "--browser-profile-dir",
        default="var/cache/eorzea_collection_browser",
        help="Persistent Chrome profile directory for browser verification.",
    )
    parser.add_argument(
        "--browser-wait-seconds",
        type=float,
        default=600.0,
        help="Maximum manual-verification wait per Eorzea page.",
    )
    parser.add_argument(
        "--release-mode",
        choices=("check", "publish"),
        default="check",
        help=(
            "check runs every release gate without rewriting current-best; "
            "publish exports assets only when release metrics are current."
        ),
    )
    parser.add_argument(
        "--allow-below-target",
        action="store_true",
        help="Allow release when metrics are current but the registered target is unmet.",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--resume-report",
        type=Path,
        help=(
            "Resume a prior lifecycle report. Passed sample-library stages "
            "with matching commands are reused and the original candidate "
            "suffix is preserved."
        ),
    )
    parser.add_argument(
        "--restart-stage",
        action="append",
        default=[],
        help=(
            "Force this sample-library stage and every following sample stage "
            "to rerun. May be repeated."
        ),
    )
    parser.add_argument("--discovery-mode", choices=("auto", "incremental", "full"), default="auto")
    parser.add_argument("--discovery-head-pages", type=int, default=3)
    parser.add_argument("--discovery-max-head-pages", type=int, default=8)
    parser.add_argument("--discovery-history-pages", type=int, default=3)
    return parser.parse_args(argv)


def python_command(script: str, *arguments: str) -> tuple[str, ...]:
    return (PYTHON, script, *arguments)


def run_artifact(args: argparse.Namespace, run_id: str, kind: str) -> Path:
    root = args.report_root if args.report_root.is_absolute() else PROJECT_ROOT / args.report_root
    return root / f"{run_id}.{kind}.json"


def sample_stages(args: argparse.Namespace, run_id: str) -> list[Stage]:
    stages: list[Stage] = []
    def add(name: str, label: str, script: str, *arguments: str, advisory: bool = False):
        stages.append(Stage(name, label, python_command(script, *arguments), "sample-library", advisory))
    def data(name: str, label: str, *arguments: str, advisory: bool = False):
        add(name, label, "scripts/run_data_pipeline.py", "--stage", name, *arguments, advisory=advisory)

    browser = []
    if args.browser_verification:
        browser = ["--browser-verification", "--browser-profile-dir", args.browser_profile_dir,
                   "--browser-wait-seconds", str(args.browser_wait_seconds)]
    refresh = ["--refresh-source"] if args.source_mode == "refresh" else []
    if args.source_mode != "local":
        data("discover", "增量发现装备图片来源", *refresh, *browser,
             "--discovery-mode", args.discovery_mode,
             "--discovery-head-pages", str(args.discovery_head_pages),
             "--discovery-max-head-pages", str(args.discovery_max_head_pages),
             "--discovery-history-pages", str(args.discovery_history_pages))
        data("build-image-manifest", "发现并验证新增图片 URL", *refresh, *browser)
        add("download-images", "下载新增与缺失装备图片", "scripts/download_eorzea_images_from_url_manifest.py",
            "--image-workers", str(args.image_workers), "--delay", str(args.download_delay),
            "--summary", str(run_artifact(args, run_id, "downloads")),
            *(["--include-full-previews"] if args.include_full_previews else []), *browser)
        data("collect-details", "关联图片的装备与部位标签", *browser, advisory=True)
    data("build-index", "整理图片清单并去除重复来源")
    data("validate-images", "完整解码与图片质量检查")
    data("build-sqlite", "关联图片与装备标签并入库")
    add("sample-eligibility", "生成合格样本与待处理图片清单", "scripts/build_equipment_image_samples.py")
    if args.candidate_mode != "skip":
        candidate = ["--suffix", args.candidate_suffix or run_id, "--allow-any-new-gearsets", "--skip-empty",
                     "--sample-manifest", SAMPLE_PATH.as_posix(), "--batch-size", str(args.batch_size),
                     "--device", args.device, "--summary", str(run_artifact(args, run_id, "candidates")),
                     "--source-selection", str(getattr(args, "source_selection_path", "data/current_best_selection.json"))]
        if args.candidate_source_suffix:
            candidate.extend(["--source-suffix", args.candidate_source_suffix])
        candidate.extend(["--dry-run"] if args.candidate_mode == "check" else ["--resume", "--activate"])
        add("candidate-indexes", "扩展并激活合格图片的检索索引", "scripts/update_active_candidate_indexes.py", *candidate)
        if args.candidate_mode == "extend":
            add("gallery-layout", "统一当前图库索引目录", "scripts/migrate_gallery_layout.py", "--apply", "--indexes-only")

    # Report enrichment must never block the image -> retrieval path.
    if args.source_mode == "refresh":
        add("cafemaker-zh", "补充物品名称与染色资料", "scripts/download_cafemaker_item_names.py", advisory=True)
        add("slot-metadata", "补充官方槽位资料", "scripts/download_cafemaker_equip_slot_categories.py", advisory=True)
        add("cafemaker-models", "补充同模判定字段", "scripts/download_cafemaker_item_models.py", advisory=True)
    if args.source_mode != "local":
        data("cafemaker-enrich-local", "补充中文名与装备限制", advisory=True)
        huiji = []
        if args.huiji_proxy:
            huiji.extend(["--huiji-proxy", args.huiji_proxy])
        if args.huiji_user_data_dir:
            huiji.extend(["--huiji-user-data-dir", args.huiji_user_data_dir])
        data("huiji-missing-zh", "补齐缺失中文名称", *huiji, advisory=True)
        add("refresh-display-database", "更新数据库展示资料", "scripts/build_eorzea_sqlite.py", advisory=True)
    if args.source_mode != "local" or (PROJECT_ROOT / CAFEMAKER_MODELS).is_file():
        add("build-same-model", "补充同模关系", "scripts/build_huiji_model_equivalence_full.py",
            "--items", CAFEMAKER_MODELS.as_posix(), advisory=True)
        add("select-same-model", "更新可见装备同模关系", "scripts/select_huiji_equivalence_relation.py",
            "--input", "data/processed/piece_appearance_equivalence_huiji_models.json",
            "--output", SAME_MODEL_OUTPUT.as_posix(), "--relation", "same_model", advisory=True)
    return stages


def release_stages(args: argparse.Namespace) -> list[Stage]:
    stages = []
    if not args.skip_tests:
        stages.append(
            Stage(
                "tests",
                "运行全量自动化测试",
                (PYTHON, "-m", "unittest", "discover", "-s", "tests"),
                "model-release",
            )
        )
    stages.extend(
        [
            Stage(
                "profile-check",
                "校验 current-best 推理配置",
                python_command(
                    "scripts/recognize_glamour.py",
                    "--profile",
                    "current-best",
                    "--check",
                ),
                "model-release",
            ),
            Stage(
                "image-coverage-check",
                "校验发布预览图片覆盖",
                python_command(
                    "scripts/download_eorzea_images_from_url_manifest.py", "--check"
                ),
                "model-release",
            ),
        ]
    )
    return stages


def run_stage(stage: Stage, *, dry_run: bool) -> StageResult:
    command = list(stage.command)
    print(f"\n[{stage.phase}] {stage.label}", flush=True)
    print("$ " + format_command(command), flush=True)
    if dry_run:
        return StageResult(
            stage.name,
            stage.label,
            stage.phase,
            "planned",
            None,
            0.0,
            command,
            [],
        )

    started = time.monotonic()
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        print(line, flush=True)
        lines.append(line)
        if len(lines) > 80:
            del lines[:-80]
    return_code = process.wait()
    duration = time.monotonic() - started
    status = "passed" if return_code == 0 else (
        "warning" if stage.advisory else "failed"
    )
    return StageResult(
        stage.name,
        stage.label,
        stage.phase,
        status,
        return_code,
        round(duration, 3),
        command,
        lines,
    )


def load_resume_report(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload.get("stages"), list):
        raise ValueError(f"Lifecycle resume report has no stages: {resolved}")
    return resolved, payload


def normalized_resume_command(command: Sequence[str]) -> list[str]:
    normalized = []
    skip = False
    for part in command:
        if skip:
            skip = False
            continue
        if part == "--summary":
            skip = True
        elif part != "--resume":
            normalized.append(portable_path(part) or part)
    return normalized


def candidate_suffix_from_report(payload: dict[str, Any]) -> str:
    explicit = str(payload.get("candidate_suffix") or "")
    if explicit:
        return explicit
    for row in payload.get("stages") or []:
        if not isinstance(row, dict) or row.get("name") != "candidate-indexes":
            continue
        command = [str(part) for part in row.get("command") or []]
        if "--suffix" in command:
            index = command.index("--suffix")
            if index + 1 < len(command):
                return command[index + 1]
    return str(payload.get("run_id") or "")


def reused_stage_result(
    stage: Stage,
    previous: dict[str, Any] | None,
) -> StageResult | None:
    if not previous or previous.get("status") not in {"passed", "reused"}:
        return None
    previous_command = [str(part) for part in previous.get("command") or []]
    if normalized_resume_command(previous_command) != normalized_resume_command(
        stage.command
    ):
        return None
    print(f"\n[{stage.phase}] {stage.label}", flush=True)
    print("reused from passed lifecycle report", flush=True)
    return StageResult(
        stage.name,
        stage.label,
        stage.phase,
        "reused",
        0,
        0.0,
        list(stage.command),
        ["reused from passed lifecycle report"],
    )


def metrics_gate(*, allow_below_target: bool) -> tuple[bool, list[str], dict[str, Any]]:
    path = PROJECT_ROOT / METRICS_PATH
    if not path.is_file():
        return False, [f"missing metrics snapshot: {METRICS_PATH.as_posix()}"], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if payload.get("metric_status") not in {"current", "validated", "release_ready"}:
        problems.append(
            "registered metrics are stale: "
            + str(payload.get("metric_status") or "missing status")
        )
    if not allow_below_target and not payload.get("target_achieved_all_slots"):
        problems.append("registered target is not achieved for every slot")
    return not problems, problems, payload


def release_asset_stage(
    args: argparse.Namespace, *, force_check: bool = False
) -> Stage:
    check_only = force_check or args.release_mode == "check"
    arguments = ["--check"] if check_only else []
    return Stage(
        "release-assets",
        "校验已发布 current-best 资产"
        if check_only
        else "导出并校验 current-best 资产",
        python_command("scripts/export_current_best_assets.py", *arguments),
        "model-release",
    )


def collect_summary(run_id: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    database = PROJECT_ROOT / "data/processed/eorzea.sqlite"
    if database.is_file():
        connection = sqlite3.connect(database)
        try:
            summary["sample_library"] = {
                "gearsets": scalar(connection, "SELECT COUNT(*) FROM gearsets"),
                "pieces": scalar(connection, "SELECT COUNT(*) FROM gear_pieces"),
                "images": scalar(connection, "SELECT COUNT(*) FROM images"),
                "qa_ok_images": scalar(
                    connection,
                    "SELECT COUNT(*) FROM images WHERE qa_status = 'ready'",
                ),
            }
        finally:
            connection.close()
    same_model_path = PROJECT_ROOT / SAME_MODEL_OUTPUT
    if same_model_path.is_file():
        payload = json.loads(same_model_path.read_text(encoding="utf-8"))
        equivalents = payload.get("same_model_equivalents") or {}
        summary["same_model"] = {
            "selected_relation": "same_model_including_main_model",
            "mapped_pieces": len(equivalents),
            "groups": count_equivalence_groups(equivalents),
            "source_mapped_pieces": int(payload.get("mapped_piece_count") or 0),
            "source_piece_count": int(payload.get("piece_count") or 0),
        }
    candidate_indexes = []
    indexes_root = PROJECT_ROOT / "data/indexes"
    if indexes_root.is_dir():
        for path in sorted(indexes_root.glob(f"*-{run_id}/summary.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidate_indexes.append(
                {
                    "path": str(path.parent.relative_to(PROJECT_ROOT)).replace(
                        "\\", "/"
                    ),
                    "source_candidate_count": int(
                        payload.get("source_candidate_count") or 0
                    ),
                    "new_candidate_count": int(
                        payload.get("new_candidate_count") or 0
                    ),
                    "candidate_count": int(payload.get("candidate_count") or 0),
                }
            )
    if candidate_indexes:
        summary["candidate_indexes"] = {
            "versioned_branches": len(candidate_indexes),
            "new_candidate_rows": sum(
                row["new_candidate_count"] for row in candidate_indexes
            ),
            "indexes": candidate_indexes,
        }
    asset_path = PROJECT_ROOT / ASSET_MANIFEST
    if asset_path.is_file():
        payload = json.loads(asset_path.read_text(encoding="utf-8"))
        files = payload.get("files") or []
        summary["release_assets"] = {
            "files": len(files),
            "bytes": sum(int(row.get("size") or 0) for row in files),
            "profile": payload.get("profile"),
        }
    return summary


def scalar(connection: sqlite3.Connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    return int(row[0]) if row else 0


def count_equivalence_groups(equivalents: dict[str, Any]) -> int:
    groups = {
        tuple(sorted({int(piece_id), *(int(value) for value in values)}))
        for piece_id, values in equivalents.items()
    }
    return len(groups)


def write_report(
    args: argparse.Namespace,
    run_id: str,
    started_at: str,
    results: list[StageResult],
    *,
    release_gate: dict[str, Any] | None,
    resumed_from: Path | None = None,
    finished: bool = True,
) -> tuple[Path, dict[str, Any]]:
    report_root = args.report_root
    if not report_root.is_absolute():
        report_root = PROJECT_ROOT / report_root
    report_root.mkdir(parents=True, exist_ok=True)
    failures = [row for row in results if row.status == "failed"]
    blocked = bool(release_gate and release_gate.get("ok") is False)
    all_planned = bool(results) and all(row.status == "planned" for row in results)
    status = (
        "failed"
        if failures
        else "blocked"
        if blocked
        else "planned"
        if all_planned
        else "passed"
    )
    payload = {
        "version": 3,
        "source_selection_path": str(getattr(args, "source_selection_path", "")),
        "baseline_path": str(getattr(args, "baseline_path", "")),
        "run_id": run_id,
        "workflow": args.workflow,
        "source_mode": args.source_mode,
        "discovery_mode": args.discovery_mode,
        "candidate_mode": args.candidate_mode,
        "candidate_suffix": args.candidate_suffix or run_id,
        "release_mode": args.release_mode,
        "resumed_from": (
            str(resumed_from.relative_to(PROJECT_ROOT)).replace("\\", "/")
            if resumed_from is not None and resumed_from.is_relative_to(PROJECT_ROOT)
            else str(resumed_from) if resumed_from is not None else None
        ),
        "status": status if finished else "running",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "stages": [asdict(row) for row in results],
        "release_gate": release_gate,
        "summary": collect_summary(args.candidate_suffix or run_id),
    }
    if finished and args.workflow in {"sample-library", "all"}:
        def evidence(kind):
            path = run_artifact(args, run_id, kind)
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not args.dry_run else {}
        before = getattr(args, "image_baseline", {})
        after = image_snapshot(PROJECT_ROOT, include_full_previews=args.include_full_previews) if not args.dry_run else {}
        payload["image_update"] = update_result(before, after, candidate=evidence("candidates"),
            download=evidence("downloads"), warnings=sum(r.status == "warning" for r in results),
            failed=bool(failures), planned=args.dry_run)
        if status == "passed" and (any(r.status == "warning" for r in results)
                or payload["image_update"]["missing_download_images"]
                or payload["image_update"]["outcome"] == "images_pending_index"):
            payload["status"] = "warning"
    report_path = report_root / f"{run_id}.json"
    write_json_atomic(report_path, payload)
    return report_path, payload


def print_report(report_path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    samples = summary.get("sample_library") or {}
    same_model = summary.get("same_model") or {}
    candidate_indexes = summary.get("candidate_indexes") or {}
    assets = summary.get("release_assets") or {}
    stages = payload.get("stages") or []
    passed = sum(row.get("status") == "passed" for row in stages)
    reused = sum(row.get("status") == "reused" for row in stages)
    planned = sum(row.get("status") == "planned" for row in stages)
    failed = sum(row.get("status") == "failed" for row in stages)
    print("\n=== FFXIV Prism 生命周期报告 ===")
    print(f"状态: {str(payload['status']).upper()}")
    print(
        f"阶段: 通过 {passed} / 复用 {reused} / 计划 {planned} / "
        f"失败 {failed} / 总计 {len(stages)}"
    )
    if samples:
        print(
            "样本库: "
            f"套装 {samples.get('gearsets', 0):,}，"
            f"部件 {samples.get('pieces', 0):,}，"
            f"图片 {samples.get('images', 0):,}，"
            f"QA通过 {samples.get('qa_ok_images', 0):,}"
        )
    if same_model:
        print(
            "同模关系: "
            f"{same_model.get('selected_relation') or 'unknown'}，"
            f"覆盖部件 {same_model.get('mapped_pieces', 0):,}，"
            f"关系组 {same_model.get('groups', 0):,}"
        )
    if candidate_indexes:
        print(
            "候选索引: "
            f"版本化分支 {candidate_indexes.get('versioned_branches', 0):,}，"
            f"新增向量行 {candidate_indexes.get('new_candidate_rows', 0):,}"
        )
    if assets:
        print(
            "发布资产: "
            f"文件 {assets.get('files', 0):,}，"
            f"大小 {assets.get('bytes', 0) / (1024 ** 3):.2f} GiB"
        )
    gate = payload.get("release_gate") or {}
    if gate and gate.get("ok") is False:
        print("发布门禁: BLOCKED")
        for problem in gate.get("problems") or []:
            print(f"  - {problem}")
    try:
        display_path = report_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = report_path
    print(f"JSON报告: {display_path}")
    if payload.get("image_update"):
        print("DATASET_RESULT " + json.dumps(payload["image_update"], ensure_ascii=False))


def format_command(command: Sequence[str]) -> str:
    return " ".join(
        f'"{part}"' if not part or any(char.isspace() for char in part) else part
        for part in command
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    results: list[StageResult] = []
    release_gate: dict[str, Any] | None = None
    hard_failure = False
    resume_path: Path | None = None
    previous_by_name: dict[str, dict[str, Any]] = {}
    previous_report: dict[str, Any] = {}
    if args.resume_report is not None:
        resume_path, previous_report = load_resume_report(args.resume_report)
        if args.candidate_suffix is None:
            args.candidate_suffix = candidate_suffix_from_report(previous_report)
            if not args.candidate_suffix:
                raise ValueError("Resume report has no candidate suffix or run ID")
        previous_by_name = {
            str(row.get("name")): row
            for row in previous_report.get("stages") or []
            if isinstance(row, dict) and row.get("name")
        }

    if args.workflow in {"sample-library", "all"}:
        if not args.dry_run:
            args.source_selection_path = Path(previous_report.get("source_selection_path") or run_artifact(args, run_id, "sources"))
            args.baseline_path = Path(previous_report.get("baseline_path") or run_artifact(args, run_id, "before"))
            if not args.source_selection_path.is_absolute():
                args.source_selection_path = PROJECT_ROOT / args.source_selection_path
            if not args.baseline_path.is_absolute():
                args.baseline_path = PROJECT_ROOT / args.baseline_path
            for path in (args.source_selection_path, args.baseline_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            if not args.source_selection_path.is_file():
                source = json.loads((PROJECT_ROOT / "data/current_best_selection.json").read_text(encoding="utf-8"))
                write_json_atomic(args.source_selection_path, source)
            if args.baseline_path.is_file():
                args.image_baseline = json.loads(args.baseline_path.read_text(encoding="utf-8"))
            else:
                args.image_baseline = image_snapshot(PROJECT_ROOT, include_full_previews=args.include_full_previews)
                write_json_atomic(args.baseline_path, args.image_baseline)
        reuse_invalidated = bool(previous_report and previous_report.get("version", 0) < 3)
        for stage in sample_stages(args, run_id):
            if stage.name in set(args.restart_stage):
                reuse_invalidated = True
            result = None
            if args.resume_report is not None and not reuse_invalidated:
                result = reused_stage_result(stage, previous_by_name.get(stage.name))
                if result is None:
                    reuse_invalidated = True
            if result is None:
                result = run_stage(stage, dry_run=args.dry_run)
            results.append(result)
            # Preserve the matching run evidence when a stage is reused.
            if result.status == "reused" and stage.name in {"download-images", "candidate-indexes"}:
                previous_command = previous_by_name.get(stage.name, {}).get("command", [])
                if "--summary" in previous_command:
                    old_path = Path(previous_command[previous_command.index("--summary") + 1])
                    if not old_path.is_absolute():
                        old_path = PROJECT_ROOT / old_path
                    kind = "downloads" if stage.name == "download-images" else "candidates"
                    if old_path.is_file():
                        write_json_atomic(run_artifact(args, run_id, kind), json.loads(old_path.read_text(encoding="utf-8")))
            write_report(args, run_id, started_at, results, release_gate=None, resumed_from=resume_path, finished=False)
            if result.status == "failed":
                hard_failure = True
                break

    if args.workflow in {"model-release", "all"} and not hard_failure:
        for stage in release_stages(args):
            result = run_stage(stage, dry_run=args.dry_run)
            results.append(result)
            if result.status == "failed":
                hard_failure = True
                break

        if not hard_failure:
            if args.dry_run:
                release_gate = {"ok": None, "problems": [], "status": "planned"}
            else:
                gate_ok, problems, metrics = metrics_gate(
                    allow_below_target=args.allow_below_target
                )
                release_gate = {
                    "ok": gate_ok,
                    "problems": problems,
                    "metrics_status": metrics.get("metric_status"),
                    "registered_at": metrics.get("registered_at"),
                    "target_achieved_all_slots": metrics.get(
                        "target_achieved_all_slots"
                    ),
                }
            result = run_stage(
                release_asset_stage(
                    args,
                    force_check=bool(
                        args.release_mode == "publish"
                        and release_gate
                        and release_gate.get("ok") is False
                    ),
                ),
                dry_run=args.dry_run,
            )
            results.append(result)
            hard_failure = result.status == "failed"

    report_path, payload = write_report(
        args,
        run_id,
        started_at,
        results,
        release_gate=release_gate,
        resumed_from=resume_path,
    )
    print_report(report_path, payload)
    return 0 if payload["status"] in {"passed", "warning", "planned"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
