from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME / "bootstrap"))
from portable import (atomic_json, configure_environment, install_package, check_install_root,
                      prepare_models, prepare_runtime, read_json, status, validate_combination,
                      cleanup_installation, installed_manifest, package_manifest)
from gpu_runtime import expose_optional, installation_choice, prepare_optional

def activate_paths(active, prepare=True):
    status(f"设置运行路径：程序 {active['app']}，图库 {active.get('resources') or '尚未导入'}")
    app = HOME / "app" / active["app"]
    resource = HOME / "resources" / active["resources"] if active.get("resources") else HOME
    configure_environment(HOME, resource)
    dependencies = prepare_runtime(HOME, app) if prepare else HOME / "runtime" / read_json(app / "manifests/runtime.json")["id"] / "site-packages"
    sys.path[:0] = [str(app / "src"), str(app), str(dependencies)]
    # Embedded CPython ignores PYTHONPATH. Persist explicit, relative package paths
    # for the worker and device-probe subprocesses; service is stopped before changes.
    pth = HOME / "runtime/python/python312._pth"
    if pth.exists():
        entries = ["python312.zip", ".", os.path.relpath(dependencies, pth.parent),
                   os.path.relpath(app / "src", pth.parent), os.path.relpath(app, pth.parent),
                   os.path.relpath(HOME / "bootstrap", pth.parent), "import site"]
        pth.write_text("\n".join(entries) + "\n", "utf-8")
    expose_optional(HOME, app)
    return app, resource

def selfcheck(active):
    status("正在自检运行环境与索引，首次加载可能需要数十秒……")
    app, resource = activate_paths(active)
    command = [sys.executable, str(HOME / "bootstrap/launch.py"), "probe"]
    subprocess.run(command, check=True, timeout=180, cwd=app)


def check_worker_entrypoint(app):
    # Exercise the same subprocess entry point used by the web UI. EOF exits
    # after imports, without loading models or submitting a recognition job.
    subprocess.run([sys.executable, "-B", str(app / "scripts/recognition_worker.py")],
                   stdin=subprocess.DEVNULL, check=True, timeout=60, cwd=app)

def check_gallery_inputs(active, filenames):
    """Reject ambiguous or incompatible input before extracting or downloading anything."""
    if len(filenames) > 1:
        raise ValueError("imports 中只能放一个图库 ZIP，请移走多余的 ZIP 后重试。")
    if not filenames:
        if not active.get("resources"):
            raise ValueError("未找到图库 ZIP。请将图库 ZIP 原样放入 imports，再运行安装环境.bat。")
        status("复用已导入的图库。")
        return
    app = installed_manifest(HOME, "app", active["app"])
    with zipfile.ZipFile(filenames[0]) as archive:
        manifest = package_manifest(archive, HOME)
    if manifest["kind"] != "resources":
        raise ValueError("请选择配套的图库 ZIP，不要放入程序安装包。")
    if manifest["compatibility"] not in app["resource_compatibility"]:
        raise ValueError("图库与当前程序不兼容，请下载配套图库。")


def install_inputs(active, filenames, requested_runtime=None):
    status("[1/4] 检查并导入图库")
    check_gallery_inputs(active, filenames)
    proposed = dict(active)
    for filename in filenames:
        manifest = install_package(HOME, filename)
        proposed["resources"] = manifest["id"]
    validate_combination(HOME, proposed, full=False)
    status("[2/4] Installing runtime and models (existing files will be reused)")
    app, _ = activate_paths(proposed)
    gpu_result = prepare_optional(HOME, app, requested_runtime)
    prepare_models(HOME, app)
    # Probe must receive the target, without publishing active.json prematurely.
    atomic_json(HOME / "work/proposed.json", proposed)
    status("[3/4] 自检运行环境与图库，请稍候……")
    log_path = HOME / "work/logs/install-check.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-B", str(HOME / "bootstrap/launch.py"), "probe", "--proposed"],
                       check=True, timeout=180, stdout=log, stderr=subprocess.STDOUT)
    atomic_json(HOME / "manifests/active.json", proposed)
    if gpu_result is False:
        print('CPU 和图库已就绪，可运行启动.bat。GPU 未就绪，下载文件保留；重跑安装环境.bat 选择 2 可继续。', flush=True)
        return 1
    status("[4/4] 清理安装下载与临时文件")
    cleanup_installation(HOME, app, filenames)
    print("安装成功，可运行“启动.bat”开始识别。", flush=True)
    return 0

def main():
    check_install_root(HOME)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "install", "repair", "probe"), default="start", nargs="?")
    parser.add_argument("packages", nargs="*")
    parser.add_argument("--proposed", action="store_true")
    parser.add_argument("--runtime", choices=("cpu", "gpu"))
    args = parser.parse_args()
    actions = {"start": "启动", "install": "安装环境", "repair": "修复", "probe": "环境自检"}
    status(f"FFXIV Prism · {actions[args.action]} · 初始化中")
    configure_environment(HOME)
    active_path = HOME / "manifests/active.json"
    active = read_json(HOME / "work/proposed.json" if args.proposed else active_path)
    if args.action == "probe":
        app, resource = activate_paths(active, prepare=False)
        status("加载 NumPy / PyTorch 与识别模块……")
        import numpy as np
        np.linalg.inv(np.eye(2, dtype=np.float32))
        import torch
        from ffxiv_prism.matching.inference import current_best_configs, inspect_assets
        from ffxiv_prism.runtime_contract import validate_runtime
        validate_runtime()
        from ffxiv_prism.matching.runtime.query import load_query_head_from_checkpoint
        from ffxiv_prism.matching.report_export import write_report_bundle
        web = runpy.run_path(str(app / "scripts/web_app.py"), run_name="prism_selfcheck")
        server = web["PrismServer"](("127.0.0.1", 0))
        server.server_close()
        check_worker_entrypoint(app)
        result = inspect_assets(current_best_configs())
        status("检查五部位索引与候选行数……")
        if not result["ok"]:
            raise ValueError(json.dumps(result, ensure_ascii=False))
        for config in current_best_configs().values():
            for folder in (config.index_dir, *config.ensemble_index_dirs):
                matrix = np.load(folder / "embeddings.npy", mmap_mode="r")
                with (folder / "metadata.jsonl").open(encoding="utf-8") as stream:
                    rows = sum(1 for line in stream if line.strip())
                if matrix.ndim != 2 or len(matrix) != rows:
                    raise ValueError(f"索引行数不匹配：{folder}")
        print(f"自检通过：Python {sys.version.split()[0]} / torch {torch.__version__}", flush=True)
        return 0
    # OS releases this byte-range lock even after a crash.
    import msvcrt
    with (HOME / "work/app.lock").open("a+b") as lock:
        if os.fstat(lock.fileno()).st_size == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError("程序已在运行。请先在原命令行按 Ctrl+C 停止，再安装或修复。")
        if args.action == "install":
            requested = args.runtime or installation_choice()
            packages = [Path(p).resolve() for p in args.packages] or sorted((HOME / "imports").glob("*.zip"))
            try:
                return install_inputs(active, packages, requested_runtime=requested)
            except Exception:
                if (HOME / "work/proposed.json").exists():
                    activate_paths(active, prepare=False)
                raise
        # The launcher can repair Python dependencies before inspecting resources.
        if args.action == "repair":
            status("开始完整修复：重新安装依赖并校验模型、程序及图库。")
            runtime_id = read_json(HOME / "app" / active["app"] / "manifests/runtime.json")["id"]
            (HOME / "runtime" / runtime_id / "ready.json").unlink(missing_ok=True)
        app, resource = activate_paths(active)
        gpu_result = None
        if args.action == "repair":
            from gpu_runtime import settings
            variant = settings(HOME).get('gpu')
            if variant in {'cu126', 'cu130'}:
                gpu_id = read_json(app / f'manifests/runtime-{variant}.json')['id']
                (HOME / 'runtime' / gpu_id / 'ready.json').unlink(missing_ok=True)
                gpu_result = prepare_optional(HOME, app)
        prepare_models(HOME, app, full=args.action == "repair")
        validate_combination(HOME, active, full=args.action == "repair")
        selfcheck(active)
        if args.action == "repair":
            if gpu_result is False:
                print("CPU 与图库自检通过，GPU 修复未完成；下载文件保留供重试。")
                return 1
            print("完整校验通过。")
            cleanup_installation(HOME, app)
            return 0
        status("正在启动本地网页服务，请等待访问地址……")
        process = subprocess.Popen([sys.executable, "-u", str(app / "scripts/web_app.py"), "--no-open"], cwd=app,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            with (HOME / "work/logs/server.log").open("a", encoding="utf-8") as log:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
            return process.wait()
        except KeyboardInterrupt:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=15)
            return 0
        finally:
            process.stdout.close()

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n准备失败：{exc}\n请保留此窗口查看原因；安装自检详情见 work/logs/install-check.log。", file=sys.stderr, flush=True)
        raise SystemExit(1)
