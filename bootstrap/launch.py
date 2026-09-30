from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME / "bootstrap"))
from portable import (atomic_json, configure_environment, installed_manifest, install_package,
                      prepare_models, prepare_runtime, read_json, validate_combination)

def activate_paths(active, prepare=True):
    app = HOME / "app" / active["app"]
    resource = HOME / "resources" / active["resources"] if active.get("resources") else HOME
    configure_environment(HOME, resource)
    dependencies = prepare_runtime(HOME, app) if prepare else HOME / "runtime/envs" / read_json(app / "manifests/runtime.json")["id"] / "site-packages"
    sys.path[:0] = [str(app / "src"), str(app), str(dependencies)]
    # Embedded CPython ignores PYTHONPATH. Persist explicit, relative package paths
    # for the worker and device-probe subprocesses; service is stopped before changes.
    pth = HOME / "runtime/python/python312._pth"
    if pth.exists():
        entries = ["python312.zip", ".", os.path.relpath(dependencies, pth.parent),
                   os.path.relpath(app / "src", pth.parent), os.path.relpath(app, pth.parent),
                   os.path.relpath(HOME / "bootstrap", pth.parent), "import site"]
        pth.write_text("\n".join(entries) + "\n", "utf-8")
    return app, resource

def selfcheck(active):
    app, resource = activate_paths(active)
    command = [sys.executable, str(HOME / "bootstrap/launch.py"), "probe"]
    subprocess.run(command, check=True, timeout=180, cwd=app)

def install_inputs(active, filenames):
    proposed = dict(active)
    kinds = set()
    for filename in filenames:
        manifest = install_package(HOME, filename)
        if manifest["kind"] in kinds:
            raise ValueError("一次只能导入一个程序包和一个图库包，请移走 imports 中的旧版本 ZIP。")
        kinds.add(manifest["kind"])
        proposed["app" if manifest["kind"] == "app" else "resources"] = manifest["id"]
    validate_combination(HOME, proposed, full=False)
    app, _ = activate_paths(proposed)
    prepare_models(HOME, app, full=True)
    # Probe must receive the target, without publishing active.json prematurely.
    atomic_json(HOME / "work/proposed.json", proposed)
    subprocess.run([sys.executable, str(HOME / "bootstrap/launch.py"), "probe", "--proposed"], check=True, timeout=180)
    if active.get("resources") and active != proposed:
        atomic_json(HOME / "manifests/previous.json", active)
    atomic_json(HOME / "manifests/active.json", proposed)
    print("安装成功。上一版本已保留，可用“回退.bat”恢复。", flush=True)
    return proposed

def main():
    if not str(HOME).isascii():
        raise ValueError("请将整个项目移到不包含中文的短英文路径，例如 D:\\FFXIV-Prism，然后重试。")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "import", "repair", "rollback", "probe"), default="start", nargs="?")
    parser.add_argument("packages", nargs="*")
    parser.add_argument("--proposed", action="store_true")
    args = parser.parse_args()
    configure_environment(HOME)
    active_path = HOME / "manifests/active.json"
    active = read_json(HOME / "work/proposed.json" if args.proposed else active_path)
    if args.action == "probe":
        app, resource = activate_paths(active, prepare=False)
        import numpy as np
        np.linalg.inv(np.eye(2, dtype=np.float32))
        import torch
        from ffxiv_prism.matching.inference import current_best_configs, inspect_assets
        from ffxiv_prism.matching.runtime.query import load_query_head_from_checkpoint
        from ffxiv_prism.matching.report_export import write_report_bundle
        web = runpy.run_path(str(app / "scripts/web_app.py"), run_name="prism_selfcheck")
        server = web["PrismServer"](("127.0.0.1", 0))
        server.server_close()
        result = inspect_assets(current_best_configs())
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
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError("程序已在运行。请先在原命令行按 Ctrl+C 停止，再导入、修复或回退。")
        if args.action == "rollback":
            previous = read_json(HOME / "manifests/previous.json")
            validate_combination(HOME, previous)
            atomic_json(HOME / "work/proposed.json", previous)
            activate_paths(previous)
            subprocess.run([sys.executable, str(HOME / "bootstrap/launch.py"), "probe", "--proposed"], check=True, timeout=180)
            atomic_json(HOME / "manifests/previous.json", active)
            atomic_json(active_path, previous)
            print("已恢复上一版本。请重新双击启动。")
            return 0
        if args.action == "import":
            packages = [Path(p).resolve() for p in args.packages] or sorted((HOME / "imports").glob("*.zip"))
            if not packages:
                raise ValueError("请将维护者提供的 ZIP 放入 imports，再运行导入更新。")
            try:
                active = install_inputs(active, packages)
            except Exception:
                activate_paths(active, prepare=False)
                raise
            return 0
        # The launcher can repair Python dependencies before inspecting resources.
        if args.action == "repair":
            runtime_id = read_json(HOME / "app" / active["app"] / "manifests/runtime.json")["id"]
            (HOME / "runtime/envs" / runtime_id / "ready.json").unlink(missing_ok=True)
        app, resource = activate_paths(active)
        prepare_models(HOME, app, full=args.action == "repair")
        validate_combination(HOME, active, full=args.action == "repair")
        selfcheck(active)
        if args.action == "repair":
            print("完整校验通过。")
            return 0
        pending = HOME / "work/pending-update.json"
        startup_fallback = None
        while True:
            app, _ = activate_paths(active)
            process = subprocess.Popen([sys.executable, str(app / "scripts/web_app.py"), "--no-open"], cwd=app,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            try:
                with (HOME / "work/logs/server.log").open("a", encoding="utf-8") as log:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                        if "本地网页已启动" in line:
                            startup_fallback = None
                returncode = process.wait()
            except KeyboardInterrupt:
                # Console Ctrl+C is delivered to both parent and child.
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                return 0
            if returncode == 75 and pending.is_file():
                filenames = read_json(pending)["packages"]
                pending.unlink()
                try:
                    startup_fallback = dict(active)
                    active = install_inputs(active, filenames)
                except Exception as exc:
                    print(f"更新失败，保留原版本：{exc}", flush=True)
                    activate_paths(active, prepare=False)
                    startup_fallback = None
                continue
            if returncode and startup_fallback is not None:
                print("新版本服务启动失败，恢复上一版本。", flush=True)
                active, startup_fallback = startup_fallback, None
                atomic_json(active_path, active)
                activate_paths(active, prepare=False)
                continue
            return returncode

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n准备失败：{exc}\n请保留此窗口查看原因。", file=sys.stderr, flush=True)
        raise SystemExit(1)
