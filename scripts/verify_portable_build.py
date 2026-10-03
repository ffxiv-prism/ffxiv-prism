"""Local build smoke setup, deliberately using bundled CPython rather than Conda."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from ffxiv_prism.model_assets import model_weight_files
from scripts.build_portable_release import VERSION

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-home', type=Path, help='Read model/wheel caches from an existing installation without changing it.')
    parser.add_argument('--gallery-zip', type=Path)
    args = parser.parse_args()
    home = ROOT / "work/p"
    home.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / f"dist/FFXIV-Prism-{VERSION}-windows.zip") as archive:
        for info in archive.infolist():
            name = info.filename.removeprefix("FFXIV-Prism/")
            if name == "manifests/active.json" and (home / name).exists():
                continue
            target = home / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    (home / "work/downloads").mkdir(parents=True, exist_ok=True)
    wheel_cache = args.reuse_home / 'work/downloads' if args.reuse_home else ROOT / 'work/wheels'
    for wheel in wheel_cache.glob("*.whl"):
        target = home / "work/downloads" / wheel.name
        if not target.exists():
            shutil.copyfile(wheel, target)
    manifest = json.loads((ROOT / "manifests/models.json").read_text("utf-8"))
    for model in manifest["models"]:
        source = args.reuse_home / model['directory'] if args.reuse_home else model_weight_files(model["id"])[0].parent
        folder = home / model["directory"]
        folder.mkdir(parents=True, exist_ok=True)
        for row in model["files"]:
            target = folder / row["path"]
            if not target.exists():
                shutil.copyfile(source / row["path"], target)
    python = home / "runtime/python/python.exe"
    result = subprocess.run([str(python), "-X", "utf8", str(home / "bootstrap/launch.py"), "import",
                             str((args.gallery_zip or ROOT / "FFXIV-Prism-gallery-20260926-head-v2.zip").resolve())], cwd=home)
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
