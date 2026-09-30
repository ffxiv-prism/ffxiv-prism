"""Local build smoke setup, deliberately using bundled CPython rather than Conda."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ffxiv_prism.model_assets import model_weight_files

def main():
    home = ROOT / "work/p"
    home.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "dist/FFXIV-Prism-v0.2.0-beta.1-windows.zip") as archive:
        for info in archive.infolist():
            name = info.filename.removeprefix("FFXIV-Prism/")
            if name == "manifests/active.json" and (home / name).exists():
                continue
            target = home / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    (home / "work/downloads").mkdir(parents=True, exist_ok=True)
    for wheel in (ROOT / "work/wheels").glob("*.whl"):
        target = home / "work/downloads" / wheel.name
        if not target.exists():
            shutil.copyfile(wheel, target)
    manifest = json.loads((ROOT / "manifests/models.json").read_text("utf-8"))
    for model in manifest["models"]:
        source = model_weight_files(model["id"])[0].parent
        folder = home / model["directory"]
        folder.mkdir(parents=True, exist_ok=True)
        for row in model["files"]:
            target = folder / row["path"]
            if not target.exists():
                shutil.copyfile(source / row["path"], target)
    python = home / "runtime/python/python.exe"
    result = subprocess.run([str(python), "-X", "utf8", str(home / "bootstrap/launch.py"), "import",
                             str(ROOT / "FFXIV-Prism-gallery-20260926-head-v2.zip")], cwd=home)
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
