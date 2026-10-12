"""Maintainer-only: record exact locally tested models and verified public mirrors."""
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.parse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ffxiv_prism.model_assets import model_weight_files

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    models = []
    ids = ["facebook/dinov2-base", "google/siglip2-base-patch16-224", "usyd-community/vitpose-base-simple", "var/models/dinov3-vitb16-pretrain-lvd1689m"]
    for model_id in ids:
        folder = model_weight_files(model_id)[0].parent
        remote = "facebook/dinov3-vitb16-pretrain-lvd1689m" if model_id.startswith("var/") else model_id
        try:
            url = f"https://modelscope.cn/api/v1/models/{remote}/repo/files?Revision=master&Recursive=true"
            with urllib.request.urlopen(url, timeout=20) as response:
                data = json.load(response)
            mirrors = {row["Path"]: row for row in data["Data"]["Files"]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"No verified ModelScope mapping for {remote}: {exc}", flush=True)
            mirrors = {}
        rows = []
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith(".") or path.suffix not in {".json", ".safetensors", ".bin", ".model", ".txt", ".md"}:
                continue
            if path.name in {"README.md", "configuration.json"}:
                continue
            row = {"path": path.name, "size": path.stat().st_size, "sha256": sha(path), "urls": []}
            # HF snapshot folder records the exact revision that was tested.
            if folder.parent.name == "snapshots":
                row["urls"].extend([f"https://hf-mirror.com/{remote}/resolve/{folder.name}/{path.name}",
                                    f"https://huggingface.co/{remote}/resolve/{folder.name}/{path.name}"])
            mirror = mirrors.get(path.name)
            if mirror and mirror.get("Sha256") == row["sha256"] and mirror.get("Size") == row["size"]:
                row["urls"].insert(0, f"https://modelscope.cn/models/{remote}/resolve/{mirror['Revision']}/{path.name}")
            rows.append(row)
        models.append({"id": model_id, "directory": "models/" + remote.replace("/", "--"), "source_page": f"https://huggingface.co/{remote}", "files": rows})
        print(f"{remote}: {len(rows)} files, {sum(bool(r['urls']) for r in rows)} downloadable", flush=True)
    (ROOT / "manifests").mkdir(exist_ok=True)
    (ROOT / "manifests/models.json").write_text(json.dumps({"schema": 1, "models": models}, ensure_ascii=False, indent=2), "utf-8")

if __name__ == "__main__":
    main()
