"""Local version metadata; never contacts a release server."""
from __future__ import annotations
import json
import subprocess
from ffxiv_prism.paths import APP_ROOT, PROJECT_ROOT
from ffxiv_prism.gallery_info import gallery_info

def version_info():
    def read(path):
        try:
            return json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return {}
    release = read(APP_ROOT / "manifests/release.json")
    gallery = gallery_info(PROJECT_ROOT, APP_ROOT)
    selection = read(PROJECT_ROOT / "data/current_best_selection.json")
    commit = release.get("code_commit")
    if not commit and (APP_ROOT / ".git").exists():
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=APP_ROOT,
                                             text=True, timeout=2, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {"release": release.get("version"), "code_commit": commit,
            "gallery_version": gallery.get("id"), "gallery_updated_at": gallery.get("updated_at"),
            "index_updated_at": selection.get("candidate_indexes_updated_at")}
