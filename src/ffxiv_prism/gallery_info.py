"""Stable collection dates for the published, searchable equipment snapshot."""
import json
from pathlib import Path

from ffxiv_prism.paths import APP_ROOT, PROJECT_ROOT


def read_json(path):
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def gallery_info(resource_root=PROJECT_ROOT, app_root=APP_ROOT):
    resource_root, app_root = Path(resource_root), Path(app_root)
    installed = read_json(resource_root / "gallery.json")
    selection = read_json(resource_root / "data/current_best_selection.json")
    for path in sorted((app_root / "manifests/galleries").glob("*.json")):
        snapshot = read_json(path)
        if ((installed and installed.get("id") == snapshot.get("id"))
                or (not installed and selection.get("candidate_indexes")
                    and selection["candidate_indexes"] == snapshot.get("candidate_indexes"))):
            return snapshot
    # Legacy gallery.updated_at described index generation, not image collection.
    if installed:
        return {**installed, "updated_at": installed.get("samples_updated_at")}
    return {}
