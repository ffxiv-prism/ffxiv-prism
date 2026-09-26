from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ffxiv_prism.dataset.update_audit import build_sample_manifest

if __name__ == "__main__":
    print(json.dumps(build_sample_manifest(PROJECT_ROOT), ensure_ascii=False, indent=2))
