from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


# Initialize the supported Conda BLAS/OpenMP runtime before importing PyTorch.
np.linalg.inv(np.eye(2, dtype=np.float32))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.matching.inference import (  # noqa: E402
    current_best_configs,
    inspect_assets,
)
from ffxiv_prism.model_assets import load_pretrained  # noqa: E402


ASSET_MANIFEST = PROJECT_ROOT / "data" / "current_best_assets.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Git/LFS current-best assets and download public model "
            "weights into the project-local Hugging Face cache."
        )
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Do not download public models; only verify repository assets.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip SHA-256 verification of large Git/LFS files.",
    )
    args = parser.parse_args()

    integrity_problems = verify_manifest(check_hashes=not args.skip_hash_check)
    assets = inspect_assets(current_best_configs())
    if integrity_problems or not assets["ok"]:
        print(
            json.dumps(
                {
                    "ok": False,
                    "integrity_problems": integrity_problems,
                    "assets": assets,
                    "hint": (
                        "Run `git lfs install` and `git lfs pull`, then retry."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    downloaded = []
    if not args.check_only:
        downloaded = prepare_remote_models()
        assets = inspect_assets(current_best_configs())

    print(
        json.dumps(
            {
                "ok": assets["ok"],
                "integrity_problems": [],
                "prepared_models": downloaded,
                "assets": assets,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if assets["ok"] else 2


def prepare_remote_models() -> list[str]:
    from transformers import (
        AutoModel,
        AutoProcessor,
        VitPoseForPoseEstimation,
        VitPoseImageProcessorPil,
    )

    prepared = []
    components = (
        ("facebook/dinov2-base", (AutoModel,)),
        (
            "google/siglip2-base-patch16-224",
            (AutoProcessor, AutoModel),
        ),
        (
            "usyd-community/vitpose-base-simple",
            (VitPoseImageProcessorPil, VitPoseForPoseEstimation),
        ),
    )
    for model_id, factories in components:
        loaded = [load_pretrained(factory, model_id) for factory in factories]
        del loaded
        gc.collect()
        prepared.append(model_id)
        print(f"prepared_model={model_id}", file=sys.stderr, flush=True)
    return prepared


def verify_manifest(*, check_hashes: bool) -> list[str]:
    if not ASSET_MANIFEST.is_file():
        return [f"missing manifest: {ASSET_MANIFEST}"]
    try:
        manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"cannot read asset manifest: {exc}"]

    problems = []
    for row in manifest.get("files", []):
        relative = str(row.get("path") or "")
        path = PROJECT_ROOT / relative
        if not path.is_file():
            problems.append(f"missing: {relative}")
            continue
        if path.stat().st_size != int(row.get("size", -1)):
            problems.append(f"size mismatch (possible Git LFS pointer): {relative}")
            continue
        if check_hashes and sha256_file(path) != str(row.get("sha256")):
            problems.append(f"sha256 mismatch: {relative}")
    return problems


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
