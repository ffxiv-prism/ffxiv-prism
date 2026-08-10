from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.matching.inference import current_best_configs  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402


SOURCE_MODEL_DIR = PROJECT_ROOT / "data" / "models"
RUNTIME_MODEL_DIR = SOURCE_MODEL_DIR / "current-best"
ASSET_MANIFEST = PROJECT_ROOT / "data" / "current_best_assets.json"
CHECKPOINT_NAMES = (
    "appearance_disjoint_cv_v52_fold01_qv64_closed_e10.pt",
    "appearance_disjoint_cv_v97_body_fold01_dinov3_e10.pt",
    "appearance_disjoint_cv_v54_head_fold01_qv65_e10.pt",
    "appearance_disjoint_cv_v71_head_fold01_dinov3_e3.pt",
    "appearance_disjoint_cv_v59_hands_fold01_qv71_v5_e10.pt",
    "appearance_disjoint_cv_v60_legs_fold01_qv72_e10.pt",
    "appearance_disjoint_cv_v61_feet_fold01_qv73_e10.pt",
)
MORPHOLOGY_NAMES = (
    "head_morphology_v73_fold01_qv65_upper_binary_ridge.npz",
)
INFERENCE_CHECKPOINT_KEYS = (
    "version",
    "model_id",
    "unfreeze_blocks",
    "trainable_state_dict",
    "trainable_parameter_names",
    "slot_attention",
    "adapter_rank",
    "query_head_type",
    "pooler_state_dict",
    "best_epoch",
    "fixed_epochs",
    "seed",
    "pose_normalize_hands",
)
LOCAL_DINOV3_FILES = (
    "config.json",
    "configuration.json",
    "LICENSE.md",
    "model.safetensors",
    "preprocessor_config.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export inference-only current-best checkpoints and write the "
            "portable runtime asset manifest."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate exported assets without rewriting them.",
    )
    args = parser.parse_args()

    if not args.check:
        export_inference_checkpoints()
        copy_small_model_assets()
        write_asset_manifest()

    problems = validate_exported_assets()
    payload = {
        "ok": not problems,
        "manifest": portable_path(ASSET_MANIFEST),
        "problems": problems,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not problems else 2


def inference_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"model_id", "trainable_state_dict"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "Training checkpoint is missing required keys: " + ", ".join(missing)
        )
    return {
        key: payload[key]
        for key in INFERENCE_CHECKPOINT_KEYS
        if key in payload
    }


def export_inference_checkpoints() -> None:
    import torch

    RUNTIME_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_NAMES:
        source = SOURCE_MODEL_DIR / name
        destination = RUNTIME_MODEL_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"Source checkpoint is missing: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=True)
        slim = inference_checkpoint_payload(payload)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(slim, temporary)
        temporary.replace(destination)


def copy_small_model_assets() -> None:
    RUNTIME_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in MORPHOLOGY_NAMES:
        source = SOURCE_MODEL_DIR / name
        destination = RUNTIME_MODEL_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"Source model is missing: {source}")
        shutil.copy2(source, destination)


def runtime_asset_paths() -> list[Path]:
    configs = current_best_configs()
    index_dirs = sorted(
        {
            path
            for config in configs.values()
            for path in (config.index_dir, *config.ensemble_index_dirs)
        },
        key=lambda path: path.as_posix(),
    )
    paths = [
        path
        for directory in index_dirs
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file()
    ]
    paths.extend(RUNTIME_MODEL_DIR / name for name in CHECKPOINT_NAMES)
    paths.extend(RUNTIME_MODEL_DIR / name for name in MORPHOLOGY_NAMES)
    dinov3 = PROJECT_ROOT / "var" / "models" / "dinov3-vitb16-pretrain-lvd1689m"
    paths.extend(dinov3 / name for name in LOCAL_DINOV3_FILES)
    return paths


def write_asset_manifest() -> None:
    files = []
    for path in runtime_asset_paths():
        if not path.is_file():
            raise FileNotFoundError(f"Runtime asset is missing: {path}")
        files.append(
            {
                "path": portable_path(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "version": 1,
        "profile": "current-best",
        "files": files,
        "remote_models": [
            "facebook/dinov2-base",
            "google/siglip2-base-patch16-224",
            "usyd-community/vitpose-base-simple",
        ],
        "user_downloaded_data": "data/raw/eorzea_collection/images",
    }
    ASSET_MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_exported_assets() -> list[str]:
    problems: list[str] = []
    if not ASSET_MANIFEST.is_file():
        return [f"missing manifest: {ASSET_MANIFEST}"]
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    for row in manifest.get("files", []):
        path = PROJECT_ROOT / str(row["path"])
        if not path.is_file():
            problems.append(f"missing: {row['path']}")
            continue
        if path.stat().st_size != int(row["size"]):
            problems.append(f"size mismatch: {row['path']}")
        if sha256_file(path) != str(row["sha256"]):
            problems.append(f"sha256 mismatch: {row['path']}")

    import torch

    allowed = set(INFERENCE_CHECKPOINT_KEYS)
    for name in CHECKPOINT_NAMES:
        path = RUNTIME_MODEL_DIR / name
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        extra = sorted(set(payload) - allowed)
        if extra:
            problems.append(f"checkpoint contains training-only keys: {name}: {extra}")
        if "optimizer_state_dict" in payload or "history" in payload:
            problems.append(f"checkpoint was not slimmed: {name}")
    return problems


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
