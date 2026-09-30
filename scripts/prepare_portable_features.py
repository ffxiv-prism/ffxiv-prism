"""Prepare the exact production HEAD candidate features; no player data."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ffxiv_prism.matching.asset_identity import asset_signature
from ffxiv_prism.matching.aligned_eyes import candidate_eye_boxes, legacy_asset_signature
from ffxiv_prism.matching.eye_accessory import CANDIDATE_BOXES, MODEL_IDS, encode_candidate_eyes
from ffxiv_prism.matching.eye_experts import EYE_COVERAGE_PROMPTS, classify_eye_coverage
from ffxiv_prism.matching.inference import current_best_configs
from ffxiv_prism.matching.runtime.encoders import build_encoder
from ffxiv_prism.cli.match import _classify_style_paths
from ffxiv_prism.vision.outfit_regions import DEFAULT_POSE_MODEL
from ffxiv_prism.paths import resolve_project_path
from ffxiv_prism.jsonl import read_jsonl

def migrate(paths, name, old_contract, new_contract, suffix, boxes=None):
    old = ROOT / "var/cache" / (name + "-v1")
    new = ROOT / "data/local_features" / (name + "-v2")
    new.mkdir(parents=True, exist_ok=True)
    hits = 0
    for path in paths:
        content = path.read_bytes()
        extra = json.dumps(("pixel-boxes-v1", boxes[str(path)])).encode() if boxes else b""
        src = old / (hashlib.sha256(old_contract + extra + content).hexdigest() + suffix)
        dst = new / (hashlib.sha256(new_contract + extra + content).hexdigest() + suffix)
        if src.is_file() and not dst.exists():
            shutil.copyfile(src, dst)
            hits += 1
    print(f"migrated {name}: {hits}", flush=True)
    return new

def main():
    config = current_best_configs()["HEAD"]
    paths = sorted({resolve_project_path(r["local_path"]) for r in read_jsonl(config.index_dir / "metadata.jsonl") if r.get("candidate_crop") == "full"})
    def contract(model, old, kind):
        assets = legacy_asset_signature(model) if old else asset_signature(model)
        if kind == "geometry":
            value = (1, model, assets, "full-image-box", "build_eye_regions-v1", .55, CANDIDATE_BOXES)
        elif kind == "coverage":
            value = (1, model, EYE_COVERAGE_PROMPTS, "full-rgb", assets)
        else:
            value = (1, model, "color", CANDIDATE_BOXES, assets)
        return json.dumps(value).encode()
    device = "cuda"
    folder = migrate(paths, "aligned-eye-geometry", contract(DEFAULT_POSE_MODEL, True, "geometry"), contract(DEFAULT_POSE_MODEL, False, "geometry"), ".json")
    geometry = candidate_eye_boxes(paths, device=device, batch_size=32, cache_dir=folder)
    boxes = {str(p): row["boxes"] for p, row in zip(paths, geometry, strict=True)}
    folder = migrate(paths, "eye-coverage", contract(MODEL_IDS[1], True, "coverage"), contract(MODEL_IDS[1], False, "coverage"), ".npy")
    # During build all missing entries belong in the candidate package.
    import ffxiv_prism.matching.eye_experts as coverage_module
    classify_eye_coverage(paths, classifier=_classify_style_paths, device=device, batch_size=32, cache_dir=folder)
    # Move any newly computed work-cache entries into the published directory.
    for p in (ROOT / "var/cache/eye-coverage-v2").glob("*.npy"):
        shutil.copyfile(p, folder / p.name)
    for model in MODEL_IDS:
        encoder = build_encoder(model, "auto", device, "color")
        for geometry_boxes in (None, boxes):
            folder = migrate(paths, "eye-accessory", contract(model, True, "eyes"), contract(model, False, "eyes"), ".npy", geometry_boxes)
            encode_candidate_eyes(encoder, paths, batch_size=32, cache_dir=folder, boxes_by_path=geometry_boxes)
        del encoder
    print(f"Complete: {len(paths)} HEAD catalogue images", flush=True)

if __name__ == "__main__":
    main()
