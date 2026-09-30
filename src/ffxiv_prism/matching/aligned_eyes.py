"""Content-cached eye landmarks for catalogue images, in the query crop frame."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

from ffxiv_prism.matching.eye_accessory import CANDIDATE_BOXES
from ffxiv_prism.model_assets import model_weight_files
from ffxiv_prism.paths import PROJECT_ROOT


def asset_signature(model_id):
    assets = []
    for weight in sorted(model_weight_files(model_id)):
        for path in [weight, *sorted(weight.parent.glob("*.json"))]:
            stat = path.stat()
            assets.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return assets


def atomic_json(path, value):
    with NamedTemporaryFile(dir=path.parent, suffix=".json", delete=False,
                            mode="w", encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(value, stream)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def boxes_for_pose(keypoints, image_size):
    """Use the same geometry/quality rules as queries; fall back as a pair."""
    from ffxiv_prism.vision.outfit_regions import build_eye_regions

    regions = build_eye_regions(keypoints, image_size)
    aligned = all(r.visibility == "visible" and r.confidence >= .55 for r in regions)
    boxes = [r.box for r in regions] if aligned else [
        tuple(round(v * image_size[i % 2]) for i, v in enumerate(box))
        for box in CANDIDATE_BOXES
    ]
    return dict(boxes=boxes, aligned=aligned)


def candidate_eye_boxes(paths, *, device, batch_size=32, cache_dir=None):
    """Estimate each distinct image once; no identities or labels are inputs."""
    from ffxiv_prism.vision.outfit_regions import (
        DEFAULT_POSE_MODEL, Keypoint, SinglePersonPoseEstimator,
    )
    import torch

    cache_dir = cache_dir or PROJECT_ROOT / "var/cache/aligned-eye-geometry-v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    contract = json.dumps((1, DEFAULT_POSE_MODEL, asset_signature(DEFAULT_POSE_MODEL),
                           "full-image-box", "build_eye_regions-v1", .55,
                           CANDIDATE_BOXES)).encode()
    keys = [hashlib.sha256(contract + Path(p).read_bytes()).hexdigest() for p in paths]
    found, missing = {}, {}
    for key, path in zip(keys, paths, strict=True):
        try:
            value = json.loads((cache_dir / (key + ".json")).read_text("utf-8"))
            boxes = value["boxes"]
            if (len(boxes) == 2 and isinstance(value["aligned"], bool)
                    and all(len(b) == 4 and all(isinstance(v, int) for v in b)
                            and b[2] > b[0] >= 0 and b[3] > b[1] >= 0 for b in boxes)):
                found[key] = value
                continue
        except (OSError, ValueError, KeyError, TypeError):
            pass
        missing[key] = path
    if missing:
        estimator = SinglePersonPoseEstimator(model_id=DEFAULT_POSE_MODEL, device=str(device))
        items = list(missing.items())
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            images = []
            try:
                for _, path in batch:
                    with Image.open(path) as source:
                        images.append(source.convert("RGB"))
                boxes = [[[0., 0., float(im.width), float(im.height)]] for im in images]
                inputs = estimator.processor(images=images, boxes=boxes, return_tensors="pt")
                inputs = {k: v.to(estimator.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = estimator.model(**inputs)
                results = estimator.processor.post_process_pose_estimation(outputs, boxes)
                for (key, _), im, result in zip(batch, images, results, strict=True):
                    result = result[0]
                    points = [Keypoint(float(x), float(y), float(score))
                              for (x, y), score in zip(result["keypoints"].cpu().tolist(),
                                                      result["scores"].cpu().tolist(), strict=True)]
                    value = boxes_for_pose(points, im.size)
                    found[key] = value
                    atomic_json(cache_dir / (key + ".json"), value)
            finally:
                for im in images:
                    im.close()
            print(f"aligned_eye_geometry={min(start + batch_size, len(items))}/{len(items)}", flush=True)
    return [found[key] for key in keys]
