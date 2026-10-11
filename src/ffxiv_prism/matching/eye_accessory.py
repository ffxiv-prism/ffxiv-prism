"""Eye-local, zero-shot candidate embeddings with content-bound caching."""

from __future__ import annotations

import hashlib
from io import BytesIO
from ffxiv_prism.matching.runtime.feature_store import read_feature_batch
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ffxiv_prism.model_assets import model_weight_files
from ffxiv_prism.paths import PROJECT_ROOT
from ffxiv_prism.matching.asset_identity import asset_signature, candidate_cache, require_candidate_cache
from ffxiv_prism.matching.runtime.session import read_image_bytes, timed

EYE_NAMES = ("head_eye_left", "head_eye_right")
PRESENCE_NAMES = ("head_tight", "head_face_accessory")
CANDIDATE_BOXES = ((0.28, 0.36, 0.53, 0.72), (0.47, 0.36, 0.72, 0.72))
MODEL_IDS = ("facebook/dinov2-base", "google/siglip2-base-patch16-224")


@timed("candidate_eye_features")
def encode_candidate_eyes(encoder, paths, *, batch_size, cache_dir=None, boxes_by_path=None):
    """Cache image-content/weights/preprocessing-bound vectors, never ranks."""
    cache_dir = cache_dir or candidate_cache("eye-accessory-v2")
    cache_dir.mkdir(parents=True, exist_ok=True)
    assets = asset_signature(encoder.model_id)
    contract = json.dumps(
        (1, encoder.model_id, getattr(encoder, "preprocess", "color"), CANDIDATE_BOXES, assets)
    ).encode()
    keys = [hashlib.sha256(contract + (
        json.dumps(("pixel-boxes-v1", boxes_by_path[str(p)])).encode()
        if boxes_by_path is not None else b""
    ) + read_image_bytes(p)).hexdigest() for p in paths]
    packed = read_feature_batch(cache_dir, keys, ".npy")
    found, missing = {}, {}
    for key, path in zip(keys, paths, strict=True):
        cache = cache_dir / (key + ".npy")
        if key in packed:
            try:
                value = np.load(BytesIO(packed[key]), allow_pickle=False)
                if value.ndim == 2 and value.shape[0] == 2 and np.isfinite(value).all():
                    found[key] = value
                    continue
            except (OSError, ValueError, EOFError):
                pass
        missing[key] = path
    if missing:
        require_candidate_cache(missing.values())
        prepare = encoder.prepare_image
        matrices = []
        try:
            for eye, box in enumerate(CANDIDATE_BOXES):

                def crop(path, box=box, eye=eye):
                    image = prepare(path)
                    result = image.crop(
                        boxes_by_path[str(path)][eye] if boxes_by_path is not None else tuple(
                            round(v * (image.width if i % 2 == 0 else image.height))
                            for i, v in enumerate(box)
                        )
                    )
                    image.close()
                    return result

                encoder.prepare_image = crop
                matrices.append(
                    encoder.encode_paths(list(missing.values()), batch_size)
                )
        finally:
            encoder.prepare_image = prepare
        for key, value in zip(missing, np.stack(matrices, axis=1), strict=True):
            found[key] = value
            with NamedTemporaryFile(
                dir=cache_dir, suffix=".npy", delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.save(stream, value, allow_pickle=False)
            try:
                temporary.replace(cache_dir / (key + ".npy"))
            finally:
                temporary.unlink(missing_ok=True)
    return np.stack([found[key] for key in keys])
