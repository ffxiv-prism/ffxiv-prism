"""Eye-local, zero-shot candidate embeddings with content-bound caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ffxiv_prism.model_assets import model_weight_files
from ffxiv_prism.paths import PROJECT_ROOT

EYE_NAMES = ("head_eye_left", "head_eye_right")
PRESENCE_NAMES = ("head_tight", "head_face_accessory")
CANDIDATE_BOXES = ((0.28, 0.36, 0.53, 0.72), (0.47, 0.36, 0.72, 0.72))
MODEL_IDS = ("facebook/dinov2-base", "google/siglip2-base-patch16-224")


def encode_candidate_eyes(encoder, paths, *, batch_size, cache_dir=None):
    """Cache image-content/weights/preprocessing-bound vectors, never ranks."""
    cache_dir = cache_dir or PROJECT_ROOT / "var/cache/eye-accessory-v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for weight in sorted(model_weight_files(encoder.model_id)):
        for path in [weight, *sorted(weight.parent.glob("*.json"))]:
            stat = path.stat()
            assets.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    contract = json.dumps(
        (1, encoder.model_id, "color", CANDIDATE_BOXES, assets)
    ).encode()
    keys = [hashlib.sha256(contract + Path(p).read_bytes()).hexdigest() for p in paths]
    found, missing = {}, {}
    for key, path in zip(keys, paths, strict=True):
        cache = cache_dir / (key + ".npy")
        if cache.is_file():
            try:
                value = np.load(cache, allow_pickle=False)
                if value.ndim == 2 and value.shape[0] == 2 and np.isfinite(value).all():
                    found[key] = value
                    continue
            except (OSError, ValueError, EOFError):
                pass
        missing[key] = path
    if missing:
        prepare = encoder.prepare_image
        matrices = []
        try:
            for box in CANDIDATE_BOXES:

                def crop(path, box=box):
                    image = prepare(path)
                    result = image.crop(
                        tuple(
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
