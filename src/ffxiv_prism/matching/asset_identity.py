"""Content identities stay valid across installs, moves and ZIP timestamps."""
import hashlib
from functools import lru_cache
from ffxiv_prism.model_assets import model_weight_files

@lru_cache(maxsize=32)
def asset_signature(model_id):
    assets = []
    for weight in sorted(model_weight_files(model_id)):
        for path in [weight, *sorted(p for p in weight.parent.glob("*.json") if not p.name.endswith(".verified.json"))]:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            assets.append((path.name, path.stat().st_size, digest))
    if not assets:
        raise FileNotFoundError(f"Model assets unavailable: {model_id}")
    return assets

def candidate_cache(name):
    from ffxiv_prism.paths import PROJECT_ROOT, CACHE_DIR
    published = PROJECT_ROOT / "data/local_features" / name
    return published if published.is_dir() else CACHE_DIR / name

def require_candidate_cache(paths):
    import os
    from pathlib import Path
    from ffxiv_prism.paths import EORZEA_IMAGE_DIR
    if os.environ.get("PRISM_PORTABLE") and any(Path(p).is_relative_to(EORZEA_IMAGE_DIR) for p in paths):
        raise RuntimeError("候选局部特征缺失或不兼容，请重新导入完整图库资源包；用户端不会重建图库。")
