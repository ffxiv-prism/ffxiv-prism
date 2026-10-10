from __future__ import annotations

from pathlib import Path
import os
import json
from typing import Any

from ffxiv_prism.paths import CACHE_DIR, HOME_ROOT, APP_ROOT, resolve_project_path


HUGGINGFACE_CACHE_DIR = CACHE_DIR / "huggingface" / "hub"

def portable_model_path(model_id: str) -> Path | None:
    if not os.environ.get("PRISM_HOME"):
        return None
    manifest = APP_ROOT / "manifests/models.json"
    if manifest.is_file():
        for row in json.loads(manifest.read_text("utf-8"))["models"]:
            if row["id"] == model_id:
                return HOME_ROOT / row["directory"]
    return None


def is_remote_model_id(model_id: str) -> bool:
    """Return whether a model identifier can be downloaded from a model hub."""

    normalized = str(model_id).replace("\\", "/")
    if normalized.startswith(("./", "../", "data/", "var/")):
        return False
    if Path(normalized).is_absolute():
        return False
    return normalized.count("/") == 1


def model_weight_files(model_id: str) -> list[Path]:
    """Find project-local or Hugging Face cached weight files for a model."""

    local_model = portable_model_path(model_id) or resolve_project_path(model_id)
    if local_model is not None and local_model.is_dir():
        return [
            path
            for name in ("model.safetensors", "pytorch_model.bin")
            if (path := local_model / name).is_file()
        ]
    model_cache = (
        HUGGINGFACE_CACHE_DIR / f"models--{model_id.replace('/', '--')}"
    )
    return [
        path
        for filename in ("model.safetensors", "pytorch_model.bin")
        for path in model_cache.glob(f"snapshots/*/{filename}")
    ]


def load_pretrained(
    factory: Any,
    model_id: str,
    **kwargs: Any,
) -> Any:
    """Load from the project cache first and download remote models if absent."""

    # AutoProcessor otherwise silently selects different resize implementations
    # according to whether torchvision happens to be installed.
    if getattr(factory, "__name__", "") == "AutoProcessor":
        kwargs.setdefault("backend", "torchvision")
    source = str(model_id)
    portable = portable_model_path(source)
    if portable is not None:
        loaded = factory.from_pretrained(str(portable), local_files_only=True, **kwargs)
        if hasattr(loaded, "config"):
            loaded.config._name_or_path = source
        return loaded
    if not is_remote_model_id(source):
        resolved = resolve_project_path(source)
        if resolved is not None:
            source = str(resolved)
        return factory.from_pretrained(
            source,
            cache_dir=HUGGINGFACE_CACHE_DIR,
            local_files_only=True,
            **kwargs,
        )

    try:
        return factory.from_pretrained(
            source,
            cache_dir=HUGGINGFACE_CACHE_DIR,
            local_files_only=True,
            **kwargs,
        )
    except OSError:
        if os.environ.get("PRISM_PORTABLE"):
            raise
        return factory.from_pretrained(
            source,
            cache_dir=HUGGINGFACE_CACHE_DIR,
            **kwargs,
        )
