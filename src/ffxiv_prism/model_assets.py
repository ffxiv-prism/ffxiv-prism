from __future__ import annotations

from pathlib import Path
from typing import Any

from ffxiv_prism.paths import CACHE_DIR, resolve_project_path


HUGGINGFACE_CACHE_DIR = CACHE_DIR / "huggingface" / "hub"


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

    local_model = resolve_project_path(model_id)
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

    source = str(model_id)
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
        return factory.from_pretrained(
            source,
            cache_dir=HUGGINGFACE_CACHE_DIR,
            **kwargs,
        )
