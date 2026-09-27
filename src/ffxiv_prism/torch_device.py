from __future__ import annotations

from typing import Any

import torch


def resolve_torch_device(
    requested: str | torch.device | None = "auto",
    *,
    torch_module: Any = torch,
) -> torch.device:
    """Resolve an explicit device or select the best available accelerator."""
    if isinstance(requested, torch.device):
        device = requested
    else:
        value = str(requested or "auto").strip().lower()
        if value == "auto":
            if torch_module.cuda.is_available():
                value = "cuda"
            elif (
                hasattr(torch_module.backends, "mps")
                and torch_module.backends.mps.is_available()
            ):
                value = "mps"
            else:
                value = "cpu"
        device = torch.device(value)

    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this PyTorch environment. "
            "Use --device auto or --device cpu, or install the GPU environment."
        )
    if device.type == "mps" and not (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        raise RuntimeError(
            "MPS was requested but is not available. "
            "Use --device auto or --device cpu."
        )
    return device
