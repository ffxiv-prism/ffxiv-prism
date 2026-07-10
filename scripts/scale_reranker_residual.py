from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.paths import portable_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create an inference-only reranker checkpoint with its nonlinear "
            "residual output scaled while preserving the cosine branch."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    if args.scale < 0:
        raise ValueError("scale must be non-negative")

    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    scaled = scale_checkpoint_payload(payload, args.scale)
    scaled["source_checkpoint"] = portable_path(args.input)
    scaled["inference_residual_scale"] = args.scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scaled, args.output)
    print(
        f"input={args.input} scale={args.scale} output={args.output}"
    )
    return 0


def scale_checkpoint_payload(
    payload: dict[str, Any],
    scale: float,
) -> dict[str, Any]:
    if scale < 0:
        raise ValueError("scale must be non-negative")
    output = copy.deepcopy(payload)
    state = output["state_dict"]
    if bool(output.get("per_slot_heads", False)):
        keys = [
            key
            for key in state
            if key.startswith("slot_heads.")
            and key.endswith((".weight", ".bias"))
        ]
    else:
        keys = [
            key
            for key in ("residual.5.weight", "residual.5.bias")
            if key in state
        ]
    if not keys:
        raise ValueError("Checkpoint has no recognized residual output layer")
    for key in keys:
        state[key] = state[key] * scale
    return output


if __name__ == "__main__":
    raise SystemExit(main())
