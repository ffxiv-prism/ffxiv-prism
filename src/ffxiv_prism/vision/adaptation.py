from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

DEFAULT_SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")


class SlotResidualAdapter(nn.Module):
    """A small slot-specific residual map from screenshot to render embeddings."""

    def __init__(
        self,
        dimension: int,
        rank: int,
        slots: Sequence[str] = DEFAULT_SLOTS,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.rank = rank
        self.slots = tuple(slots)
        self.adapters = nn.ModuleDict()
        for slot in self.slots:
            down = nn.Linear(dimension, rank, bias=False)
            up = nn.Linear(rank, dimension, bias=False)
            nn.init.normal_(down.weight, std=0.02)
            nn.init.zeros_(up.weight)
            self.adapters[slot] = nn.Sequential(down, nn.GELU(), up)

    def forward(self, embeddings: torch.Tensor, slot: str) -> torch.Tensor:
        residual = self.adapters[slot](embeddings)
        return torch.nn.functional.normalize(embeddings + residual, dim=-1)


def load_adapter_checkpoint(
    path: Path,
    device: str | torch.device = "cpu",
) -> tuple[SlotResidualAdapter, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    adapter = SlotResidualAdapter(
        dimension=int(payload["dimension"]),
        rank=int(payload["rank"]),
        slots=payload.get("slots", DEFAULT_SLOTS),
    )
    adapter.load_state_dict(payload["state_dict"])
    adapter.to(device).eval()
    return adapter, payload
