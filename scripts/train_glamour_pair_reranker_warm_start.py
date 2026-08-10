from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import scripts.train_glamour_pair_reranker as base


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume-reranker", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    resume_path = known.resume_reranker
    original_class = base.PairReranker

    class WarmStartPairReranker(original_class):
        def __init__(
            self,
            dimension: int,
            hidden_dimension: int = 256,
            slot_dimension: int = 16,
            per_slot_heads: bool = False,
            dropout: float = 0.1,
        ) -> None:
            super().__init__(
                dimension,
                hidden_dimension=hidden_dimension,
                slot_dimension=slot_dimension,
                per_slot_heads=per_slot_heads,
                dropout=dropout,
            )
            payload = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=True,
            )
            if int(payload["dimension"]) != dimension:
                raise ValueError("Resume checkpoint dimension does not match")
            if bool(payload.get("per_slot_heads", False)) != per_slot_heads:
                raise ValueError(
                    "Resume checkpoint per-slot-head setting does not match"
                )
            self.load_state_dict(payload["state_dict"])

    base.PairReranker = WarmStartPairReranker
    sys.argv = [sys.argv[0], *remaining]
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
