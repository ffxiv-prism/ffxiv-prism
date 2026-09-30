"""Shared inference primitives; training entry points live in scripts."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch import nn
SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")


class PairReranker(nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden_dimension: int = 256,
        slot_dimension: int = 16,
        per_slot_heads: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dimension = dimension
        self.per_slot_heads = per_slot_heads
        self.dropout = dropout
        self.slot_embeddings = nn.ParameterDict(
            {
                slot: nn.Parameter(torch.randn(slot_dimension) * 0.02)
                for slot in SLOTS
            }
        )
        input_dimension = dimension * 2 + 1 + slot_dimension
        if per_slot_heads:
            self.residual = nn.Sequential(
                nn.Linear(input_dimension, hidden_dimension),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dimension, hidden_dimension // 2),
                nn.GELU(),
            )
            self.slot_heads = nn.ModuleDict(
                {
                    slot: nn.Linear(hidden_dimension // 2, 1)
                    for slot in SLOTS
                }
            )
            for head in self.slot_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            self.base_scales = nn.ParameterDict(
                {
                    slot: nn.Parameter(torch.tensor(10.0))
                    for slot in SLOTS
                }
            )
            self.base_scale = None
        else:
            self.residual = nn.Sequential(
                nn.Linear(input_dimension, hidden_dimension),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dimension, hidden_dimension // 2),
                nn.GELU(),
                nn.Linear(hidden_dimension // 2, 1),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
            self.slot_heads = None
            self.base_scales = None
            self.base_scale = nn.Parameter(torch.tensor(10.0))

    def forward(
        self,
        queries: torch.Tensor,
        candidates: torch.Tensor,
        slot: str,
    ) -> torch.Tensor:
        if queries.shape != candidates.shape:
            raise ValueError("Query and candidate pair tensors must align")
        product = queries * candidates
        difference = torch.abs(queries - candidates)
        cosine = product.sum(dim=-1, keepdim=True)
        slot_embedding = self.slot_embeddings[slot].expand(
            len(queries),
            -1,
        )
        features = torch.cat(
            (product, difference, cosine, slot_embedding),
            dim=-1,
        )
        if self.per_slot_heads:
            residual = self.slot_heads[slot](self.residual(features)).squeeze(-1)
            return self.base_scales[slot] * cosine.squeeze(-1) + residual
        return self.base_scale * cosine.squeeze(-1) + self.residual(features).squeeze(-1)


@torch.inference_mode()
def rerank_crops(
    reranker: PairReranker,
    metadata: list[dict[str, Any]],
    candidate_embeddings: np.ndarray,
    records: dict[str, list[dict[str, Any]]],
    query_embeddings: dict[str, np.ndarray],
    retrieval_image_k: int,
    candidate_k: int,
    device: torch.device,
    *,
    candidate_view_aggregate: str = "max",
) -> dict[str, list[dict[str, Any]]]:
    reranker.eval()
    slot_indices = {
        slot: np.asarray(
            [
                index
                for index, row in enumerate(metadata)
                if row.get("slot") == slot
            ],
            dtype=np.int64,
        )
        for slot in SLOTS
    }
    slot_tensors = {
        slot: torch.tensor(
            np.asarray(candidate_embeddings[indices]),
            device=device,
        )
        for slot, indices in slot_indices.items()
    }
    results = {}
    for rows in records.values():
        for record in rows:
            path = str(record["path"])
            if path in results or path not in query_embeddings:
                continue
            slot = record["slot"]
            query = torch.tensor(query_embeddings[path], device=device)
            base = slot_tensors[slot] @ query
            local_order = torch.topk(
                base,
                k=min(retrieval_image_k, len(base)),
            ).indices
            candidate_tensor = slot_tensors[slot][local_order]
            scores = reranker(
                query.expand(len(candidate_tensor), -1),
                candidate_tensor,
                slot,
            ).cpu().tolist()
            best_by_piece_and_view: dict[int, dict[str, dict[str, Any]]] = {}
            for local_index, score in zip(
                local_order.cpu().tolist(),
                scores,
                strict=True,
            ):
                global_index = int(slot_indices[slot][local_index])
                row = metadata[global_index]
                for piece in row["pieces"]:
                    piece_id = int(piece["piece_id"])
                    candidate_view = str(row.get("candidate_crop") or "default")
                    views = best_by_piece_and_view.setdefault(piece_id, {})
                    current = views.get(candidate_view)
                    if current is not None and current["score"] >= score:
                        continue
                    views[candidate_view] = {
                        "rank": 0,
                        "score": float(score),
                        **piece,
                        "image_row_id": row["image_row_id"],
                        "local_path": row["local_path"],
                        "source_image_url": row["source_image_url"],
                        "model_gender": row.get("model_gender"),
                        "race": row.get("race"),
                        "dye_state": row.get("dye_state"),
                        "candidate_crop": row.get("candidate_crop"),
                    }
            best_by_piece = [
                aggregate_candidate_views(views, candidate_view_aggregate)
                for views in best_by_piece_and_view.values()
            ]
            ranked = sorted(
                best_by_piece,
                key=lambda item: item["score"],
                reverse=True,
            )[:candidate_k]
            for rank, row in enumerate(ranked, start=1):
                row["rank"] = rank
            results[path] = ranked
    return results


def aggregate_candidate_views(
    views: dict[str, dict[str, Any]],
    strategy: str,
) -> dict[str, Any]:
    if not views:
        raise ValueError("candidate views must not be empty")
    ranked = sorted(
        views.values(),
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    output = dict(ranked[0])
    output["best_view_score"] = float(ranked[0]["score"])
    output["candidate_view_count"] = len(ranked)
    if strategy == "max":
        return output
    if strategy == "top2_mean":
        selected = ranked[:2]
        output["score"] = sum(float(row["score"]) for row in selected) / len(
            selected
        )
        return output
    raise ValueError(f"unsupported candidate view aggregation: {strategy}")
