from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import CACHE_DIR, PROCESSED_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from ffxiv_prism.vision.evaluation_split import (
    development_split_map,
    load_split_assignments,
)
from scripts.evaluate_glamour_matching import (
    SLOTS,
    available_appearance_keys_by_slot,
    available_pieces_by_slot,
    collect_crop_records,
)
from scripts.train_glamour_adapter import (
    build_candidate_state,
    build_samples,
    select_on_validation_and_evaluate_test,
)
from scripts.train_glamour_query_encoder import (
    encode_records,
    group_training_samples,
    load_training_checkpoint,
    prune_training_samples,
    set_seed,
)

DEFAULT_INDEX = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_REGIONS = Path("data/debug/outfit-regions-v3-624")
DEFAULT_QUERY_CHECKPOINT = Path(
    "data/models/glamour_query_encoder_v5_160g_grouped.pt"
)
DEFAULT_SPLIT = PROCESSED_DIR / "glamour_split_v2.json"
DEFAULT_CHECKPOINT = Path("data/models/glamour_pair_reranker.pt")
DEFAULT_OUTPUT = PROCESSED_DIR / "glamour_pair_reranker_eval.json"
DEFAULT_QUERY_CACHE = PROCESSED_DIR / "glamour_query_embeddings_v5_624.npz"


class PairReranker(nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden_dimension: int = 256,
        slot_dimension: int = 16,
        per_slot_heads: bool = False,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.per_slot_heads = per_slot_heads
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
                nn.Dropout(0.1),
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
                nn.Dropout(0.1),
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a nonlinear query-candidate reranker on mined hard negatives."
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROCESSED_DIR / "glamour_examples.jsonl",
    )
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--query-checkpoint",
        type=Path,
        default=DEFAULT_QUERY_CHECKPOINT,
    )
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hard-image-k", type=int, default=128)
    parser.add_argument("--retrieval-image-k", type=int, default=400)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-crops-per-example-slot", type=int, default=2)
    parser.add_argument(
        "--per-slot-heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a shared feature trunk with independent output heads per equipment slot.",
    )
    parser.add_argument(
        "--group-by-glamour",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Aggregate all training views sharing one glamour label.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_torch_device(args.device)
    args.device = str(device)
    metadata = read_jsonl(args.index_dir / "metadata.jsonl")
    candidate_embeddings = np.load(
        args.index_dir / "embeddings.npy",
        mmap_mode="r",
    )
    examples = read_jsonl(args.examples)
    records = collect_crop_records(examples, args.regions_dir)
    assignments = load_split_assignments(args.split_manifest)
    split_map = development_split_map(
        {str(example["glamour_id"]) for example in examples},
        assignments,
    )
    development_examples = [
        example
        for example in examples
        if str(example["glamour_id"]) in split_map
    ]
    split_examples = {
        split: [
            example
            for example in development_examples
            if split_map[str(example["glamour_id"])] == split
        ]
        for split in ("train", "validation", "test")
    }
    split_records = {
        split: {
            example["example_id"]: records.get(example["example_id"], [])
            for example in split_examples[split]
        }
        for split in split_examples
    }
    development_records = {
        **split_records["train"],
        **split_records["validation"],
        **split_records["test"],
    }

    query_embeddings = load_or_build_query_embeddings(
        args,
        development_records,
        device,
    )
    candidate_state = build_candidate_state(
        metadata,
        candidate_embeddings,
        device,
    )
    samples = build_samples(
        split_examples["train"],
        split_records["train"],
        query_embeddings,
        candidate_state["piece_to_local"],
    )
    samples = prune_training_samples(
        samples,
        split_records["train"],
        args.max_crops_per_example_slot,
    )
    training_group_ids = (
        {
            str(example["example_id"]): str(example["glamour_id"])
            for example in split_examples["train"]
        }
        if args.group_by_glamour
        else None
    )
    grouped_samples = group_training_samples(
        samples,
        split_records["train"],
        group_ids=training_group_ids,
    )

    reranker = PairReranker(
        int(candidate_embeddings.shape[1]),
        per_slot_heads=args.per_slot_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        reranker.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    available_piece_ids = available_pieces_by_slot(metadata)
    available_appearance_keys = available_appearance_keys_by_slot(metadata)
    best_state = None
    best_epoch = 0
    best_score = (-1.0, -1.0, -1.0)
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            reranker,
            optimizer,
            grouped_samples,
            query_embeddings,
            candidate_state["tensors"],
            args.hard_image_k,
            args.batch_size,
            device,
        )
        validation_search = rerank_crops(
            reranker,
            metadata,
            candidate_embeddings,
            split_records["validation"],
            query_embeddings,
            args.retrieval_image_k,
            args.candidate_k,
            device,
        )
        selection = select_on_validation_and_evaluate_test(
            {
                "validation": split_examples["validation"],
                "test": split_examples["validation"],
            },
            {
                "validation": split_records["validation"],
                "test": split_records["validation"],
            },
            validation_search,
            available_piece_ids,
            available_appearance_keys,
            args.top_k,
        )
        validation = selection["validation"]
        score = (
            validation["appearance_recall_at_1"],
            validation["appearance_mrr"],
            validation["appearance_recall_at_k"],
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(reranker.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "validation": validation,
            }
        )
        print(
            f"epoch={epoch} loss={loss:.4f} "
            f"val@1={validation['appearance_recall_at_1']:.3f} "
            f"val@{args.top_k}={validation['appearance_recall_at_k']:.3f} "
            f"mrr={validation['appearance_mrr']:.3f} best_epoch={best_epoch}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Reranker training produced no checkpoint")
    reranker.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "state_dict": best_state,
            "dimension": reranker.dimension,
            "per_slot_heads": reranker.per_slot_heads,
            "group_by_glamour": args.group_by_glamour,
            "query_checkpoint": portable_path(args.query_checkpoint),
            "best_epoch": best_epoch,
            "seed": args.seed,
        },
        args.checkpoint,
    )

    relevant_records = {
        **split_records["validation"],
        **split_records["test"],
    }
    search = rerank_crops(
        reranker,
        metadata,
        candidate_embeddings,
        relevant_records,
        query_embeddings,
        args.retrieval_image_k,
        args.candidate_k,
        device,
    )
    evaluation = select_on_validation_and_evaluate_test(
        split_examples,
        split_records,
        search,
        available_piece_ids,
        available_appearance_keys,
        args.top_k,
    )
    payload = {
        "model": {
            "type": "nonlinear_pair_reranker",
            "checkpoint": portable_path(args.checkpoint),
            "query_checkpoint": portable_path(args.query_checkpoint),
            "parameter_count": sum(
                parameter.numel() for parameter in reranker.parameters()
            ),
            "best_epoch": best_epoch,
            "per_slot_heads": reranker.per_slot_heads,
        },
        "training": {
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "hard_image_k": args.hard_image_k,
            "group_by_glamour": args.group_by_glamour,
            "history": history,
        },
        "evaluation": evaluation,
        "split_manifest": portable_path(args.split_manifest),
        "regions_dir": portable_path(args.regions_dir),
        "index_dir": portable_path(args.index_dir),
        "retrieval_image_k": args.retrieval_image_k,
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
    }
    write_json(args.output, payload)
    test = evaluation["test"]
    print(
        f"test@1={test['appearance_recall_at_1']:.3f} "
        f"test@{args.top_k}={test['appearance_recall_at_k']:.3f} "
        f"mrr={test['appearance_mrr']:.3f} "
        f"checkpoint={args.checkpoint} output={args.output}"
    )
    return 0


def load_or_build_query_embeddings(
    args: argparse.Namespace,
    records: dict[str, list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    paths = sorted(
        {
            str(record["path"])
            for rows in records.values()
            for record in rows
        }
    )
    if args.query_cache.is_file():
        cached = np.load(args.query_cache, allow_pickle=False)
        if (
            [str(value) for value in cached["paths"].tolist()] == paths
            and str(cached["checkpoint"].tolist()) == str(args.query_checkpoint)
        ):
            return {
                path: embedding
                for path, embedding in zip(
                    paths,
                    cached["embeddings"],
                    strict=True,
                )
            }

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device)
    load_training_checkpoint(
        args.query_checkpoint,
        model,
        None,
        device,
    )
    embeddings = encode_records(
        model,
        None,
        records,
        batch_size=32,
        device=device,
    )
    args.query_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.query_cache,
        paths=np.asarray(paths),
        embeddings=np.stack([embeddings[path] for path in paths]),
        checkpoint=np.asarray(str(args.query_checkpoint)),
    )
    return embeddings


def train_epoch(
    reranker: PairReranker,
    optimizer: torch.optim.Optimizer,
    samples: dict[str, list[Any]],
    query_embeddings: dict[str, np.ndarray],
    candidates: dict[str, torch.Tensor],
    hard_image_k: int,
    batch_size: int,
    device: torch.device,
) -> float:
    reranker.train()
    losses = []
    for slot in SLOTS:
        order = torch.randperm(len(samples[slot])).tolist()
        for start in range(0, len(order), batch_size):
            batch = [
                samples[slot][index]
                for index in order[start : start + batch_size]
            ]
            sample_losses = []
            for sample in batch:
                queries = torch.tensor(
                    np.stack(
                        [query_embeddings[path] for path in sample.paths]
                    ),
                    device=device,
                )
                base = queries @ candidates[slot].T
                aggregate_base = base.max(dim=0).values
                hard = torch.topk(
                    aggregate_base,
                    k=min(hard_image_k, aggregate_base.numel()),
                ).indices.tolist()
                candidate_indices = list(
                    dict.fromkeys([*hard, *sample.positives])
                )
                candidate_tensor = candidates[slot][candidate_indices]
                crop_scores = []
                for query in queries:
                    crop_scores.append(
                        reranker(
                            query.expand(len(candidate_tensor), -1),
                            candidate_tensor,
                            slot,
                        )
                    )
                scores = torch.logsumexp(
                    torch.stack(crop_scores),
                    dim=0,
                ) - np.log(len(crop_scores))
                positive_set = set(sample.positives)
                positive_positions = [
                    index
                    for index, candidate_index in enumerate(candidate_indices)
                    if candidate_index in positive_set
                ]
                numerator = torch.logsumexp(
                    scores[positive_positions],
                    dim=0,
                )
                denominator = torch.logsumexp(scores, dim=0)
                sample_losses.append(
                    (denominator - numerator) * sample.weight
                )
            loss = torch.stack(sample_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


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
            best_by_piece = {}
            for local_index, score in zip(
                local_order.cpu().tolist(),
                scores,
                strict=True,
            ):
                global_index = int(slot_indices[slot][local_index])
                row = metadata[global_index]
                for piece in row["pieces"]:
                    piece_id = int(piece["piece_id"])
                    current = best_by_piece.get(piece_id)
                    if current is not None and current["score"] >= score:
                        continue
                    best_by_piece[piece_id] = {
                        "rank": 0,
                        "score": float(score),
                        **piece,
                        "image_row_id": row["image_row_id"],
                        "local_path": row["local_path"],
                        "source_image_url": row["source_image_url"],
                        "model_gender": row.get("model_gender"),
                        "race": row.get("race"),
                        "dye_state": row.get("dye_state"),
                    }
            ranked = sorted(
                best_by_piece.values(),
                key=lambda item: item["score"],
                reverse=True,
            )[:candidate_k]
            for rank, row in enumerate(ranked, start=1):
                row["rank"] = rank
            results[path] = ranked
    return results


if __name__ == "__main__":
    raise SystemExit(main())
