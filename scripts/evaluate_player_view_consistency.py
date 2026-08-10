from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.glamour_parser import (  # noqa: E402
    appearance_name_key,
)
from ffxiv_prism.jsonl import read_jsonl, write_json  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402


DEFAULT_PROFILES = {
    "local": ("hands_left", "hands_right"),
    "forearm": ("hands_forearm_left", "hands_forearm_right"),
    "context": ("hands_arm_cover_context",),
    "all": (),
}


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero embedding")
    return np.asarray(vector / norm, dtype=np.float32)


def source_key(example: Mapping[str, Any]) -> str:
    path = Path(str(example["local_path"]))
    return f"{path.parent.name}-{path.stem}"


def crop_source_key(path: str | Path) -> str:
    crop_path = Path(str(path))
    if crop_path.parent.name == "crops":
        return crop_path.parent.parent.name
    return crop_path.parent.name


def crop_suffix(path: str | Path, key: str) -> str:
    stem = Path(str(path)).stem
    prefix = f"{key}-"
    if not stem.startswith(prefix):
        raise ValueError(f"Crop path does not start with source key {key}: {path}")
    return stem[len(prefix) :]


def aggregate_embeddings(
    examples: Sequence[Mapping[str, Any]],
    cache_paths: Sequence[str],
    embeddings: np.ndarray,
    *,
    slot: str,
    allowed_suffixes: set[str] | None,
) -> list[dict[str, Any]]:
    if len(cache_paths) != len(embeddings):
        raise ValueError(
            f"Cache path count {len(cache_paths)} != embeddings {len(embeddings)}"
        )
    examples_by_source = {}
    for example in examples:
        key = source_key(example)
        if key in examples_by_source:
            raise ValueError(f"Duplicate source key: {key}")
        examples_by_source[key] = example

    vectors_by_id: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    examples_by_id: dict[str, Mapping[str, Any]] = {}
    for path, embedding in zip(cache_paths, embeddings, strict=True):
        key = crop_source_key(path)
        example = examples_by_source.get(key)
        if example is None:
            continue
        suffix = crop_suffix(path, key)
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue
        label = ((example.get("core_slot_labels") or {}).get(slot) or {})
        if not label.get("piece_ids"):
            continue
        example_id = str(example["example_id"])
        vectors_by_id[example_id].append(normalize(embedding))
        examples_by_id[example_id] = example

    rows = []
    for example_id, vectors in sorted(vectors_by_id.items()):
        example = examples_by_id[example_id]
        label = ((example.get("core_slot_labels") or {}).get(slot) or {})
        expected_name = str(label.get("item_name") or "")
        rows.append(
            {
                "example_id": example_id,
                "glamour_id": str(example["glamour_id"]),
                "expected_name": expected_name,
                "appearance_key": appearance_name_key(expected_name),
                "crop_count": len(vectors),
                "embedding": normalize(np.mean(vectors, axis=0)),
            }
        )
    return rows


def summarize_ranks(
    ranks: Sequence[int],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    count = len(ranks)
    hits = {
        str(cutoff): sum(rank <= cutoff for rank in ranks)
        for cutoff in cutoffs
    }
    return {
        "query_count": count,
        "hits": hits,
        "recall": {
            key: value / count if count else 0.0
            for key, value in hits.items()
        },
        "mrr": (
            sum(1.0 / rank for rank in ranks) / count
            if count
            else 0.0
        ),
        "median_rank": float(np.median(ranks)) if ranks else None,
    }


def evaluate_consistency(
    rows: Sequence[Mapping[str, Any]],
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    if not rows:
        return {
            "image_count": 0,
            "glamour_count": 0,
            "same_glamour": {**summarize_ranks([], cutoffs), "cases": []},
            "same_appearance_other_glamour": {
                **summarize_ranks([], cutoffs),
                "cases": [],
            },
        }
    matrix = np.stack(
        [normalize(np.asarray(row["embedding"])) for row in rows]
    )
    # Keep this diagnostic compatible with the model test suite on Windows,
    # where NumPy BLAS and PyTorch can otherwise initialize different OpenMP
    # runtimes in the same process.
    similarities = np.einsum(
        "id,jd->ij",
        matrix,
        matrix,
        optimize=False,
    )
    np.fill_diagonal(similarities, -np.inf)
    same_glamour_cases = []
    same_appearance_cases = []
    for index, row in enumerate(rows):
        order = np.argsort(-similarities[index], kind="stable")
        same_glamour = [
            candidate
            for candidate, other in enumerate(rows)
            if candidate != index
            and str(other["glamour_id"]) == str(row["glamour_id"])
        ]
        same_appearance = [
            candidate
            for candidate, other in enumerate(rows)
            if candidate != index
            and str(other["glamour_id"]) != str(row["glamour_id"])
            and row.get("appearance_key")
            and other.get("appearance_key") == row.get("appearance_key")
        ]
        for positives, output in (
            (same_glamour, same_glamour_cases),
            (same_appearance, same_appearance_cases),
        ):
            if not positives:
                continue
            rank = min(
                int(np.flatnonzero(order == positive)[0]) + 1
                for positive in positives
            )
            nearest = rows[int(order[0])]
            output.append(
                {
                    "example_id": str(row["example_id"]),
                    "glamour_id": str(row["glamour_id"]),
                    "expected_name": row.get("expected_name"),
                    "rank": rank,
                    "best_positive_similarity": max(
                        float(similarities[index, positive])
                        for positive in positives
                    ),
                    "nearest_example_id": str(nearest["example_id"]),
                    "nearest_glamour_id": str(nearest["glamour_id"]),
                    "nearest_similarity": float(
                        similarities[index, int(order[0])]
                    ),
                }
            )

    return {
        "image_count": len(rows),
        "glamour_count": len({str(row["glamour_id"]) for row in rows}),
        "same_glamour": {
            **summarize_ranks(
                [int(case["rank"]) for case in same_glamour_cases],
                cutoffs,
            ),
            "cases": same_glamour_cases,
        },
        "same_appearance_other_glamour": {
            **summarize_ranks(
                [int(case["rank"]) for case in same_appearance_cases],
                cutoffs,
            ),
            "cases": same_appearance_cases,
        },
    }


def parse_profile(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("profile must be NAME=SUFFIX[,SUFFIX]")
    name, suffix_text = value.split("=", 1)
    name = name.strip()
    suffixes = tuple(
        suffix.strip()
        for suffix in suffix_text.split(",")
        if suffix.strip()
    )
    if not name:
        raise argparse.ArgumentTypeError("profile name cannot be empty")
    return name, suffixes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether frozen player-query embeddings are consistent "
            "across views before comparing them with canonical renders."
        )
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--slot", default="HANDS")
    parser.add_argument(
        "--profile",
        action="append",
        type=parse_profile,
        help=(
            "Crop profile NAME=SUFFIX[,SUFFIX]. Repeat to define multiple "
            "profiles; defaults to HANDS local/forearm/context/all."
        ),
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if any(cutoff < 1 for cutoff in args.cutoffs):
        parser.error("--cutoffs must be positive")
    profiles = dict(args.profile or DEFAULT_PROFILES.items())
    examples = read_jsonl(args.examples)
    with np.load(args.query_cache, allow_pickle=False) as cache:
        cache_paths = [str(value) for value in cache["paths"].tolist()]
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        checkpoint = (
            str(cache["checkpoint"].item())
            if "checkpoint" in cache.files
            else None
        )

    profile_results = {}
    for name, suffixes in profiles.items():
        rows = aggregate_embeddings(
            examples,
            cache_paths,
            embeddings,
            slot=args.slot,
            allowed_suffixes=set(suffixes) if suffixes else None,
        )
        profile_results[name] = evaluate_consistency(
            rows,
            cutoffs=tuple(sorted(set(args.cutoffs))),
        )

    payload = {
        "version": 1,
        "diagnostic": "player_view_embedding_consistency",
        "slot": args.slot,
        "examples": portable_path(args.examples),
        "query_cache": portable_path(args.query_cache),
        "checkpoint": portable_path(checkpoint),
        "profiles": profile_results,
        "interpretation_limit": (
            "Same-glamour positives also share character, outfit and scene. "
            "This diagnostic isolates gross canonical-render domain mismatch "
            "but is not a recognition accuracy estimate."
        ),
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                name: {
                    key: value
                    for key, value in result["same_glamour"].items()
                    if key != "cases"
                }
                for name, result in profile_results.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
