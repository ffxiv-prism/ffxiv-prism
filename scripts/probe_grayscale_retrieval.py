"""Offline, label-free probe of full-view colour/gray DINO retrieval.

This is an experiment, not a production ranking policy. Both sides use the
same frozen base encoder (no colour-trained query adapter on gray inputs).
The query manifest, candidate snapshot and preprocessing are recorded so a
failed experiment is reproducible. No target item IDs or names are inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# Initialize NumPy BLAS before CUDA Torch on Windows.
np.linalg.inv(np.eye(2, dtype=np.float32))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffxiv_prism.cli.match import _load_appearance_equivalence
from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.inference import (
    collapse_appearance_equivalent_candidates,
    current_best_configs,
)
from ffxiv_prism.matching.runtime.encoders import build_encoder, search_crops
from ffxiv_prism.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regions", type=Path)
    parser.add_argument("--slots", nargs="+", choices=("HEAD", "BODY", "HANDS", "LEGS", "FEET"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.regions.read_text("utf-8"))
    configs = current_best_configs()
    equivalence = _load_appearance_equivalence()
    encoder = build_encoder("facebook/dinov2-base", "dinov2", args.device, "color")
    for slot in args.slots:
        index = configs[slot].index_dir
        all_metadata = read_jsonl(index / "metadata.jsonl")
        # Full rows correspond to the unmodified source candidate image.
        metadata = [r for r in all_metadata if r["slot"] == slot and r.get("candidate_crop") == "full"]
        if not metadata:
            raise ValueError(f"{slot} index has no full-image candidate rows")
        valid_metadata, candidate_paths, excluded = [], [], []
        for row in metadata:
            path = resolve_project_path(row["local_path"])
            try:
                with Image.open(path) as source:
                    source.load()
            except (OSError, ValueError) as exc:
                excluded.append({"path":str(path), "error":str(exc), "pieces":row["pieces"]})
                continue
            valid_metadata.append(row)
            candidate_paths.append(path)
        metadata = valid_metadata
        if not metadata:
            raise ValueError(f"No decodable {slot} candidate images")
        records = [dict(r, path=resolve_project_path(p), example_id="query")
                   for r, p in zip(manifest["regions"], manifest["crop_paths"], strict=True)
                   if r["slot"] == slot and r["visibility"] not in {"out_of_frame", "pose_uncertain"}]
        query_paths = [r["path"] for r in records]
        for mode in ("color", "gray"):
            encoder.preprocess = mode
            print(f"probe slot={slot} mode={mode} candidates={len(metadata)}", flush=True)
            # Re-encode colour too: avoids confusing preexisting index drift
            # with a preprocessing effect. The gallery is never modified.
            vectors = encoder.encode_paths(candidate_paths, args.batch_size)
            queries = dict(zip(map(str, query_paths), encoder.encode_paths(query_paths, args.batch_size), strict=True))
            results = search_crops(metadata, vectors, {"query":records}, queries, 10000, device=args.device)
            output = {}
            for record in records:
                rows = collapse_appearance_equivalent_candidates(results[str(record["path"])], equivalence)
                output[record["name"]] = [
                    {"rank": i, "piece_id": r["piece_id"], "name_en": r.get("name_en"),
                     "score": r["score"], "local_path": r.get("local_path")}
                    for i, r in enumerate(rows, 1)
                ]
            payload = {"experimental": True, "regions":str(args.regions), "index":str(index),
                       "model":"facebook/dinov2-base", "preprocess":mode,
                       "candidate_rows":len(metadata), "excluded_images":excluded, "results":output}
            (args.output_dir / f"{slot}-{mode}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
