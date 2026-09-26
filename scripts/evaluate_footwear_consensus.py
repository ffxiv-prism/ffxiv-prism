"""Replay frozen folds before admitting the whole-foot consensus gate.

No fitting or threshold search is performed. Labels are read only to score
completed rankings; the support encoder is the frozen zero-shot SigLIP model.
"""

# Windows requires NumPy's BLAS runtime before importing CUDA Torch.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

np.linalg.inv(np.eye(2, dtype=np.float32))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffxiv_prism.cli.match import _load_appearance_equivalence, _classify_style_paths
from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.equipment_slots import (
    filter_candidates_for_slot,
    load_official_slots,
)
from ffxiv_prism.matching.footwear_consensus import (
    apply_footwear_consensus,
    footwear_consensus_records,
)
from ffxiv_prism.matching.inference import (
    collapse_appearance_equivalent_candidates,
    fuse_branch_search_results,
    fuse_crop_results,
)
from scripts.evaluate_glamour_embedding_model import build_encoder, search_crops
from scripts.evaluate_rank_ensemble_retrieval import load_branch_search_cache
from scripts.evaluate_retrieval_funnel import build_fold_context
from scripts.train_glamour_query_encoder import load_slot_eligibility
from ffxiv_prism.matching.garment_morphology import (
    FEET_STYLE_PROMPTS,
    StylePrediction,
    style_policy,
)


def exclude_height_reranking(cases, examples, regions_dir, output_dir, device):
    """Avoid composing two independently evaluated rerankers.

    Only otherwise changed cases need this check; the rest have zero delta
    regardless of the height classifier. Identity labels are never used.
    """
    import torch
    from PIL import Image
    from ffxiv_prism.vision.outfit_regions import Keypoint, build_outfit_regions

    by_id = {str(e["example_id"]): e for e in examples}
    changed = [c for c in cases if c["diagnostic"]["ranking_changed"]]
    paths = []
    for case in changed:
        source = ROOT / by_id[case["example_id"]]["local_path"]
        stem = f"{by_id[case['example_id']]['glamour_id']}-{source.stem}"
        manifest_path = regions_dir / stem / f"{stem}-regions.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        with Image.open(source) as image:
            points = [
                Keypoint(r["x"], r["y"], r["score"]) for r in manifest["keypoints"]
            ]
            region = next(
                r
                for r in build_outfit_regions(points, image.size)
                if r.name == "feet_height_context"
            )
            path = (
                output_dir
                / "height-crops"
                / f"{case['example_id'].replace(':', '-')}.jpg"
            )
            path.parent.mkdir(exist_ok=True)
            image.crop(region.box).convert("RGB").save(path, quality=92)
            paths.append(path)
    if not paths:
        return
    probabilities = _classify_style_paths(
        "google/siglip2-base-patch16-224",
        FEET_STYLE_PROMPTS,
        paths,
        device=torch.device(device),
        batch_size=16,
    )
    for case, values in zip(changed, probabilities, strict=True):
        prediction = StylePrediction.from_probabilities("FEET", values)
        case["height_probabilities"] = dict(
            zip(prediction.classes, prediction.probabilities, strict=True)
        )
        if style_policy(prediction, minimum_confidence=0.7).active:
            case["revised_rank"] = case["baseline_rank"]
            case["diagnostic"] = {
                **case["diagnostic"],
                "ranking_changed": False,
                "reason": "height_style_reranker_active",
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "data/outputs/current-best-refresh-20260829",
    )
    parser.add_argument(
        "--regions-dir",
        type=Path,
        default=ROOT / "data/debug/outfit-regions/slot-quality-20260801",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reuse-ranking",
        type=Path,
        help="Reuse the label-free whole-foot ranking replay before applying the height guard.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = read_jsonl(
        ROOT / "data/processed/glamour_examples_huiji_models_closed.jsonl"
    )
    eligibility = load_slot_eligibility(
        ROOT / "data/processed/slot_quality_eligibility_20260801.jsonl"
    )
    quality = {
        str(e["example_id"]): "FEET" in eligibility.get(str(e["example_id"]), set())
        for e in examples
    }
    eq = _load_appearance_equivalence()
    slots = load_official_slots()
    encoder = None
    cases = (
        json.loads(args.reuse_ranking.read_text("utf-8"))["cases"]
        if args.reuse_ranking
        else []
    )
    for fold in () if args.reuse_ranking else (1, 2, 3):
        baseline = json.loads(
            (args.baseline_dir / f"final-feet-fold{fold:02}.json").read_text("utf-8")
        )
        context = build_fold_context(
            fold=fold,
            fold_manifest=ROOT / baseline["fold_manifest"],
            all_examples=examples,
            quality_by_id=quality,
            regions_dir=args.regions_dir,
            slot="FEET",
        )
        branches = []
        for branch in baseline["branches"]:
            metadata = read_jsonl(ROOT / branch["index_dir"] / "metadata.jsonl")
            branches.append(
                load_branch_search_cache(
                    ROOT / branch["search_cache"],
                    metadata,
                    candidate_k=1000,
                    index_dir=Path(branch["index_dir"]),
                    query_cache=Path(branch["query_cache"]),
                )
            )
        merged = fuse_branch_search_results(branches, top_k=1000, rrf_k=1)
        # Read all region records: the normal retriever deliberately excludes
        # diagnostic crops, so build_fold_context alone does not contain them.
        support_records = {}
        for example in context["examples"]:
            eid = str(example["example_id"])
            primary = context["records"][eid]
            if not primary:
                continue
            directory = Path(primary[0]["path"]).parent.parent
            manifest_path = next(directory.glob("*-regions.json"))
            manifest = json.loads(manifest_path.read_text("utf-8"))
            all_records = [
                dict(r, path=ROOT / p, example_id=eid)
                for r, p in zip(
                    manifest["regions"], manifest["crop_paths"], strict=True
                )
            ]
            selected = footwear_consensus_records(all_records)
            if selected:
                support_records[eid] = selected
        support_path = args.output_dir / f"support-fold{fold:02}.json"
        paths = [r["path"] for rows in support_records.values() for r in rows]
        if support_path.is_file():
            support = json.loads(support_path.read_text("utf-8"))
            if set(support) != set(map(str, paths)):
                raise ValueError("Support cache does not match the frozen crop paths")
        else:
            if encoder is None:
                encoder = build_encoder(
                    "google/siglip2-base-patch16-224", "auto", args.device, "color"
                )
            vectors = dict(
                zip(map(str, paths), encoder.encode_paths(paths, 32), strict=True)
            )
            index_dir = ROOT / baseline["branches"][1]["index_dir"]
            metadata = read_jsonl(index_dir / "metadata.jsonl")
            embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")
            support = search_crops(
                metadata, embeddings, support_records, vectors, 1, device=args.device
            )
            support_path.write_text(json.dumps(support, ensure_ascii=False), "utf-8")
        for example in context["examples"]:
            eid = str(example["example_id"])
            original = fuse_crop_results(
                context["records"][eid], merged, top_k=1000, fusion="rrf", rrf_k=1
            )
            original = filter_candidates_for_slot(original, "FEET", slots)
            original = collapse_appearance_equivalent_candidates(original, eq)
            rows = support_records.get(eid, [])
            revised, diagnostic = apply_footwear_consensus(
                original, [support[str(r["path"])] for r in rows], eq
            )
            # Scoring begins only after the label-free decision.
            truth = set(example["core_slot_labels"]["FEET"]["piece_ids"])
            truth |= {v for p in list(truth) for v in eq.get(p, ())}

            def rank(ranking):
                return next(
                    (i for i, r in enumerate(ranking, 1) if r["piece_id"] in truth),
                    None,
                )

            cases.append(
                {
                    "example_id": eid,
                    "fold": fold,
                    "gender": example.get("model_gender"),
                    "race": example.get("model_race"),
                    "baseline_rank": rank(original),
                    "revised_rank": rank(revised),
                    "diagnostic": diagnostic,
                }
            )
        print(
            f"fold={fold} cases={len(context['examples'])} usable_support={len(support_records)}",
            flush=True,
        )
    exclude_height_reranking(
        cases, examples, args.regions_dir, args.output_dir, args.device
    )

    def metrics(rows):
        return {
            "count": len(rows),
            "changed": sum(r["diagnostic"]["ranking_changed"] for r in rows),
            "baseline_hits": {
                k: sum(
                    r["baseline_rank"] is not None and r["baseline_rank"] <= k
                    for r in rows
                )
                for k in (1, 5, 10, 20)
            },
            "revised_hits": {
                k: sum(
                    r["revised_rank"] is not None and r["revised_rank"] <= k
                    for r in rows
                )
                for k in (1, 5, 10, 20)
            },
            "top1_gains": sum(
                r["baseline_rank"] != 1 and r["revised_rank"] == 1 for r in rows
            ),
            "top1_losses": sum(
                r["baseline_rank"] == 1 and r["revised_rank"] != 1 for r in rows
            ),
        }

    groups = defaultdict(list)
    for case in cases:
        for key in ("fold", "gender", "race"):
            groups[f"{key}:{case[key]}"].append(case)
    payload = {
        "policy": {
            "max_base_rank": 5,
            "minimum_pose_confidence": 0.8,
            "required_views": 3,
        },
        "scope": "Frozen appearance-disjoint folds; same-model deduplicated identity baseline. The gate is disabled whenever height-style reranking is active; unchanged cases retain baseline identity ranks for accounting, so only the deltas describe the production change.",
        "summary": metrics(cases),
        "slices": {key: metrics(rows) for key, rows in groups.items()},
        "cases": cases,
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), "utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k != "cases"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
