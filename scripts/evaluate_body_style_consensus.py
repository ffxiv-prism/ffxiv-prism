"""Evaluate the fixed BODY context/torso agreement rule on frozen folds.

Only changed cases need identity retrieval. Unchanged cases have exactly zero
metric delta. Cached fold-specific query adapters are never trained here.
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
import torch
from ffxiv_prism.cli.match import _classify_style_paths, _load_appearance_equivalence
from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.garment_morphology import (
    BODY_STYLE_PROMPTS,
    StylePrediction,
    apply_style_crop_multipliers,
    body_style_consensus_policy,
    rerank_candidates_with_style,
    style_policy,
)
from ffxiv_prism.matching.inference import (
    collapse_appearance_equivalent_candidates,
    crop_weight,
    fuse_crops_then_branch_results,
    select_slot_candidate_index_rows,
)
from ffxiv_prism.matching.equipment_slots import (
    filter_candidates_for_slot,
    load_official_slots,
)
from scripts.evaluate_glamour_embedding_model import search_crops
from scripts.evaluate_cached_query_retrieval import load_query_embeddings
from scripts.evaluate_retrieval_funnel import build_fold_context
from scripts.train_glamour_query_encoder import load_slot_eligibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = "google/siglip2-base-patch16-224"
    base = ROOT / "data/outputs/current-best-refresh-20260829"
    examples = read_jsonl(
        ROOT / "data/processed/glamour_examples_huiji_models_closed.jsonl"
    )
    eligibility = load_slot_eligibility(
        ROOT / "data/processed/slot_quality_eligibility_20260801.jsonl"
    )
    quality = {
        str(e["example_id"]): "BODY" in eligibility.get(str(e["example_id"]), set())
        for e in examples
    }
    eq, slots = _load_appearance_equivalence(), load_official_slots()
    cache_path = args.output_dir / "style-profiles.json"
    profiles = json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}

    def classify(paths):
        missing = list(dict.fromkeys(str(p) for p in paths if str(p) not in profiles))
        if missing:
            values = _classify_style_paths(
                model,
                BODY_STYLE_PROMPTS,
                [Path(p) for p in missing],
                device=device,
                batch_size=32,
            )
            profiles.update(
                {p: v.tolist() for p, v in zip(missing, values, strict=True)}
            )
            cache_path.write_text(json.dumps(profiles), "utf-8")

    cases, total, old_active_count = [], 0, 0
    for fold in (1, 2, 3):
        context = build_fold_context(
            fold=fold,
            fold_manifest=ROOT
            / f"data/interim/appearance-disjoint-cv-v55-slot-recovery-body/fold-{fold:02}.json",
            all_examples=examples,
            quality_by_id=quality,
            regions_dir=ROOT / "data/debug/outfit-regions/slot-quality-20260801",
            slot="BODY",
        )
        paths_by_id = {}
        for eid, records in context["records"].items():
            if not records:
                continue
            folder = Path(records[0]["path"]).parent.parent
            manifest = json.loads(
                next(folder.glob("*-regions.json")).read_text("utf-8")
            )
            rows = {
                r["name"]: dict(r, path=ROOT / p)
                for r, p in zip(
                    manifest["regions"], manifest["crop_paths"], strict=True
                )
            }
            # Earlier frozen geometry lacks body_upper_core. Recreate only
            # this diagnostic from saved keypoints, never rerun pose or labels.
            if "body_upper_core" not in rows:
                from PIL import Image
                from ffxiv_prism.vision.outfit_regions import (
                    Keypoint,
                    build_outfit_regions,
                )

                with Image.open(ROOT / manifest["source_image"]) as im:
                    kp = [
                        Keypoint(r["x"], r["y"], r["score"])
                        for r in manifest["keypoints"]
                    ]
                    core = next(
                        r
                        for r in build_outfit_regions(kp, im.size)
                        if r.name == "body_upper_core"
                    )
                    path = (
                        args.output_dir / "core-crops" / f"{eid.replace(':', '-')}.jpg"
                    )
                    path.parent.mkdir(exist_ok=True)
                    im.crop(core.box).convert("RGB").save(path, quality=92)
                    rows["body_upper_core"] = {
                        "path": path,
                        "visibility": core.visibility,
                        "confidence": core.confidence,
                    }
            paths_by_id[eid] = [rows.get("body_context"), rows.get("body_upper_core")]
        classify(
            [r["path"] for rows in paths_by_id.values() for r in rows if r is not None]
        )
        affected, predictions = {}, {}
        for eid, rows in paths_by_id.items():
            context_row, core = rows
            prediction = StylePrediction.from_probabilities(
                "BODY", profiles[str(context_row["path"])]
            )
            corroboration = (
                StylePrediction.from_probabilities("BODY", profiles[str(core["path"])])
                if core
                and core["visibility"] == "visible"
                and core["confidence"] >= 0.75
                else None
            )
            old = style_policy(prediction, minimum_confidence=0.75)
            new = body_style_consensus_policy(
                prediction, corroboration, minimum_confidence=0.75
            )
            old_active_count += int(old.active)
            if old.active and not new.active:
                affected[eid] = [
                    r for r in context["records"][eid] if crop_weight(r) > 0
                ]
                predictions[eid] = prediction
        total += len(context["examples"])
        print(
            f"fold={fold} total={len(context['examples'])} affected={len(affected)}",
            flush=True,
        )
        old_records = {
            eid: apply_style_crop_multipliers(
                rows, style_policy(predictions[eid], minimum_confidence=0.75)
            )
            for eid, rows in affected.items()
        }
        old_searches, new_searches = [], []
        for index_name, cache_name in (
            ("facebook-dinov2-base-slot-aligned-v3-body-multiscale", "dinov2-adapted"),
            ("facebook-dinov3-vitb16-body-multiscale-v3", "dinov3-zero"),
        ):
            index = ROOT / f"data/indexes/{index_name}-20260829-024239"
            metadata = read_jsonl(index / "metadata.jsonl")
            vectors = np.load(index / "embeddings.npy", mmap_mode="r")
            metadata, vectors = select_slot_candidate_index_rows(
                metadata, vectors, "BODY"
            )
            queries = load_query_embeddings(
                base / f"cache/body-fold{fold:02}-{cache_name}-query.npz"
            )
            new_searches.append(
                search_crops(
                    metadata, vectors, affected, queries, 1000, device=args.device
                )
            )
            old_search = {}
            for eid, rows in old_records.items():
                route = style_policy(
                    predictions[eid], minimum_confidence=0.75
                ).candidate_crop_routes
                old_search.update(
                    search_crops(
                        metadata,
                        vectors,
                        {eid: rows},
                        queries,
                        1000,
                        device=args.device,
                        candidate_crop_routes=route,
                    )
                )
            old_searches.append(old_search)
        pre_old, pre_new = {}, {}
        for eid in affected:
            for rows, searches, dest in (
                (old_records[eid], old_searches, pre_old),
                (affected[eid], new_searches, pre_new),
            ):
                dest[eid] = fuse_crops_then_branch_results(
                    rows,
                    searches,
                    top_k=1000,
                    candidate_k=1000,
                    crop_fusion="max",
                    crop_rrf_k=60,
                    branch_rrf_k=1,
                    final_pool_branches=1,
                    branch_weights=(1.0, 1.0),
                )
        candidate_paths = [
            ROOT / c["local_preview"]
            for ranking in pre_old.values()
            for c in ranking[:100]
        ]
        classify([p for p in candidate_paths if p.is_file()])
        examples_by_id = {str(e["example_id"]): e for e in context["examples"]}
        for eid in affected:
            ranking = pre_old[eid]
            p = {
                c["piece_id"]: profiles[str(ROOT / c["local_preview"])]
                for c in ranking[:100]
                if str(ROOT / c["local_preview"]) in profiles
            }
            old = rerank_candidates_with_style(
                ranking,
                predictions[eid],
                p,
                rerank_k=100,
                blend=0.7,
                minimum_confidence=0.75,
            )
            old = collapse_appearance_equivalent_candidates(
                filter_candidates_for_slot(old, "BODY", slots), eq
            )
            new = collapse_appearance_equivalent_candidates(
                filter_candidates_for_slot(pre_new[eid], "BODY", slots), eq
            )
            example = examples_by_id[eid]
            truth = set(example["core_slot_labels"]["BODY"]["piece_ids"])
            truth |= {v for p in list(truth) for v in eq.get(p, ())}

            def rank(rs):
                return next(
                    (i for i, r in enumerate(rs, 1) if r["piece_id"] in truth), None
                )

            cases.append(
                {
                    "example_id": eid,
                    "fold": fold,
                    "gender": example.get("model_gender"),
                    "race": example.get("model_race"),
                    "old_rank": rank(old),
                    "new_rank": rank(new),
                }
            )
        (args.output_dir / "changed-cases.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2), "utf-8"
        )

    def metrics(rows):
        return {
            "changed_cases": len(rows),
            "old_hits": {
                k: sum(r["old_rank"] is not None and r["old_rank"] <= k for r in rows)
                for k in (1, 10, 20)
            },
            "new_hits": {
                k: sum(r["new_rank"] is not None and r["new_rank"] <= k for r in rows)
                for k in (1, 10, 20)
            },
        }

    groups = defaultdict(list)
    for case in cases:
        for key in ("fold", "gender", "race"):
            groups[f"{key}:{case[key]}"].append(case)
    result = {
        "eligible": total,
        "old_style_active": old_active_count,
        "minimum_confidence": 0.75,
        "summary": metrics(cases),
        "slices": {k: metrics(v) for k, v in groups.items()},
        "cases": cases,
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), "utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "cases"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
