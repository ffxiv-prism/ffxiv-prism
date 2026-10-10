"""Evaluate a frozen DINOv3 full-render branch on the registered FEET folds.

Equal-weight RRF(k=1), existing crops, no item-specific rules or training.
Candidate images and query vectors are cached, never ground-truth ranks.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from functools import lru_cache
import numpy as np
from PIL import Image

np.linalg.inv(np.eye(2, dtype=np.float32))
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ffxiv_prism.cli.match import _load_appearance_equivalence
from ffxiv_prism.jsonl import read_jsonl, write_jsonl, write_json
from ffxiv_prism.matching.equipment_slots import filter_candidates_for_slot, load_official_slots
from ffxiv_prism.matching.inference import collapse_appearance_equivalent_candidates, fuse_branch_search_results, fuse_crop_results
from ffxiv_prism.matching.runtime.encoders import build_encoder, search_crops
from scripts.evaluate_rank_ensemble_retrieval import load_branch_search_cache, save_branch_search_cache
from scripts.evaluate_retrieval_funnel import build_fold_context
from scripts.train_glamour_query_encoder import load_slot_eligibility

# This offline replay keeps one cwd and one immutable gallery snapshot.
# Avoid repeated Windows filesystem resolution for the same preview paths.
import ffxiv_prism.matching.inference as inference
inference.portable_path = lru_cache(maxsize=32768)(inference.portable_path)

MODEL = "var/models/dinov3-vitb16-pretrain-lvd1689m"


def style_queries(context, output, device, batch_size):
    import torch
    from dataclasses import asdict
    from ffxiv_prism.vision.outfit_regions import Keypoint, build_outfit_regions
    from ffxiv_prism.cli.match import _classify_style_paths
    from ffxiv_prism.matching.garment_morphology import FEET_STYLE_PROMPTS, footwear_style_records
    selected, paths = {}, []
    output.mkdir(parents=True, exist_ok=True)
    for example in context["examples"]:
        eid = str(example["example_id"])
        directory = Path(context["records"][eid][0]["path"]).parent.parent
        manifest = json.loads(next(directory.glob("*-regions.json")).read_text("utf-8"))
        with Image.open(ROOT / example["local_path"]) as source:
            regions = build_outfit_regions([Keypoint(r["x"],r["y"],r["score"]) for r in manifest["keypoints"]], source.size)
            rows = [asdict(r) for r in regions if r.slot == "FEET"]
            chosen = [r for r in rows if r["name"] == "feet_height_context"] + footwear_style_records(rows)
            selected[eid] = []
            for r in chosen:
                path = output / f"{eid.replace(':','-')}-{r['name']}.jpg"
                if not path.is_file():
                    source.crop(r["box"]).convert("RGB").save(path, quality=92)
                paths.append(path)
                selected[eid].append(str(path))
    values = _classify_style_paths("google/siglip2-base-patch16-224", FEET_STYLE_PROMPTS,
                                  paths, device=torch.device(device), batch_size=batch_size)
    by_path = dict(zip(map(str,paths),values,strict=True))
    return {eid:[by_path[p] for p in items] for eid,items in selected.items()}


def full_index(source, output, encoder, batch_size, *, preserve_variant_rows=False):
    rows, rejected = [], []
    for row in read_jsonl(source / "metadata.jsonl"):
        if row["slot"] != "FEET" or row.get("candidate_crop") != "full":
            continue
        try:
            with Image.open(ROOT / row["local_path"]) as image:
                image.load()
        except (OSError, ValueError) as exc:
            rejected.append(dict(path=row["local_path"], error=str(exc)))
            continue
        rows.append(row)
    full_rows = rows
    if preserve_variant_rows:
        allowed = {r["source_image_url"] for r in full_rows}
        rows = [r for r in read_jsonl(source / "metadata.jsonl")
                if r["slot"] == "FEET" and r["source_image_url"] in allowed]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "embeddings.npy").is_file():
        if read_jsonl(output / "metadata.jsonl") != rows:
            raise ValueError("Cached gallery metadata changed")
        vectors = np.load(output / "embeddings.npy")
    else:
        full_vectors = encoder.encode_paths([ROOT / r["local_path"] for r in full_rows], batch_size)
        positions = {r["source_image_url"]: i for i,r in enumerate(full_rows)}
        vectors = full_vectors[[positions[r["source_image_url"]] for r in rows]]
        np.save(output / "embeddings.npy", vectors, allow_pickle=False)
        write_jsonl(output / "metadata.jsonl", rows)
        write_json(output / "summary.json", dict(model_id=MODEL, pooling="cls", preprocess="color",
            source_index=source.as_posix(), source_image_count=len(full_rows), aligned_candidate_count=len(rows),
            metadata_row_count=len(rows), embedding_shape=list(vectors.shape), completed=True,
            slot_policy={"FEET": sorted({r["candidate_crop"] for r in rows})},
            full_image_encoding=True, excluded_images=rejected))
    assert len(rows) == len(vectors) and np.isfinite(vectors).all()
    return rows, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-source", type=Path)
    parser.add_argument("--build-index", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--with-style", action="store_true")
    args = parser.parse_args()
    encoder = build_encoder(MODEL, "dinov3", args.device, "color")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.build_source:
        full_index(args.build_source, args.build_index, encoder, args.batch_size, preserve_variant_rows=True)
    baseline_dir = ROOT / "data/outputs/current-best-refresh-20260829"
    examples = read_jsonl(ROOT / "data/processed/glamour_examples_huiji_models_closed.jsonl")
    eligibility = load_slot_eligibility(ROOT / "data/processed/slot_quality_eligibility_20260801.jsonl")
    quality = {str(e["example_id"]): "FEET" in eligibility.get(str(e["example_id"]), set()) for e in examples}
    eq, slots, cases = _load_appearance_equivalence(), load_official_slots(), []
    baseline0 = json.loads((baseline_dir / "final-feet-fold01.json").read_text("utf-8"))
    index = args.output_dir / "frozen-gallery"
    metadata, vectors = full_index(ROOT / baseline0["branches"][0]["index_dir"], index, encoder, args.batch_size)
    candidate_profiles = {}
    if args.with_style:
        import torch
        from ffxiv_prism.cli.match import _classify_style_paths
        from ffxiv_prism.matching.garment_morphology import (FEET_STYLE_PROMPTS, StylePrediction,
            style_policy, footwear_style_consensus_policy, rerank_candidates_with_style)
        paths = [ROOT / r["local_path"] for r in metadata]
        profiles = _classify_style_paths("google/siglip2-base-patch16-224", FEET_STYLE_PROMPTS,
                                        paths, device=torch.device(args.device), batch_size=args.batch_size)
        candidate_profiles = dict(zip([r["local_path"] for r in metadata], profiles, strict=True))
    for fold in (1, 2, 3):
        baseline = json.loads((baseline_dir / f"final-feet-fold{fold:02}.json").read_text("utf-8"))
        context = build_fold_context(fold=fold, fold_manifest=ROOT / baseline["fold_manifest"],
            all_examples=examples, quality_by_id=quality,
            regions_dir=ROOT / "data/debug/outfit-regions/slot-quality-20260801", slot="FEET")
        style_values = style_queries(context, args.output_dir / "style-crops", args.device, args.batch_size) if args.with_style else {}
        paths = sorted({str(r["path"]) for rows in context["records"].values() for r in rows})
        query_cache = args.output_dir / f"query-{fold}.npz"
        if query_cache.is_file():
            with np.load(query_cache, allow_pickle=False) as saved:
                assert paths == saved["paths"].tolist()
                queries = dict(zip(paths, saved["embeddings"], strict=True))
        else:
            queries = dict(zip(paths, encoder.encode_paths([Path(p) for p in paths], args.batch_size), strict=True))
            np.savez_compressed(query_cache, paths=paths, embeddings=np.asarray(list(queries.values())))
        branches = []
        for branch in baseline["branches"]:
            branch_metadata = read_jsonl(ROOT / branch["index_dir"] / "metadata.jsonl")
            if args.with_style:
                # Recompute winning preview images; rank-only caches choose the
                # first render per item, which is not faithful for style reranking.
                with np.load(ROOT / branch["query_cache"], allow_pickle=False) as q:
                    branch_queries = dict(zip(q["paths"].tolist(), q["embeddings"], strict=True))
                branches.append(search_crops(branch_metadata, np.load(ROOT / branch["index_dir"] / "embeddings.npy"),
                    context["records"], branch_queries, 1000, device=args.device))
            else:
                branches.append(load_branch_search_cache(ROOT / branch["search_cache"], branch_metadata, candidate_k=1000,
                    index_dir=Path(branch["index_dir"]), query_cache=Path(branch["query_cache"])))
        support_cache = args.output_dir / f"search-{fold}.npz"
        if support_cache.is_file():
            support = load_branch_search_cache(support_cache, metadata, candidate_k=1000,
                                              index_dir=index, query_cache=query_cache)
        else:
            support = search_crops(metadata, vectors, context["records"], queries, 1000, device=args.device)
            save_branch_search_cache(support_cache, support, candidate_k=1000, index_dir=index, query_cache=query_cache)
        for number, example in enumerate(context["examples"], 1):
            eid = str(example["example_id"])
            records = context["records"][eid]
            per_example = [{str(r["path"]): b[str(r["path"])] for r in records} for b in branches + [support]]
            rankings = []
            for variant, selected in enumerate((per_example[:2], per_example)):
                merged = fuse_branch_search_results(selected, top_k=1000, rrf_k=1)
                ranked = fuse_crop_results(records, merged, top_k=1000, fusion="rrf", rrf_k=1)
                if args.with_style:
                    predictions = [StylePrediction.from_probabilities("FEET", p) for p in style_values[eid]]
                    policy = (footwear_style_consensus_policy(predictions[0], predictions[1:])
                              if variant else style_policy(predictions[0], minimum_confidence=0.7))
                    if policy.active:
                        profiles = {r["piece_id"]:candidate_profiles[r["local_preview"]] for r in ranked[:50]}
                        ranked = rerank_candidates_with_style(ranked, predictions[0], profiles,
                                                              rerank_k=50, blend=0.7, minimum_confidence=0.7)
                rankings.append(collapse_appearance_equivalent_candidates(filter_candidates_for_slot(ranked, "FEET", slots), eq))
            # Truth is consulted only after both label-free rankings are complete.
            truth = set(example["core_slot_labels"]["FEET"]["piece_ids"])
            truth |= {v for p in list(truth) for v in eq.get(p, ())}
            ranks = [next((i for i,r in enumerate(rows,1) if r["piece_id"] in truth), None) for rows in rankings]
            cases.append(dict(example_id=eid, fold=fold, gender=example.get("model_gender"),
                              race=example.get("model_race"), baseline_rank=ranks[0], revised_rank=ranks[1]))
            if number % 50 == 0:
                print(f"fold={fold} ranked={number}/{len(context['examples'])}", flush=True)
        print(f"fold={fold} cases={len(context['examples'])}", flush=True)
        del branches, support
    def metrics(rows):
        return dict(count=len(rows), **{name: {str(k): sum(r[name] is not None and r[name] <= k for r in rows)
                    for k in (1, 5, 10, 20, 100)} for name in ("baseline_rank", "revised_rank")})
    groups = defaultdict(list)
    for case in cases:
        for key in ("fold", "gender", "race"):
            groups[f"{key}:{case[key]}"].append(case)
    report = dict(scope="Frozen three-fold identity retrieval; same-model deduplicated. Existing validation folds, not independent test. Production height-style heuristics excluded.",
                  policy="Equal-weight per-crop DINOv2 + SigLIP2 + frozen full-render DINOv3 RRF(k=1), existing crop RRF(k=1)",
                  summary=metrics(cases), slices={k:metrics(v) for k,v in groups.items()}, cases=cases)
    if args.with_style:
        report["scope"] = "Frozen three-fold production-style replay; current geometry regenerated from saved pose, batch-32 SigLIP style classification; same-model deduplicated. Cross-slot gate excluded. Existing validation folds, not an independent test."
        report["policy"] += "; revised height reranking requires visible untruncated whole-shoe corroboration"
    write_json(args.output_dir / ("evaluation-with-style.json" if args.with_style else "evaluation.json"), report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
