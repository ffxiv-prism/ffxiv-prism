"""Compare old and continued weights on one frozen catalogue and fold protocol.

This is model selection on existing validation folds, not a new blind test.
Runtime style, presence, and local-detail heuristics are excluded from the
identity benchmark and checked separately before publication.
"""
# ruff: noqa: E402
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
np.linalg.inv(np.eye(2, dtype=np.float32))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModel
from ffxiv_prism.cli.match import _load_appearance_equivalence
from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.equipment_slots import load_official_slots, filter_candidates_for_slot
from ffxiv_prism.matching import inference
from ffxiv_prism.matching.inference import (
    current_best_configs, crop_weight, fuse_branch_search_results, fuse_crop_results,
    fuse_crops_then_branch_results, constrain_candidates_to_pool,
    rerank_occupied_slot_search_results, apply_occupied_slot_top1_gate,
    collapse_appearance_equivalent_candidates, select_slot_candidate_index_rows,
)
from ffxiv_prism.model_assets import load_pretrained, model_weight_files
from scripts.run_weight_refresh_training import sha256, write_json
from scripts.evaluate_glamour_embedding_model import build_encoder, search_crops
from scripts.evaluate_rank_ensemble_retrieval import save_branch_search_cache
from scripts.evaluate_retrieval_funnel import build_fold_context
from scripts.train_glamour_query_encoder import load_slot_eligibility, load_query_head_from_checkpoint, encode_records

from scripts.weight_refresh_metrics import compare_variants


def fuse_example(config, records, searches):
    if config.ensemble_fusion_order == "crop_then_branch":
        return fuse_crops_then_branch_results(
            records, searches, top_k=1000, candidate_k=1000, crop_fusion=config.fusion,
            crop_rrf_k=config.rrf_k, branch_rrf_k=config.ensemble_rrf_k,
            final_pool_branches=config.ensemble_final_pool_branches,
            branch_weights=config.ensemble_branch_weights or None,
        )
    pool = fuse_branch_search_results(searches[:config.ensemble_pool_branches], top_k=1000, rrf_k=config.ensemble_pool_rrf_k) if config.ensemble_pool_branches else None
    search = fuse_branch_search_results(searches, top_k=1000, rrf_k=config.ensemble_rrf_k, candidate_pool=pool, branch_weights=config.ensemble_branch_weights or None) if len(searches) > 1 else searches[0]
    candidates = fuse_crop_results(records, search, top_k=1000, fusion=config.fusion, rrf_k=config.rrf_k)
    if config.occupied_slot_multiplier is not None:
        adjusted = rerank_occupied_slot_search_results(search, occupied_multiplier=config.occupied_slot_multiplier, rrf_k=1)
        adjusted = fuse_crop_results(records, adjusted, top_k=1000, fusion=config.fusion, rrf_k=config.rrf_k)
        candidates = apply_occupied_slot_top1_gate(candidates, adjusted, max_base_rank=config.occupied_slot_gate_max_base_rank)
    if config.ensemble_final_pool_branches is not None:
        final = fuse_branch_search_results(searches[:config.ensemble_final_pool_branches], top_k=1000, rrf_k=config.ensemble_pool_rrf_k)
        final = fuse_crop_results(records, final, top_k=1000, fusion=config.fusion, rrf_k=config.rrf_k)
        candidates = constrain_candidates_to_pool(final, candidates)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slots", nargs="+", default=["HEAD", "BODY", "HANDS", "LEGS", "FEET"])
    args = parser.parse_args()
    os.chdir(ROOT)
    # Fusion formats the same preview paths millions of times. Resolve each
    # path once; this preserves production values without repeated filesystem I/O.
    inference.portable_path = lru_cache(maxsize=100000)(inference.portable_path)
    run = args.run_dir
    plan = json.loads((run / "training-plan.json").read_text("utf-8"))
    # Protect the exact candidate/data snapshot used to train and compare.
    for path, digest in plan["sources"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    jobs = {(j["slot"], j["fold"], j["branch"]): j for j in plan["jobs"]}
    training = json.loads((run / "training-status.json").read_text("utf-8"))
    for job in plan["jobs"]:
        if job["slot"] not in args.slots:
            continue
        evidence = training["jobs"].get(job["name"], {})
        if evidence.get("status") != "passed" or sha256(ROOT / job["checkpoint"]) != evidence.get("sha256"):
            raise ValueError(f"Challenger is not a verified completed training job: {job['name']}")
    configs = current_best_configs()
    if plan["selection"]["body_policy"] != "baseline":
        raise ValueError("This refresh evaluator requires the released BODY baseline policy")
    examples = read_jsonl(ROOT / "data/processed/glamour_examples_huiji_models_closed.jsonl")
    eligibility = load_slot_eligibility(ROOT / "data/processed/slot_quality_eligibility_20260801.jsonl")
    eq, official = _load_appearance_equivalence(), load_official_slots()
    device = torch.device(args.device)
    hash_cache = {}

    def file_hash(path):
        path = Path(path).resolve()
        if path not in hash_cache:
            hash_cache[path] = sha256(path)
        return hash_cache[path]

    def branch(name, model_id, index, checkpoint, records, genders, config):
        paths = sorted({str(r["path"]) for rows in records.values() for r in rows})
        contract = dict(model=model_id, checkpoint=file_hash(checkpoint) if checkpoint else None,
                        weights={str(p): file_hash(p) for p in model_weight_files(model_id)},
                        crop_bytes={p: file_hash(p) for p in paths}, preprocess="color",
                        metadata=file_hash(index / "metadata.jsonl"), vectors=file_hash(index / "embeddings.npy"),
                        routes=config.candidate_crop_routes, genders=genders,
                        evaluator=file_hash(Path(__file__)),
                        inference=file_hash(ROOT / "src/ffxiv_prism/matching/inference.py"))
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        cache = run / "evaluation-cache" / name
        cache.mkdir(parents=True, exist_ok=True)
        contract_file = cache / "contract.json"
        if contract_file.is_file():
            if json.loads(contract_file.read_text("utf-8"))["sha256"] != digest:
                raise ValueError(f"Evaluation cache contract changed: {name}")
        else:
            write_json(contract_file, dict(sha256=digest, contract=contract))
        metadata, vectors = select_slot_candidate_index_rows(read_jsonl(index / "metadata.jsonl"), np.load(index / "embeddings.npy", mmap_mode="r"), config.slot)
        query_path, search_path = cache / "queries.npz", cache / "search.npz"
        if not search_path.exists():
            if query_path.exists():
                with np.load(query_path, allow_pickle=False) as data:
                    queries = dict(zip(map(str, data["paths"]), data["embeddings"], strict=True))
            else:
                print(f"encoding={name} crops={len(paths)}", flush=True)
                if checkpoint:
                    model = load_pretrained(AutoModel, model_id).to(device)
                    head = load_query_head_from_checkpoint(checkpoint, model, device)
                    with torch.inference_mode():
                        queries = encode_records(model, head, records, 32, device)
                    del model, head
                else:
                    encoder = build_encoder(model_id, "auto", args.device, "color")
                    queries = dict(zip(paths, encoder.encode_paths([Path(p) for p in paths], 32), strict=True))
                    del encoder
                np.savez_compressed(query_path, paths=np.asarray(list(queries)), embeddings=np.asarray(list(queries.values())))
                gc.collect()
                torch.cuda.empty_cache()
            print(f"retrieving={name}", flush=True)
            search = search_crops(metadata, vectors, records, queries, 1000, device=args.device,
                                  candidate_crop_routes=config.candidate_crop_routes, candidate_gender_by_example=genders)
            save_branch_search_cache(search_path, search, candidate_k=1000, index_dir=index, query_cache=query_path)
            del search, queries
        exemplars = {}
        for row in metadata:
            for piece in row["pieces"]:
                exemplars.setdefault(int(piece["piece_id"]), {**piece, "local_path": row["local_path"],
                                                           "source_image_url": row["source_image_url"],
                                                           "image_row_id": row["image_row_id"]})
        with np.load(search_path, allow_pickle=False) as data:
            arrays = {str(p): (ids, scores) for p, ids, scores in zip(data["paths"], data["piece_ids"], data["scores"], strict=True)}
        del metadata, vectors
        gc.collect()
        torch.cuda.empty_cache()
        return arrays, exemplars

    for slot in args.slots:
        config = configs[slot]
        variants = {"baseline": ["old2"], "continued": ["new2"]}
        if slot == "HEAD":
            variants = {"baseline": ["old2", "zero0", "old3"], "continued": ["new2", "zero0", "new3"],
                        "dinov2_only": ["new2", "zero0", "old3"], "dinov3_only": ["old2", "zero0", "new3"]}
        else:
            for variant in variants:
                variants[variant] += [f"zero{i}" for i in range(len(config.ensemble_index_dirs))]
        cases = []
        quality = {str(e["example_id"]): slot in eligibility.get(str(e["example_id"]), set()) for e in examples}
        for fold in (1, 2, 3):
            context = build_fold_context(fold=fold, fold_manifest=ROOT / f"data/interim/appearance-disjoint-cv-v55-slot-recovery-{slot.lower()}/fold-{fold:02}.json",
                                         all_examples=examples, quality_by_id=quality, regions_dir=ROOT / "data/debug/outfit-regions/slot-quality-20260801", slot=slot)
            records = {eid: [r for r in rows if crop_weight(r) > 0 and (config.query_crop_names is None or r["name"] in config.query_crop_names)] for eid, rows in context["records"].items()}
            genders = {str(e["example_id"]): str(e["model_gender"]).lower() for e in context["examples"]} if config.gender_routing else None
            primary = jobs[slot, fold, "dinov2"]
            specs = {"old2": ("facebook/dinov2-base", config.index_dir, ROOT / primary["parent"]),
                     "new2": ("facebook/dinov2-base", config.index_dir, ROOT / primary["checkpoint"])}
            for i, (model_id, index) in enumerate(zip(config.ensemble_model_ids, config.ensemble_index_dirs, strict=True)):
                if slot == "HEAD" and i == 1:
                    job = jobs[slot, fold, "dinov3"]
                    specs.update(old3=(model_id, index, ROOT / job["parent"]), new3=(model_id, index, ROOT / job["checkpoint"]))
                else:
                    specs[f"zero{i}"] = (model_id, index, None)
            branches = {key: branch(f"{slot.lower()}-fold{fold:02}-{key}", *spec, records, genders, config) for key, spec in specs.items()}
            for example in context["examples"]:
                eid = str(example["example_id"])
                query_rows = records[eid]
                searches = {}
                for key, (arrays, exemplars) in branches.items():
                    searches[key] = {}
                    for row in query_rows:
                        path = str(row["path"])
                        ids, scores = arrays[path]
                        searches[key][path] = [dict(exemplars[int(pid)], piece_id=int(pid), rank=i, score=float(score)) for i, (pid, score) in enumerate(zip(ids, scores, strict=True), 1) if pid >= 0]
                # Truth is only consulted after retrieval for all variants.
                truth = {int(p) for p in example["core_slot_labels"][slot]["piece_ids"]}
                truth |= {v for p in list(truth) for v in eq.get(p, ())}
                case = dict(example_id=eid, fold=fold, race=example.get("model_race"), gender=example.get("model_gender"))
                for variant, keys in variants.items():
                    ranking = fuse_example(config, query_rows, [searches[k] for k in keys])
                    ranking = collapse_appearance_equivalent_candidates(filter_candidates_for_slot(ranking, slot, official), eq)
                    case[variant] = next((i for i, r in enumerate(ranking, 1) if int(r["piece_id"]) in truth), None)
                cases.append(case)
            write_json(run / f"compare-{slot.lower()}-fold{fold:02}.json", dict(slot=slot, fold=fold, variants=variants, cases=cases[-len(context["examples"]):]))
            print(f"evaluated={slot} fold={fold}", flush=True)
            del branches
            gc.collect()
        result = compare_variants(cases, list(variants))
        result.update(slot=slot, variants=variants, cases=cases, candidate_indexes=plan["selection"]["candidate_indexes"],
                      scope="strict appearance-disjoint folds; official-equivalence deduplicated identity retrieval; production heuristics excluded")
        write_json(run / f"comparison-{slot.lower()}.json", result)
        print(f"selected={slot} variant={result['selected']} metrics={result['groups']['all'][result['selected']]}", flush=True)


if __name__ == "__main__":
    main()
