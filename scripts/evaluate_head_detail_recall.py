"""Replay fixed local-detail recall on the frozen three-fold HEAD evaluation.

Uses fold-specific cached adapter rankings; only new detail crops are encoded.
Truth labels are read by the scoring function, never by crop or rank policies.
Outputs are local audit evidence, not a replacement for registered metrics.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
np.linalg.inv(np.eye(2, dtype=np.float32))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from ffxiv_prism.cli.match import _load_appearance_equivalence, _load_appearance_equivalence_catalog
from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching.head_detail import detail_records, recall_detail_candidate
from ffxiv_prism.matching.equipment_slots import filter_candidates_for_slot, load_official_slots
from ffxiv_prism.matching.inference import (
    attach_appearance_equivalent_candidates, collapse_appearance_equivalent_candidates,
    fuse_branch_search_results, fuse_crop_results, crop_records,
)
from ffxiv_prism.paths import resolve_project_path
from ffxiv_prism.vision.outfit_regions import (
    Keypoint, build_lower_face_regions, build_head_detail_regions, save_region_crops,
)
from scripts.evaluate_glamour_embedding_model import build_encoder, search_crops
from scripts.evaluate_retrieval_funnel import build_fold_context
from scripts.train_glamour_query_encoder import load_slot_eligibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = ROOT / "data/outputs/current-best-refresh-20260829"
    examples = read_jsonl(ROOT / "data/processed/glamour_examples_huiji_models_closed.jsonl")
    eligibility = load_slot_eligibility(ROOT / "data/processed/slot_quality_eligibility_20260801.jsonl")
    quality = {str(e["example_id"]): "HEAD" in eligibility.get(str(e["example_id"]), set()) for e in examples}
    eq, catalog, official = _load_appearance_equivalence(), _load_appearance_equivalence_catalog(), load_official_slots()

    def canonical(ranking):
        ranking = attach_appearance_equivalent_candidates(ranking, eq, piece_catalog=catalog, max_equivalents_per_anchor=None)
        return collapse_appearance_equivalent_candidates(filter_candidates_for_slot(ranking, "HEAD", official), eq)

    contexts, details, paths = [], {}, []
    for fold in (1, 2, 3):
        context = build_fold_context(
            fold=fold, fold_manifest=ROOT / f"data/interim/appearance-disjoint-cv-v55-slot-recovery-head/fold-{fold:02}.json",
            all_examples=examples, quality_by_id=quality,
            regions_dir=ROOT / "data/debug/outfit-regions/slot-quality-20260801", slot="HEAD",
        )
        contexts.append(context)
        for eid, records in context["records"].items():
            if not records:
                details[eid] = []
                continue
            folder = Path(records[0]["path"]).parent.parent
            manifest = json.loads(next(folder.glob("*-regions.json")).read_text("utf-8"))
            with Image.open(resolve_project_path(manifest["source_image"])) as image:
                points = [Keypoint(p["x"], p["y"], p["score"]) for p in manifest["keypoints"]]
                regions = build_head_detail_regions(build_lower_face_regions(points, image.size))
                crop_paths = save_region_crops(image, regions, args.output_dir / "crops", eid.replace(":", "-"))
            details[eid] = detail_records(crop_records(regions, crop_paths)["query"])
            paths.extend(Path(r["path"]) for r in details[eid])
    model_id = "google/siglip2-base-patch16-224"
    # Cache binds crop bytes and encoder ID. Use a fresh output directory
    # after replacing weights or preprocessing, as documented below.
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    digest.update(model_id.encode())
    cache = args.output_dir / "detail-queries.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["contract"].item()) != digest.hexdigest():
                raise ValueError("Detail crop cache changed; use a fresh output directory")
            vectors = data["embeddings"]
    else:
        encoder = build_encoder(model_id, "auto", args.device, "color")
        vectors = encoder.encode_paths(paths, 32)
        np.savez_compressed(cache, contract=digest.hexdigest(), embeddings=vectors)
        del encoder
    queries = dict(zip(map(str, paths), vectors, strict=True))
    cases = []
    for context in contexts:
        fold = context["fold"]
        report = json.loads((base / f"final-head-fold{fold:02}.json").read_text("utf-8"))
        branches = report["branches"]
        index = resolve_project_path(branches[1]["index_dir"])
        metadata = read_jsonl(index / "metadata.jsonl")
        embeddings = np.load(index / "embeddings.npy", mmap_mode="r")
        local = search_crops(metadata, embeddings, {eid: details[eid] for eid in context["records"]}, queries, 100, device=args.device)
        arrays, exemplars = [], []
        for branch in branches:
            with np.load(resolve_project_path(branch["search_cache"]), allow_pickle=False) as data:
                if int(data["candidate_k"].item()) != 1000:
                    raise ValueError("Unexpected baseline retrieval size")
                if Path(str(data["index_dir"].item())).resolve() != resolve_project_path(branch["index_dir"]).resolve():
                    raise ValueError("Baseline index cache mismatch")
                arrays.append({str(p): (ids, scores) for p, ids, scores in zip(data["paths"], data["piece_ids"], data["scores"], strict=True)})
            by_piece = {}
            for row in read_jsonl(resolve_project_path(branch["index_dir"]) / "metadata.jsonl"):
                for piece in row["pieces"]:
                    by_piece.setdefault(int(piece["piece_id"]), {**row, **piece})
            exemplars.append(by_piece)
        for example in context["examples"]:
            eid = str(example["example_id"])
            upper = next(r for r in context["records"][eid] if r["name"] == "head_upper_accessory")
            path = str(upper["path"])
            searches = []
            for array, exemplar in zip(arrays, exemplars, strict=True):
                ids, scores = array[path]
                searches.append({path: [dict(exemplar[int(pid)], piece_id=int(pid), rank=i, score=float(score)) for i, (pid, score) in enumerate(zip(ids, scores, strict=True), 1) if pid >= 0]})
            pool = fuse_branch_search_results(searches[:2], top_k=1000, rrf_k=1)
            search = fuse_branch_search_results(searches, top_k=1000, rrf_k=10, candidate_pool=pool)
            before = canonical(fuse_crop_results([upper], search, top_k=1000))
            views = {r["name"]: canonical(fuse_crop_results([r], local, top_k=100)) for r in details[eid]}
            after, diagnostic = recall_detail_candidate(before, views, eq)
            # Labels enter only after all rankings have been constructed.
            truth = {int(p) for p in example["core_slot_labels"]["HEAD"]["piece_ids"]}
            truth |= {v for p in list(truth) for v in eq.get(int(p), ())}
            def rank(rows):
                return next((i for i, r in enumerate(rows, 1) if int(r["piece_id"]) in truth), None)
            assert [r["piece_id"] for r in before[:5]] == [r["piece_id"] for r in after[:5]]
            cases.append(dict(example_id=eid, fold=fold, gender=example.get("model_gender"), race=example.get("model_race"), before=rank(before), after=rank(after), diagnostic=diagnostic))
        print(f"fold={fold} cases={len(cases)}", flush=True)
    groups = defaultdict(list)
    for case in cases:
        for group in ("all", f"fold:{case['fold']}", f"gender:{case['gender']}", f"race:{case['race']}"):
            groups[group].append(case)
    summary = {}
    for name, rows in groups.items():
        summary[name] = dict(count=len(rows), changed=sum(r["diagnostic"]["ranking_changed"] for r in rows), hits={str(k): {phase: sum(r[phase] is not None and r[phase] <= k for r in rows) for phase in ("before", "after")} for k in (1, 5, 10, 20, 100, 1000)})
    output = dict(policy="two top1 votes + third top10; preserve first five", benchmark_scope="frozen fold-specific identity retrieval with official equivalence; before cross-slot and no-equipment gates", summary=summary, cases=cases)
    (args.output_dir / "evaluation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
