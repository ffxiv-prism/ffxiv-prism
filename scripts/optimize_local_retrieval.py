"""Select global/local rank fusion on validation folds, then evaluate a locked policy on blind_test.

User screenshots never enter fitting or scoring. All appearance labels are used
only for metrics after image-only features and rankings have been constructed.
"""

# ruff: noqa: E402
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np

np.linalg.inv(np.eye(2, dtype=np.float32))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from transformers import AutoModel
from ffxiv_prism.cli import match as m
from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.matching import inference as inf
from ffxiv_prism.matching.eye_accessory import encode_candidate_eyes, MODEL_IDS
from ffxiv_prism.matching.head_detail import HEAD_TYPE_PROMPTS
from ffxiv_prism.matching.garment_morphology import STYLE_PROMPTS_BY_SLOT
from ffxiv_prism.matching.local_fusion import (
    fused_scores,
    ranks_of_truth,
)
from ffxiv_prism.matching.local_retrieval import (
    catalogue,
    visual_gates,
    grouped_eye_ranks,
    search_ranks,
)
from ffxiv_prism.model_assets import load_pretrained
from ffxiv_prism.paths import resolve_project_path
from ffxiv_prism.vision.outfit_regions import (
    Keypoint,
    build_outfit_regions,
    save_region_crops,
)
from scripts.evaluate_retrieval_funnel import build_fold_context
from scripts.evaluate_glamour_embedding_model import build_encoder, search_crops
from scripts.train_glamour_query_encoder import (
    load_slot_eligibility,
    encode_records,
    load_query_head_from_checkpoint,
)
from scripts.evaluate_weight_refresh import fuse_example

RUN = ROOT / "data/outputs/local-fusion-20260929"
GATES = {"HEAD": ("eye_mean", "eye_min", "all"), "LEGS": ("shorts", "exposed", "all")}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(slot, fold, model):
    name = f"refresh_v110_{slot.lower()}_fold{fold:02}_{model}_e{3 if model == 'dinov3' else 10}.pt"
    return ROOT / "data/models" / name


def canonical_catalog(config):
    eq = m._load_appearance_equivalence()
    official = m.load_official_slots()
    return catalogue(
        read_jsonl(config.index_dir / "metadata.jsonl"), config.slot, official, eq
    )


class CompactSearch:
    def __init__(self, path, metadata):
        self.exemplars = {}
        for row in metadata:
            for piece in row["pieces"]:
                self.exemplars.setdefault(int(piece["piece_id"]), {**row, **piece})
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {
                str(Path(p)): (ids, scores)
                for p, ids, scores in zip(
                    data["paths"], data["piece_ids"], data["scores"], strict=True
                )
            }

    def __getitem__(self, path):
        ids, scores = self.arrays[str(Path(path))]
        return [
            dict(
                self.exemplars[int(pid)], piece_id=int(pid), rank=i, score=float(score)
            )
            for i, (pid, score) in enumerate(zip(ids, scores, strict=True), 1)
            if pid >= 0
        ]


def cached_branch(
    slot, fold, role, config, records, model_id, index, ckpt, tag, device
):
    from scripts.evaluate_rank_ensemble_retrieval import save_branch_search_cache

    metadata, vectors = inf.select_slot_candidate_index_rows(
        read_jsonl(index / "metadata.jsonl"),
        np.load(index / "embeddings.npy", mmap_mode="r"),
        slot,
    )
    prior = (
        ROOT
        / f"data/outputs/model-refresh-20260926/evaluation-cache/{slot.lower()}-fold{fold:02}-{tag}"
    )
    cache = RUN / "branches" / f"{role}-{slot}-{fold}-{tag}.npz"
    if role == "validation" and (prior / "search.npz").is_file():
        contract = json.loads((prior / "contract.json").read_text("utf-8"))["contract"]
        assert contract["metadata"] == digest(index / "metadata.jsonl")
        assert contract["vectors"] == digest(index / "embeddings.npy")
        if ckpt:
            assert contract["checkpoint"] == digest(ckpt)
        return CompactSearch(prior / "search.npz", metadata)
    if not cache.exists():
        paths = sorted({str(r["path"]) for rs in records.values() for r in rs})
        if ckpt:
            model = load_pretrained(AutoModel, model_id).to(device)
            head = load_query_head_from_checkpoint(ckpt, model, device)
            with torch.inference_mode():
                queries = encode_records(model, head, records, 32, device)
            del model, head
        else:
            encoder = build_encoder(model_id, "auto", str(device), "color")
            queries = dict(
                zip(
                    paths,
                    encoder.encode_paths([Path(p) for p in paths], 32),
                    strict=True,
                )
            )
            del encoder
        search = search_crops(
            metadata,
            vectors,
            records,
            queries,
            1000,
            device=str(device),
            candidate_crop_routes=config.candidate_crop_routes,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_branch_search_cache(
            cache,
            search,
            candidate_k=1000,
            index_dir=index,
            query_cache=cache.with_suffix(".queries.npz"),
        )
    m._release_memory()
    return CompactSearch(cache, metadata)


def prepare(slot, role, device):
    output = RUN / f"{slot}-{role}.npz"
    info_path = output.with_suffix(".json")
    config = inf.current_best_configs()[slot]
    if output.exists():
        info = json.loads(info_path.read_text("utf-8"))
        assert info["index_sha256"] == digest(config.index_dir / "metadata.jsonl")
        return dict(np.load(output, allow_pickle=False)), info
    examples = read_jsonl(
        ROOT / "data/processed/glamour_examples_huiji_models_closed.jsonl"
    )
    eligibility = load_slot_eligibility(
        ROOT / "data/processed/slot_quality_eligibility_20260801.jsonl"
    )
    quality = {
        e["example_id"]: slot in eligibility.get(e["example_id"], set())
        for e in examples
    }
    full, ids, keys = canonical_catalog(config)
    columns = {pid: i for i, pid in enumerate(ids)}
    eq = m._load_appearance_equivalence()
    official = m.load_official_slots()
    cases = []
    base_rows = []
    local_records = {}
    local_paths = []
    reliable = []
    folds = (1, 2, 3) if role == "validation" else (1,)
    for fold in folds:
        context = build_fold_context(
            fold=fold,
            fold_role=role,
            fold_manifest=ROOT
            / f"data/interim/appearance-disjoint-cv-v55-slot-recovery-{slot.lower()}/fold-{fold:02}.json",
            all_examples=examples,
            quality_by_id=quality,
            regions_dir=ROOT / "data/debug/outfit-regions/slot-quality-20260801",
            slot=slot,
        )
        base_records = {
            eid: [
                r
                for r in rs
                if inf.crop_weight(r) > 0
                and (
                    config.query_crop_names is None
                    or r["name"] in config.query_crop_names
                )
            ]
            for eid, rs in context["records"].items()
        }
        specs = [
            (
                "facebook/dinov2-base",
                config.index_dir,
                checkpoint(slot, fold, "dinov2"),
                "old2",
            )
        ]
        specs.extend(
            (
                mid,
                index,
                checkpoint(slot, fold, "dinov3") if slot == "HEAD" and i == 1 else None,
                "old3" if slot == "HEAD" and i == 1 else f"zero{i}",
            )
            for i, (mid, index) in enumerate(
                zip(config.ensemble_model_ids, config.ensemble_index_dirs, strict=True)
            )
        )
        branches = [
            cached_branch(slot, fold, role, config, base_records, *spec, device)
            for spec in specs
        ]
        for example in context["examples"]:
            eid = example["example_id"]
            rs = context["records"][eid]
            rows = fuse_example(
                config,
                base_records[eid],
                [
                    {str(r["path"]): b[str(r["path"])] for r in base_records[eid]}
                    for b in branches
                ],
            )
            rows = inf.collapse_appearance_equivalent_candidates(
                m.filter_candidates_for_slot(rows, slot, official), eq
            )
            ranks = np.full(len(ids), 1_000_000, dtype=np.float32)
            for rank, row in enumerate(rows, 1):
                key = keys.get(int(row["piece_id"]), int(row["piece_id"]))
                if key in columns:
                    ranks[columns[key]] = rank
            base_rows.append(ranks)
            folder = Path(rs[0]["path"]).parent.parent
            manifest = json.loads(
                next(folder.glob("*-regions.json")).read_text("utf-8")
            )
            names = (
                ("head_eye_left", "head_eye_right", "head_tight", "head_face_accessory")
                if slot == "HEAD"
                else ("legs_waist_thighs", "legs_thighs")
            )
            with Image.open(resolve_project_path(manifest["source_image"])) as im:
                regions_by_name = {
                    r.name: r
                    for r in build_outfit_regions(
                        [
                            Keypoint(p["x"], p["y"], p["score"])
                            for p in manifest["keypoints"]
                        ],
                        im.size,
                    )
                }
                regions = [regions_by_name[n] for n in names]
                paths = save_region_crops(
                    im,
                    regions,
                    RUN / "crops" / role,
                    eid.replace(":", "-") + "-" + slot,
                )
            local_records[eid] = inf.crop_records(regions, paths, example_id=eid)[eid]
            local_paths.extend(paths)
            reliable.append(
                all(
                    r.visibility == "visible"
                    and r.confidence >= 0.55
                    and "source_edge_truncation_risk" not in r.tags
                    for r in regions
                )
            )
            cases.append(
                dict(
                    example_id=eid,
                    fold=fold,
                    glamour_id=example["glamour_id"],
                    name=example["core_slot_labels"][slot]["item_name"],
                    truth=example["core_slot_labels"][slot]["piece_ids"],
                    race=example.get("model_race"),
                    gender=example.get("model_gender"),
                )
            )
        print("baseline", slot, role, fold, len(cases), flush=True)
    base = np.asarray(base_rows)
    local = np.full(
        (len(cases), 2 if slot == "HEAD" else 1, 2, len(ids)),
        1_000_000,
        dtype=np.float32,
    )
    gates = np.ones((len(cases), 3), dtype=np.float32)
    if slot == "HEAD":
        paths = [
            Path(r["path"]) for c in cases for r in local_records[c["example_id"]][2:]
        ]
        probs = m._classify_style_paths(
            MODEL_IDS[1], HEAD_TYPE_PROMPTS, paths, device=device, batch_size=32
        ).reshape(len(cases), 2, -1)
        gates = visual_gates(slot, probs)
        available = [r for r in full if resolve_project_path(r["local_path"]).is_file()]
        paths = [resolve_project_path(r["local_path"]) for r in available]
        groups = [[] for _ in ids]
        for i, row in enumerate(available):
            for p in row["pieces"]:
                groups[
                    columns[keys.get(int(p["piece_id"]), int(p["piece_id"]))]
                ].append(i)
        qpaths = [
            Path(r["path"]) for c in cases for r in local_records[c["example_id"]][:2]
        ]
        for b, mid in enumerate(MODEL_IDS):
            encoder = build_encoder(mid, "auto", str(device), "color")
            vectors = encode_candidate_eyes(encoder, paths, batch_size=32)
            queries = encoder.encode_paths(qpaths, 32).reshape(len(cases), 2, -1)
            del encoder
            cv = torch.as_tensor(vectors.reshape(-1, vectors.shape[-1]), device=device)
            for start in range(0, len(cases), 32):
                q = torch.as_tensor(queries[start : start + 32], device=device)
                raw = (q @ cv.T).reshape(len(q), 2, len(paths), 2).cpu().numpy()
                local[start : start + len(q), b] = grouped_eye_ranks(raw, groups)
            del cv, queries, vectors
            m._release_memory()
            print("local", slot, mid, "completed", flush=True)
    else:
        paths = [Path(r["path"]) for c in cases for r in local_records[c["example_id"]]]
        probs = m._classify_style_paths(
            "google/siglip2-base-patch16-224",
            STYLE_PROMPTS_BY_SLOT["LEGS"],
            paths,
            device=device,
            batch_size=32,
        ).reshape(len(cases), 2, -1)
        gates = visual_gates(slot, probs)
        metadata, vectors = inf.select_slot_candidate_index_rows(
            read_jsonl(config.index_dir / "metadata.jsonl"),
            np.load(config.index_dir / "embeddings.npy", mmap_mode="r"),
            slot,
        )
        for fold in folds:
            selected = {
                c["example_id"]: local_records[c["example_id"]]
                for c in cases
                if c["fold"] == fold
            }
            model = load_pretrained(AutoModel, "facebook/dinov2-base").to(device)
            head = load_query_head_from_checkpoint(
                checkpoint(slot, fold, "dinov2"), model, device
            )
            with torch.inference_mode():
                queries = encode_records(model, head, selected, 32, device)
            del model, head
            searches = search_crops(
                metadata,
                vectors,
                selected,
                queries,
                1000,
                device=str(device),
                candidate_crop_routes={
                    n: ("legs_thighs",) for n in ("legs_waist_thighs", "legs_thighs")
                },
            )
            for i, c in enumerate(cases):
                if c["fold"] != fold:
                    continue
                for v, r in enumerate(selected[c["example_id"]]):
                    local[i, 0, v] = search_ranks(searches[str(r["path"])], ids, keys)
            m._release_memory()
            print("local", slot, fold, "completed", flush=True)
    gates *= np.asarray(reliable)[:, None]
    truth = np.zeros_like(base, dtype=bool)
    for i, c in enumerate(cases):
        for pid in c["truth"]:
            key = keys.get(int(pid), int(pid))
            if key in columns:
                truth[i, columns[key]] = True
    group = np.asarray(
        [
            bool(
                re.search(
                    r"glasses|spectacles|goggles|eyepatch|blindfold|visor",
                    c["name"],
                    re.I,
                )
            )
            if slot == "HEAD"
            else bool(
                re.search(r"shorts|pantalettes|tanga|culottes|smalls", c["name"], re.I)
            )
            for c in cases
        ]
    )
    data = dict(
        base=base,
        local=local,
        gates=gates,
        truth=truth,
        group=group,
        ids=np.asarray(ids),
    )
    np.savez_compressed(output, **data)
    info = dict(
        slot=slot,
        role=role,
        cases=cases,
        index_sha256=digest(config.index_dir / "metadata.jsonl"),
        catalogue_size=len(ids),
        reliable_count=sum(reliable),
        scope="appearance-disjoint identity retrieval; production style/local-detail/no-equipment/cross-slot heuristics excluded",
    )
    write(info_path, info)
    return data, info


def metrics(ranks, baseline=None):
    r = np.minimum(ranks, 1001)
    value = dict(
        count=len(r),
        hits={str(k): int((r <= k).sum()) for k in (1, 5, 10, 20, 100)},
        mrr=float(np.where(ranks <= 1000, 1 / r, 0).mean()),
        mean_capped_rank=float(r.mean()),
        mean_log_rank=float(np.log(r).mean()),
    )
    if baseline is not None:
        value.update(
            rescued_from_beyond100_to10=int(((baseline > 100) & (r <= 10)).sum()),
            lost_top10=int(((baseline <= 10) & (r > 10)).sum()),
            gained_top10=int(((baseline > 10) & (r <= 10)).sum()),
        )
    return value


def evaluate(data, policy):
    scores = fused_scores(
        data["base"],
        data["local"],
        data["gates"][:, policy["gate"]],
        weight=policy["weight"],
        rrf_k=policy["rrf_k"],
        view_fusion=policy["view_fusion"],
    )
    return ranks_of_truth(scores, data["truth"])


def select(slot, data, info):
    base = np.where(data["truth"], data["base"], 1_000_000).min(1)
    group = data["group"]
    configs = []
    # Declared before any blind_test labels are evaluated. A modest broad loss is
    # allowed if mean log-rank and the weak-category metric improve.
    for k in (1, 10, 30, 60):
        for weight in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            for gate in range(3):
                for vf in ("max", "mean"):
                    cfg = dict(
                        weight=weight,
                        rrf_k=k,
                        gate=gate,
                        gate_name=GATES[slot][gate],
                        view_fusion=vf,
                    )
                    after = evaluate(data, cfg)
                    allm = metrics(after, base)
                    gm = metrics(after[group], base[group])
                    bm = metrics(base)
                    loss10 = bm["hits"]["10"] - allm["hits"]["10"]
                    loss1 = bm["hits"]["1"] - allm["hits"]["1"]
                    objective = allm["mean_log_rank"] + 0.5 * gm["mean_log_rank"]
                    eligible = loss10 <= 0.03 * len(base) and loss1 <= 0.03 * len(base)
                    configs.append(
                        dict(
                            policy=cfg,
                            objective=objective,
                            eligible=eligible,
                            all=allm,
                            weak_category=gm,
                        )
                    )
    baseline_objective = (
        metrics(base)["mean_log_rank"] + 0.5 * metrics(base[group])["mean_log_rank"]
    )
    chosen = min((c for c in configs if c["eligible"]), key=lambda c: c["objective"])
    if chosen["objective"] >= baseline_objective:
        chosen = dict(
            policy=dict(
                weight=0.0,
                rrf_k=10,
                gate=0,
                gate_name=GATES[slot][0],
                view_fusion="max",
            ),
            objective=baseline_objective,
            eligible=True,
            all=metrics(base, base),
            weak_category=metrics(base[group], base[group]),
        )
    result = dict(
        slot=slot,
        selection_role="validation",
        objective="mean_log_rank(all) + 0.5 * mean_log_rank(predeclared name-based weak category); at most 3 percentage points Top1/Top10 loss",
        baseline=metrics(base),
        baseline_weak_category=metrics(base[group]),
        selected=chosen,
        experiments=configs,
    )
    write(RUN / f"{slot}-selection.json", result)
    print("SELECTED", slot, json.dumps(chosen), flush=True)
    return chosen["policy"]


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validation", "test"], required=True)
    parser.add_argument("--slots", nargs="+", default=["HEAD", "LEGS"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", type=Path, default=RUN)
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.stage == "validation" and (
        any(RUN.glob("*-blind_test.*")) or (RUN / "test-opened.json").exists()
    ):
        raise ValueError(
            "This run has opened blind_test; validation reselection is prohibited"
        )
    inf.portable_path = lru_cache(maxsize=100000)(inf.portable_path)
    policy_path = RUN / "locked-policy.json"
    if args.stage == "test" and not policy_path.is_file():
        raise ValueError("Select and lock parameters before opening blind_test")
    policies = (
        json.loads(policy_path.read_text("utf-8")) if policy_path.exists() else {}
    )
    if args.stage == "test":
        marker = RUN / "test-opened.json"
        policy_sha = digest(policy_path)
        if (
            marker.exists()
            and json.loads(marker.read_text("utf-8"))["policy_sha256"] != policy_sha
        ):
            raise ValueError("The locked policy changed after blind_test was opened")
        write(marker, dict(policy_sha256=policy_sha))
    device = torch.device(args.device)
    for slot in args.slots:
        data, info = prepare(
            slot, "validation" if args.stage == "validation" else "blind_test", device
        )
        if args.stage == "validation":
            policies[slot] = select(slot, data, info)
            write(policy_path, policies)
        else:
            base = np.where(data["truth"], data["base"], 1_000_000).min(1)
            after = evaluate(data, policies[slot])
            group = data["group"]
            report = dict(
                slot=slot,
                policy=policies[slot],
                locked_policy_sha256=digest(policy_path),
                baseline=metrics(base),
                optimized=metrics(after, base),
                weak_category=dict(
                    baseline=metrics(base[group]),
                    optimized=metrics(after[group], base[group]),
                ),
                cases=[
                    dict(c, before=int(b), after=int(a))
                    for c, b, a in zip(info["cases"], base, after, strict=True)
                ],
            )
            write(RUN / f"{slot}-test.json", report)
            print(
                "TEST",
                slot,
                json.dumps({k: v for k, v in report.items() if k != "cases"}),
                flush=True,
            )


if __name__ == "__main__":
    main()
