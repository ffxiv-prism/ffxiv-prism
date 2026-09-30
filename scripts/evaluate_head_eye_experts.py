"""Replay the frozen eye-expert policy on appearance-disjoint validation/held-out data.

Uses the production geometry, content caches and scorer. This script never
selects parameters from the held-out set or uses a user screenshot as a label.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
np.linalg.inv(np.eye(2, dtype=np.float32))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from ffxiv_prism.cli import match as m
from ffxiv_prism.matching import inference as inf
from ffxiv_prism.matching.aligned_eyes import candidate_eye_boxes, asset_signature
from ffxiv_prism.matching.eye_accessory import MODEL_IDS, encode_candidate_eyes
from ffxiv_prism.matching.eye_experts import (
    EYE_COVERAGE_PROMPTS, EYE_EXPERT_POLICY, classify_eye_coverage,
    grouped_coverage_profiles, eye_expert_scores,
)
from ffxiv_prism.matching.local_fusion import fused_scores, ranks_of_truth
from ffxiv_prism.matching.local_retrieval import grouped_eye_ranks
from ffxiv_prism.paths import resolve_project_path
from scripts import optimize_local_retrieval as baseline
from scripts.evaluate_glamour_embedding_model import build_encoder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(data, info, source, output, role, device):
    full, ids, keys = baseline.canonical_catalog(inf.current_best_configs()["HEAD"])
    if not np.array_equal(ids, data["ids"]):
        raise ValueError("Baseline and current candidate appearances differ")
    full = [r for r in full if resolve_project_path(r["local_path"]).is_file()]
    paths = [resolve_project_path(r["local_path"]) for r in full]
    groups = [[] for _ in ids]
    columns = {pid: i for i, pid in enumerate(ids)}
    for i, row in enumerate(full):
        for piece in row["pieces"]:
            pid = int(piece["piece_id"])
            groups[columns[keys.get(pid, pid)]].append(i)
    query_paths = [source / "crops" / role /
                   f"{case['example_id'].replace(':', '-')}-HEAD-{name}.jpg"
                   for case in info["cases"]
                   for name in ("head_eye_left", "head_eye_right")]
    profile_paths = [source / "crops" / role /
                     f"{case['example_id'].replace(':', '-')}-HEAD-{name}.jpg"
                     for case in info["cases"]
                     for name in ("head_tight", "head_face_accessory")]
    geometry = candidate_eye_boxes(paths, device=device, batch_size=32)
    boxes = {str(p): r["boxes"] for p, r in zip(paths, geometry, strict=True)}
    contract = json.dumps(dict(
        version=1, baseline=digest(source / f"HEAD-{role}.npz"),
        metadata=info["index_sha256"], groups=groups, geometry=geometry,
        images=[(str(p), digest(p)) for p in paths + query_paths + profile_paths],
        assets={mid: asset_signature(mid) for mid in MODEL_IDS},
        prompts=EYE_COVERAGE_PROMPTS,
    ), sort_keys=True)
    cache = output / f"features-{role}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as cached:
            if str(cached["contract"]) != contract:
                raise ValueError("Feature provenance changed; use a new output directory")
            return {k: cached[k] for k in ("local", "query_profiles", "candidate_profiles")}
    profiles = classify_eye_coverage(
        profile_paths + paths, classifier=m._classify_style_paths,
        device=device, batch_size=32,
    )
    query_profiles = profiles[:len(profile_paths)].reshape(len(info["cases"]), 2, 6)
    candidate_profiles = grouped_coverage_profiles(profiles[len(profile_paths):], groups)
    all_ranks = []
    for mid in MODEL_IDS:
        encoder = build_encoder(mid, "auto", str(device), "color")
        queries = encoder.encode_paths(query_paths, 32).reshape(len(info["cases"]), 2, -1)
        candidates = encode_candidate_eyes(encoder, paths, batch_size=32, boxes_by_path=boxes)
        cv = torch.as_tensor(candidates.reshape(-1, candidates.shape[-1]), device=device)
        rows = []
        for start in range(0, len(queries), 32):
            q = torch.as_tensor(queries[start:start+32], device=device)
            raw = (q @ cv.T).reshape(len(q), 2, len(paths), 2).cpu().numpy()
            rows.append(grouped_eye_ranks(raw, groups))
        all_ranks.append(np.concatenate(rows))
        del encoder, candidates, cv
        m._release_memory()
    result = dict(local=np.stack(all_ranks, 1), query_profiles=query_profiles,
                  candidate_profiles=candidate_profiles)
    np.savez_compressed(cache, contract=contract, **result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("validation", "blind_test"), required=True)
    parser.add_argument("--baseline-dir", type=Path,
                        default=ROOT / "data/outputs/local-fusion-20260929")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/outputs/head-eye-experts-20260930/production-evaluation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy_path = output / "policy.json"
    policy = dict(EYE_EXPERT_POLICY)
    if policy_path.exists() and json.loads(policy_path.read_text("utf-8")) != policy:
        raise ValueError("Frozen policy differs from runtime; use a new experiment directory")
    if args.role == "blind_test" and not policy_path.exists():
        raise ValueError("Run validation and freeze the policy before held-out evaluation")
    baseline.write(policy_path, policy)
    if args.role == "blind_test":
        baseline.write(output / "test-opened.json", dict(policy_sha256=digest(policy_path)))
    baseline.RUN = args.baseline_dir.resolve()
    data, info = baseline.prepare("HEAD", args.role, torch.device(args.device))
    evidence = features(data, info, baseline.RUN, output, args.role, torch.device(args.device))
    scores = fused_scores(data["base"], data["local"], data["gates"][:, 1],
                          weight=.5, rrf_k=10, view_fusion="max")
    extra = eye_expert_scores(evidence["local"], evidence["query_profiles"],
                              evidence["candidate_profiles"], data["gates"][:, 1], **policy)
    before, after = (ranks_of_truth(s, data["truth"]) for s in (scores, scores + extra))
    masks = dict(all=np.ones(len(before), dtype=bool), eyewear_proxy=data["group"])
    for fold in sorted({c["fold"] for c in info["cases"]}):
        masks[f"fold-{fold}"] = np.array([c["fold"] == fold for c in info["cases"]])
    summary = {name: dict(before=baseline.metrics(before[mask]),
                          after=baseline.metrics(after[mask], before[mask]))
               for name, mask in masks.items()}
    result = dict(
        policy=policy, policy_sha256=digest(policy_path), role=args.role,
        scope="Frozen identity retrieval plus local experts; excludes production morphology, legacy head-detail, no-equipment and cross-slot rules. This is an existing held-out set, not newly collected blind data.",
        baseline_sha256=digest(baseline.RUN / f"HEAD-{args.role}.npz"),
        summary=summary,
        cases=[dict(c, before=int(b), after=int(a), expert_active=bool(np.any(extra[i] > 0)))
               for i, (c, b, a) in enumerate(zip(info["cases"], before, after, strict=True))],
    )
    baseline.write(output / f"{args.role}.json", result)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
