"""Apply the production presence gate to a frozen detail-recall audit.

The retrieval report retains the rejected ungated proposal as evidence.
This stage only disables proposals; it never changes their identity or rank.
"""
# ruff: noqa: E402
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
np.linalg.inv(np.eye(2, dtype=np.float32))
import torch
from ffxiv_prism.cli.match import _classify_style_paths
from ffxiv_prism.matching.head_detail import HEAD_TYPE_PROMPTS, DETAIL_NAMES, detail_presence_decision
from ffxiv_prism.model_assets import model_weight_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.audit_dir
    report_path = root / "evaluation.json"
    report = json.loads(report_path.read_text("utf-8"))
    paths = sorted((root / "crops").glob("*.jpg"))
    model_id = "google/siglip2-base-patch16-224"
    digest = hashlib.sha256(report_path.read_bytes())
    digest.update(json.dumps(HEAD_TYPE_PROMPTS).encode())
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    for path in sorted(model_weight_files(model_id)):
        digest.update(str(path).encode())
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
        for config in sorted(path.parent.glob("*.json")):
            digest.update(config.read_bytes())
    contract = digest.hexdigest()
    cache = root / "presence-cache.json"
    if cache.is_file():
        payload = json.loads(cache.read_text("utf-8"))
        if payload["contract"] != contract:
            raise ValueError("Presence inputs changed; use a fresh audit directory")
        profiles = payload["profiles"]
    else:
        values = _classify_style_paths(
            model_id, HEAD_TYPE_PROMPTS, paths,
            device=torch.device(args.device), batch_size=32,
        )
        profiles = {str(p): v.tolist() for p, v in zip(paths, values, strict=True)}
        cache.write_text(json.dumps(dict(contract=contract, profiles=profiles)), "utf-8")
    groups = defaultdict(list)
    for case in report["cases"]:
        views = {
            name: profiles[str(root / "crops" / (case["example_id"].replace(":", "-") + "-" + name + ".jpg"))]
            for name in DETAIL_NAMES
        }
        case["presence"] = detail_presence_decision(views)
        case["guarded_after"] = case["after"] if case["presence"]["active"] else case["before"]
        for group in ("all", f"fold:{case['fold']}", f"gender:{case['gender']}", f"race:{case['race']}"):
            groups[group].append(case)
    summary = {}
    for group, rows in groups.items():
        summary[group] = dict(
            count=len(rows), activated=sum(r["presence"]["active"] for r in rows),
            changed=sum(r["presence"]["active"] and r["diagnostic"]["ranking_changed"] for r in rows),
            hits={str(k): {phase: sum(r[phase] is not None and r[phase] <= k for r in rows) for phase in ("before", "guarded_after")} for k in (1, 5, 10, 20, 100, 1000)},
        )
    payload = dict(contract=contract, summary=summary, cases=report["cases"])
    (root / "presence-evaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
