"""Register independently selected branches from a completed audited refresh."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.weight_refresh_metrics import compare_variants  # noqa: E402
from scripts.register_current_recognition_metrics import METHODS, SLOTS  # noqa: E402
from scripts.run_weight_refresh_training import sha256, write_json  # noqa: E402


def selected_checkpoint_keys(slot: str, variant: str) -> set[str]:
    if variant == "baseline":
        return set()
    if slot == "HEAD":
        choices = {
            "continued": {"head_dinov2", "head_dinov3"},
            "dinov2_only": {"head_dinov2"},
            "dinov3_only": {"head_dinov3"},
        }
        return choices[variant]
    if variant != "continued":
        raise ValueError(f"Unknown {slot} variant: {variant}")
    return {f"{slot.lower()}_dinov2"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--candidate-suffix", required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = json.loads((args.run_dir / "training-plan.json").read_text("utf-8"))
    status = json.loads((args.run_dir / "training-status.json").read_text("utf-8"))
    if status.get("status") != "completed":
        raise ValueError("Training has not completed")
    for path, digest in plan["sources"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    for job in plan["jobs"]:
        result = status["jobs"][job["name"]]
        if result["status"] != "passed" or sha256(ROOT / job["checkpoint"]) != result["sha256"]:
            raise ValueError(f"Unverified training job: {job['name']}")
    audit_path = Path(plan.get("fold_audit", args.run_dir / "fold-audit.json"))
    if str(audit_path) not in plan["sources"]:
        raise ValueError("Fold audit is not part of the frozen training sources")
    audit = json.loads(audit_path.read_text("utf-8"))
    if audit.get("status") != "ready_for_training" or not all(audit.get("slot_gates", {}).get(s) for s in SLOTS):
        raise ValueError("All five slot audits must pass")
    selection = copy.deepcopy(plan["selection"])
    if not all(p.endswith(args.candidate_suffix) for p in selection["candidate_indexes"].values()):
        raise ValueError("Candidate suffix does not match the frozen indexes")
    rows, comparisons = [], {}
    promoted = set()
    for slot in SLOTS:
        path = args.run_dir / f"comparison-{slot.lower()}.json"
        report = json.loads(path.read_text("utf-8"))
        if report["slot"] != slot or report["candidate_indexes"] != selection["candidate_indexes"]:
            raise ValueError(f"Evaluation snapshot mismatch: {slot}")
        cases = report["cases"]
        expected = int(audit["folds"][slot]["pooled_validation_images"])
        if len(cases) != expected or len({c["example_id"] for c in cases}) != expected or {c["fold"] for c in cases} != {1, 2, 3}:
            raise ValueError(f"Evaluation coverage mismatch: {slot}")
        verified = compare_variants(cases, list(report["variants"]))
        if any(report[k] != v for k, v in verified.items()):
            raise ValueError(f"Evaluation decision or metrics changed: {slot}")
        selected = verified["selected"]
        promoted.update(selected_checkpoint_keys(slot, selected))
        metric = verified["groups"]["all"][selected]
        rows.append(dict(slot=slot, eligible=expected, hits=metric["hits"],
                         **{f"top{k}": metric["recall"][str(k)] for k in (1, 10, 100, 1000)},
                         method=METHODS["BODY:baseline" if slot == "BODY" else slot],
                         fold_reports=[dict(fold=f, eligible=verified["groups"][f"fold:{f}"][selected]["eligible_case_count"],
                                            hits=verified["groups"][f"fold:{f}"][selected]["hits"], path=path.as_posix()) for f in (1, 2, 3)]))
        comparisons[slot] = dict(**verified, scope=report["scope"], comparison_sha256=sha256(path))
    for job in plan["jobs"]:
        key = f"{job['slot'].lower()}_{job['branch']}"
        if key in promoted and job["fold"] == 1:
            selection["checkpoints"][key] = Path(job["checkpoint"]).name
    scope = ("strict appearance-disjoint three-fold pooled validation; official-equivalence "
             "deduplicated identity retrieval; production style/presence/local-detail heuristics excluded; "
             "model selection on existing validation folds, not an independent test")
    selection.update(release_id=args.release_id, candidate_suffix=args.candidate_suffix,
                     benchmark_scope=scope,
                     benchmarks={r["slot"]: {k: r[k] for k in ("eligible", "top1", "top10")} for r in rows})
    evidence = dict(version=1, release_id=args.release_id, baseline_selection=plan["selection"],
                    candidate_snapshot=args.candidate_suffix, scope=scope,
                    promotion_rule="Top1 or Top10 pooled gain; no Top1/Top10 loss in any fold, race, or gender",
                    promoted_branches=sorted(promoted), comparisons=comparisons,
                    training_policy=plan["policy"], training_sources=plan["sources"],
                    training_checkpoints={j["name"]: dict(path=j["checkpoint"], sha256=status["jobs"][j["name"]]["sha256"]) for j in plan["jobs"]})
    metrics = dict(version=3, release_id=args.release_id, registered_at=datetime.now(timezone.utc).isoformat(),
                   metric="appearance_recall", protocol=scope, metric_status="release_ready",
                   candidate_snapshot=args.candidate_suffix, fold_audit=audit_path.as_posix(),
                   fold_audit_status="ready_for_training", target_top10=0.8,
                   target_achieved_all_slots=all(r["top10"] >= 0.8 for r in rows),
                   release_policies={"BODY": "baseline", "HANDS": "occupied-slot-top1-gate"},
                   weight_refresh_evidence=args.evidence_output.as_posix(), slots=rows)
    write_json(args.evidence_output, evidence)
    write_json(ROOT / "data/processed/current_recognition_metrics.json", metrics)
    write_json(ROOT / "data/current_best_selection.json", selection)
    print(json.dumps(dict(promoted_branches=sorted(promoted), slots=selection["benchmarks"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
