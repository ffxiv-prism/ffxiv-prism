from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import write_json  # noqa: E402
from ffxiv_prism.paths import portable_path  # noqa: E402


SLOTS = ("BODY", "HEAD", "HANDS", "LEGS", "FEET")
CUTOFFS = (1, 10, 100, 1000)
DEFAULT_OUTPUT = Path("data/processed/current_recognition_metrics.json")
DEFAULT_SELECTION_OUTPUT = Path("data/current_best_selection.json")
DEFAULT_FOLD_AUDIT = Path(
    "data/outputs/slot-quality-fold-audit-20260801/summary.json"
)
METHODS = {
    "BODY:safe-gate": (
        "DINOv2 adapted query plus zero-shot/adapted DINOv3 with frozen "
        "Top-1 agreement gate"
    ),
    "BODY:baseline": (
        "DINOv2 adapted query plus zero-shot DINOv3 inside the frozen "
        "DINOv2 Top-1000 pool; unsafe adapted DINOv3 candidate excluded"
    ),
    "HEAD": (
        "DINOv2, SigLIP2 and adapted DINOv3 rank ensemble with "
        "upper-to-face routing"
    ),
    "HANDS": (
        "DINOv2 adapted query, multi-crop/gender routing and occupied-slot "
        "Top-1 gate"
    ),
    "LEGS": (
        "DINOv2 plus DINOv3 rank ensemble inside a frozen DINOv2 "
        "Top-1000 pool"
    ),
    "FEET": "clean primary-family candidate index with DINOv2 and SigLIP2 rank ensemble",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register a strict three-fold current-best metric snapshot from "
            "frozen per-fold release reports."
        )
    )
    parser.add_argument("--candidate-suffix", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--fold-audit", type=Path, default=DEFAULT_FOLD_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=DEFAULT_SELECTION_OUTPUT,
        help="Write the inference selection manifest used by current-best.",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        required=True,
        help="Training checkpoint prefix, for example refresh_v110.",
    )
    parser.add_argument("--dinov2-epochs", type=int, default=10)
    parser.add_argument("--body-dinov3-epochs", type=int, default=10)
    parser.add_argument("--head-dinov3-epochs", type=int, default=3)
    parser.add_argument("--target-top10", type=float, default=0.8)
    parser.add_argument(
        "--body-policy",
        choices=("safe-gate", "baseline"),
        default="safe-gate",
        help=(
            "Select the frozen BODY policy represented by the compact gate "
            "reports. Use baseline when any fold rejects the adapted branch."
        ),
    )
    for slot in SLOTS:
        parser.add_argument(
            f"--{slot.lower()}-report",
            action="append",
            type=Path,
            required=True,
            help=f"Repeat exactly three times for {slot}, once per frozen fold.",
        )
    return parser.parse_args()


def selected_summary(
    slot: str,
    payload: Mapping[str, Any],
    *,
    body_policy: str = "safe-gate",
) -> Mapping[str, Any]:
    if slot == "BODY":
        if body_policy == "baseline":
            summary = payload.get("baseline_summary")
        else:
            selected = payload.get("best_safe_policy")
            if not isinstance(selected, Mapping):
                raise ValueError("BODY report has no domain-safe frozen gate policy")
            expected = {
                "max_base_rank": 10,
                "max_last_branch_rank": 1,
                "min_last_branch_rank_ratio": 1.0,
                "max_base_last_branch_rank": 10,
            }
            if dict(selected.get("policy") or {}) != expected:
                raise ValueError(
                    "BODY report does not use the frozen production gate policy"
                )
            if selected.get("domain_regressions"):
                raise ValueError("BODY frozen gate has domain regressions")
            summary = selected.get("summary")
    elif slot == "HANDS":
        selected = payload.get("best_safe_gate")
        if not isinstance(selected, Mapping):
            raise ValueError("HANDS report has no domain-safe occupied-slot gate")
        if float(selected.get("occupied_multiplier", -1)) != 0.25:
            raise ValueError("HANDS report does not use occupied multiplier 0.25")
        if int(selected.get("max_base_rank", -1)) != 3:
            raise ValueError("HANDS report does not use max base rank 3")
        if selected.get("domain_regressions") or selected.get("cutoff_regressions"):
            raise ValueError("HANDS frozen gate has regressions")
        summary = selected.get("summary")
    else:
        summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"{slot} report has no selected summary")
    return summary


def pool_reports(
    slot: str,
    paths: Sequence[Path],
    *,
    expected_eligible: int,
    candidate_suffix: str,
    body_policy: str = "safe-gate",
) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError(f"{slot} requires exactly three fold reports")
    folds: set[int] = set()
    hits = {str(cutoff): 0 for cutoff in CUTOFFS}
    eligible = 0
    evidence = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("slot")) != slot:
            raise ValueError(f"Slot mismatch in {path}: {payload.get('slot')}")
        fold = int(payload.get("fold", 0))
        if fold not in {1, 2, 3} or fold in folds:
            raise ValueError(f"Invalid or duplicate fold in {path}: {fold}")
        folds.add(fold)
        serialized = json.dumps(payload, ensure_ascii=False)
        if candidate_suffix not in serialized:
            raise ValueError(
                f"{path} does not reference candidate suffix {candidate_suffix}"
            )
        summary = selected_summary(slot, payload, body_policy=body_policy)
        fold_eligible = int(summary["eligible_case_count"])
        eligible += fold_eligible
        fold_hits = summary.get("hits") or {}
        for cutoff in CUTOFFS:
            hits[str(cutoff)] += int(fold_hits[str(cutoff)])
        evidence.append(
            {
                "fold": fold,
                "path": portable_path(path),
                "eligible": fold_eligible,
                "hits": {
                    str(cutoff): int(fold_hits[str(cutoff)])
                    for cutoff in CUTOFFS
                },
            }
        )
    if folds != {1, 2, 3}:
        raise ValueError(f"{slot} reports do not cover all three folds")
    if eligible != expected_eligible:
        raise ValueError(
            f"{slot} pooled denominator mismatch: reports={eligible}, "
            f"audit={expected_eligible}"
        )
    return {
        "slot": slot,
        "eligible": eligible,
        "hits": hits,
        "top1": hits["1"] / eligible,
        "top10": hits["10"] / eligible,
        "top100": hits["100"] / eligible,
        "top1000": hits["1000"] / eligible,
        "method": METHODS[f"BODY:{body_policy}"] if slot == "BODY" else METHODS[slot],
        "fold_reports": sorted(evidence, key=lambda row: row["fold"]),
    }


def build_current_best_selection(
    *,
    release_id: str,
    candidate_suffix: str,
    body_policy: str,
    checkpoint_prefix: str,
    slots: Sequence[Mapping[str, Any]],
    dinov2_epochs: int = 10,
    body_dinov3_epochs: int = 10,
    head_dinov3_epochs: int = 3,
) -> dict[str, Any]:
    prefix = checkpoint_prefix.strip()
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("checkpoint-prefix must be a non-empty filename prefix")
    metrics = {str(row["slot"]): row for row in slots}
    missing = sorted(set(SLOTS) - set(metrics))
    if missing:
        raise ValueError("Missing selection metrics for slots: " + ", ".join(missing))

    def checkpoint(slot: str, model: str, epochs: int) -> str:
        return f"{prefix}_{slot.lower()}_fold01_{model}_e{epochs}.pt"

    return {
        "version": 1,
        "release_id": release_id,
        "candidate_suffix": candidate_suffix,
        "body_policy": body_policy,
        "checkpoints": {
            "body_dinov2": checkpoint("body", "dinov2", dinov2_epochs),
            "body_dinov3": (
                checkpoint("body", "dinov3", body_dinov3_epochs)
                if body_policy == "safe-gate"
                else None
            ),
            "head_dinov2": checkpoint("head", "dinov2", dinov2_epochs),
            "head_dinov3": checkpoint("head", "dinov3", head_dinov3_epochs),
            "hands_dinov2": checkpoint("hands", "dinov2", dinov2_epochs),
            "legs_dinov2": checkpoint("legs", "dinov2", dinov2_epochs),
            "feet_dinov2": checkpoint("feet", "dinov2", dinov2_epochs),
        },
        "benchmarks": {
            slot: {
                "eligible": int(metrics[slot]["eligible"]),
                "top1": float(metrics[slot]["top1"]),
                "top10": float(metrics[slot]["top10"]),
            }
            for slot in SLOTS
        },
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.target_top10 <= 1:
        raise ValueError("target-top10 must be within (0, 1]")
    fold_audit = json.loads(args.fold_audit.read_text(encoding="utf-8"))
    if fold_audit.get("status") != "ready_for_training":
        raise ValueError("Fold audit is not ready_for_training")
    if not all((fold_audit.get("slot_gates") or {}).get(slot) for slot in SLOTS):
        raise ValueError("At least one slot failed the frozen fold audit")
    audited_folds = fold_audit.get("folds") or {}
    slots = []
    for slot in SLOTS:
        paths = list(getattr(args, f"{slot.lower()}_report"))
        expected = int(audited_folds[slot]["pooled_validation_images"])
        slots.append(
            pool_reports(
                slot,
                paths,
                expected_eligible=expected,
                candidate_suffix=args.candidate_suffix,
                body_policy=args.body_policy,
            )
        )

    target_achieved = all(row["top10"] >= args.target_top10 for row in slots)
    payload = {
        "version": 2,
        "release_id": args.release_id,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "metric": "appearance_recall",
        "protocol": "strict item-disjoint three-fold pooled",
        "metric_status": "release_ready",
        "candidate_snapshot": args.candidate_suffix,
        "fold_audit": portable_path(args.fold_audit),
        "fold_audit_status": fold_audit.get("status"),
        "target_top10": args.target_top10,
        "target_achieved_all_slots": target_achieved,
        "release_policies": {
            "BODY": args.body_policy,
            "HANDS": "occupied-slot-top1-gate",
        },
        "slots": slots,
        "canonical_training_assets": {
            "examples": "data/processed/glamour_examples_huiji_models_closed.jsonl",
            "slot_eligibility": "data/processed/slot_quality_eligibility_20260801.jsonl",
            "visibility_reviews": "data/processed/training_visibility_reviews_20260801.jsonl",
            "regions": "data/debug/outfit-regions/slot-quality-20260801",
            "folds": "data/interim/appearance-disjoint-cv-v55-slot-recovery-{slot}",
        },
    }
    write_json(args.output, payload)
    selection = build_current_best_selection(
        release_id=args.release_id,
        candidate_suffix=args.candidate_suffix,
        body_policy=args.body_policy,
        checkpoint_prefix=args.checkpoint_prefix,
        slots=slots,
        dinov2_epochs=args.dinov2_epochs,
        body_dinov3_epochs=args.body_dinov3_epochs,
        head_dinov3_epochs=args.head_dinov3_epochs,
    )
    write_json(args.selection_output, selection)
    print(
        json.dumps(
            {
                "output": portable_path(args.output),
                "selection_output": portable_path(args.selection_output),
                "metric_status": payload["metric_status"],
                "target_achieved_all_slots": target_achieved,
                "slots": [
                    {
                        "slot": row["slot"],
                        "eligible": row["eligible"],
                        "top1": row["top1"],
                        "top10": row["top10"],
                    }
                    for row in slots
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
