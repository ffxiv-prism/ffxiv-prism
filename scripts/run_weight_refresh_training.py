"""Resume audited fold-specific query models against a frozen active catalogue.

This stage creates challenger weights only. Evaluation and promotion are
separate; the active inference selection is never changed by training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-audit", type=Path, required=True)
    parser.add_argument("--prefix", default="refresh_20260926")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.prefix.strip() or Path(args.prefix).name != args.prefix or any(c in args.prefix for c in '/\\:'):
        raise ValueError("prefix must be a non-empty filename prefix")
    os.chdir(ROOT)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads(args.fold_audit.read_text("utf-8"))
    if audit.get("status") != "ready_for_training" or not all(audit.get("slot_gates", {}).get(s) for s in ("HEAD", "BODY", "HANDS", "LEGS", "FEET")):
        raise ValueError("Training requires a passed five-slot fold audit")
    plan_path = output / "training-plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text("utf-8"))
    else:
        import numpy as np
        np.linalg.inv(np.eye(2, dtype=np.float32))
        import torch
        from ffxiv_prism.matching.inference import current_best_configs

        selection_path = ROOT / "data/current_best_selection.json"
        selection = json.loads(selection_path.read_text("utf-8"))
        configs = current_best_configs()
        jobs = []
        sources = {
            "data/current_best_selection.json": sha256(selection_path),
            str(args.fold_audit): sha256(args.fold_audit),
        }
        for path in (
            "data/processed/glamour_examples_huiji_models_closed.jsonl",
            "data/processed/slot_quality_eligibility_20260801.jsonl",
            "data/processed/training_visibility_reviews_20260801.jsonl",
        ):
            sources[path] = sha256(ROOT / path)
        for slot, branch in (("HEAD", "dinov2"), ("HEAD", "dinov3"), ("BODY", "dinov2"), ("HANDS", "dinov2"), ("LEGS", "dinov2"), ("FEET", "dinov2")):
            config = configs[slot]
            index = config.index_dir if branch == "dinov2" else config.ensemble_index_dirs[1]
            index_relative = index.relative_to(ROOT)
            for filename in ("metadata.jsonl", "embeddings.npy", "summary.json"):
                path = index_relative / filename
                sources[path.as_posix()] = sha256(ROOT / path)
            key = f"{slot.lower()}_{branch}"
            for fold in (1, 2, 3):
                parent = Path("data/models") / selection["checkpoints"][key].replace("fold01", f"fold{fold:02}")
                payload = torch.load(parent, map_location="cpu", weights_only=True)
                parent_epoch = int(payload["epochs_completed"])
                epochs = 1 if branch == "dinov3" else 3
                checkpoint = Path("data/models") / f"{args.prefix}_{slot.lower()}_fold{fold:02}_{branch}_e{parent_epoch + epochs}.pt"
                fold_path = Path(f"data/interim/appearance-disjoint-cv-v55-slot-recovery-{slot.lower()}/fold-{fold:02}.json")
                sources[str(parent)] = sha256(parent)
                sources[str(fold_path)] = sha256(fold_path)
                name = f"{slot.lower()}-fold{fold:02}-{branch}"
                command = [
                    sys.executable, "scripts/train_glamour_query_encoder.py",
                    "--model-id", str(payload["model_id"]), "--index-dir", str(index_relative),
                    "--examples", "data/processed/glamour_examples_huiji_models_closed.jsonl",
                    "--slot-eligibility-manifest", "data/processed/slot_quality_eligibility_20260801.jsonl",
                    "--vision-reviews", "data/processed/training_visibility_reviews_20260801.jsonl",
                    "--regions-dir", "data/debug/outfit-regions/slot-quality-20260801",
                    "--slots", slot, "--split-manifest", str(fold_path),
                    "--resume-checkpoint", str(parent), "--checkpoint", str(checkpoint),
                    "--output", str(output / f"train-{name}.json"),
                    "--device", args.device, "--epochs", str(epochs), "--fixed-epochs",
                    "--unfreeze-blocks", str(payload["unfreeze_blocks"]),
                    "--seed", str(payload["seed"]), "--batch-size", "16",
                    "--alternate-probability", "0", "--skip-evaluation",
                ]
                jobs.append(dict(name=name, slot=slot, fold=fold, branch=branch, index=str(index_relative),
                                 parent=str(parent), checkpoint=str(checkpoint), parent_epoch=parent_epoch,
                                 additional_epochs=epochs, command=command))
        plan = dict(version=1, created_at=datetime.now(timezone.utc).isoformat(), selection=selection,
                    fold_audit=str(args.fold_audit),
                    sources=sources, jobs=jobs,
                    policy="fixed continuation: DINOv2 +3 epochs; HEAD DINOv3 +1; preserve optimizer; no validation-driven epoch search")
        write_json(plan_path, plan)
    for path, expected in plan["sources"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Frozen training input changed: {path}")
    print(f"plan={plan_path} jobs={len(plan['jobs'])}", flush=True)
    if not args.execute:
        return 0
    status_path = output / "training-status.json"
    status = json.loads(status_path.read_text("utf-8")) if status_path.exists() else dict(status="running", jobs={})
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    for job in plan["jobs"]:
        previous = status["jobs"].get(job["name"], {})
        checkpoint = ROOT / job["checkpoint"]
        if previous.get("status") == "passed" and checkpoint.is_file() and sha256(checkpoint) == previous.get("sha256"):
            continue
        status.update(status="running", active_job=job["name"])
        status["jobs"][job["name"]] = dict(status="running", started_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        print(f"training={job['name']}", flush=True)
        with (output / f"train-{job['name']}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(job["command"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record = status["jobs"][job["name"]]
        record.update(status="passed" if result.returncode == 0 else "failed", returncode=result.returncode,
                      finished_at=datetime.now(timezone.utc).isoformat())
        if result.returncode == 0:
            record["sha256"] = sha256(checkpoint)
        else:
            status["status"] = "failed"
        write_json(status_path, status)
        if result.returncode:
            return result.returncode
    status.update(status="completed", active_job=None, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
