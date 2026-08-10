from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.evaluate_feet_visibility_gate import (  # noqa: E402
    semantic_feet_features,
    visibility_rule,
)
from ffxiv_prism.jsonl import (  # noqa: E402
    read_jsonl,
    write_json,
    write_jsonl,
)
from ffxiv_prism.paths import portable_path, resolve_project_path  # noqa: E402
from ffxiv_prism.vision.human_parsing import HumanParser  # noqa: E402


def has_feet_label(example: Mapping[str, Any]) -> bool:
    return bool(
        (((example.get("core_slot_labels") or {}).get("FEET") or {}).get(
            "piece_ids"
        ))
    )


def triage_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    features = row["features"]
    return (
        float(features["shoe_share_person"]),
        -float(features["lower_garment_bottom_band_share"]),
        -float(features["lower_garment_bottom_ratio"]),
        float(features["person_bottom_edge_gap_ratio"]),
        str(row["example_id"]),
    )


def load_thresholds(path: Path) -> tuple[dict[str, float], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    triage = payload.get("high_recall_triage") or {}
    thresholds = triage.get("thresholds") or payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("diagnostic does not contain visibility thresholds")
    required = {
        "min_lower_bottom_ratio",
        "min_bottom_band_share",
        "max_shoe_share",
        "max_edge_gap_ratio",
    }
    missing = sorted(required - set(thresholds))
    if missing:
        raise ValueError(f"visibility thresholds missing keys: {missing}")
    return (
        {key: float(thresholds[key]) for key in sorted(required)},
        (
            "high_recall_triage.thresholds"
            if triage.get("thresholds")
            else "thresholds"
        ),
    )


def parse_features(
    parser_model: HumanParser,
    examples: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        images = []
        for example in batch:
            image_path = resolve_project_path(example.get("local_path"))
            if image_path is None or not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as source:
                images.append(source.convert("RGB"))
        segmentations = parser_model.parse_native(images)
        for example, segmentation in zip(batch, segmentations, strict=True):
            rows.append(
                {
                    "example_id": str(example["example_id"]),
                    "glamour_id": str(example["glamour_id"]),
                    "local_path": str(example["local_path"]),
                    "item_name": str(
                        ((example.get("core_slot_labels") or {}).get("FEET")
                        or {}).get("item_name")
                        or ""
                    ),
                    "features": semantic_feet_features(segmentation),
                }
            )
        for image in images:
            image.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a human-review priority queue for semantically hidden "
            "FEET labels. This tool never makes automatic training decisions."
        )
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path(
            "data/processed/glamour_examples_huiji_models_closed.jsonl"
        ),
    )
    parser.add_argument(
        "--quality-manifest",
        type=Path,
        default=Path("data/processed/front_fullbody_quality_final_v46.jsonl"),
    )
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--features-output", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    passed_ids = {
        str(row["example_id"])
        for row in read_jsonl(args.quality_manifest)
        if row.get("passed")
    }
    examples = [
        row
        for row in read_jsonl(args.examples)
        if str(row.get("example_id")) in passed_ids and has_feet_label(row)
    ]
    thresholds, threshold_source = load_thresholds(args.diagnostic)
    parser_model = HumanParser(device=args.device)
    rows = parse_features(
        parser_model,
        examples,
        batch_size=args.batch_size,
    )
    for row in rows:
        row["triage_for_review"] = visibility_rule(
            row["features"],
            **thresholds,
        )
    queue = sorted(
        [row for row in rows if row["triage_for_review"]],
        key=triage_sort_key,
    )
    for rank, row in enumerate(queue, start=1):
        row["priority_rank"] = rank
        row["slot"] = "FEET"
        row["visibility"] = None
        row["usability"] = None
        row["issue"] = None
        row["review_source"] = "manual_parser_triage"

    write_jsonl(args.queue_output, queue)
    if args.features_output is not None:
        write_jsonl(args.features_output, rows)
    summary = {
        "version": 1,
        "diagnostic": portable_path(args.diagnostic),
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "parser_model": parser_model.model_id,
        "eligible_images": len(rows),
        "eligible_glamours": len({row["glamour_id"] for row in rows}),
        "queued_images": len(queue),
        "queued_glamours": len({row["glamour_id"] for row in queue}),
        "queue_rate": len(queue) / len(rows) if rows else 0.0,
        "queue_output": portable_path(args.queue_output),
        "features_output": portable_path(args.features_output),
        "policy": (
            "Human review priority only; predictions must not filter training "
            "rows without a manual decision."
        ),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
