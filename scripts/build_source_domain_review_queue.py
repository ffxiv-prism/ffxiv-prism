from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402
from ffxiv_prism.paths import portable_path, resolve_project_path  # noqa: E402
from ffxiv_prism.torch_device import resolve_torch_device  # noqa: E402


DEFAULT_MODEL = "openai/clip-vit-base-patch32"
PROMPT_GROUPS = {
    "ffxiv_game": (
        "a screenshot of a character in Final Fantasy XIV",
        "a 3D MMORPG character screenshot",
        "a screenshot of a fantasy video game character",
    ),
    "illustration": (
        "a 2D cartoon drawing of a person",
        "a digital illustration or character reference art",
        "an animated cartoon character",
    ),
    "real_photo": (
        "a photograph of a real human wearing clothing",
        "a cosplay photograph of a real person",
    ),
    "non_character": (
        "a clothing catalog image without a video game character",
        "a webpage, user interface, or text document",
    ),
}


def domain_risk(probabilities: Mapping[str, float]) -> float:
    return 1.0 - float(probabilities.get("ffxiv_game", 0.0))


def domain_sort_key(row: Mapping[str, Any]) -> tuple[float, str]:
    return -float(row["non_game_risk"]), str(row["example_id"])


def score_domains(
    paths: Sequence[Path],
    *,
    model_id: str,
    device: str,
    batch_size: int,
) -> list[dict[str, float]]:
    from transformers import CLIPModel, CLIPProcessor

    resolved_device = resolve_torch_device(device)
    cache_dir = PROJECT_ROOT / "var" / "cache" / "huggingface" / "hub"
    processor = CLIPProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model = CLIPModel.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
    ).to(resolved_device).eval()
    labels = tuple(PROMPT_GROUPS)
    prompts = [prompt for group in PROMPT_GROUPS.values() for prompt in group]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    text_inputs = {
        key: value.to(resolved_device) for key, value in text_inputs.items()
    }
    with torch.inference_mode():
        text_outputs = model.text_model(**text_inputs)
        text_features = model.text_projection(text_outputs.pooler_output)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        prototypes = []
        start = 0
        for group in PROMPT_GROUPS.values():
            end = start + len(group)
            prototype = text_features[start:end].mean(dim=0)
            prototypes.append(torch.nn.functional.normalize(prototype, dim=-1))
            start = end
        text_features = torch.stack(prototypes)

    output = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        for image in images:
            image.close()
        pixel_values = inputs["pixel_values"].to(resolved_device)
        with torch.inference_mode():
            image_outputs = model.vision_model(pixel_values=pixel_values)
            image_features = model.visual_projection(image_outputs.pooler_output)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            probabilities = torch.softmax(
                100.0 * image_features @ text_features.T,
                dim=-1,
            ).float().cpu().numpy()
        output.extend(
            {
                label: float(value)
                for label, value in zip(labels, row, strict=True)
            }
            for row in np.asarray(probabilities)
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rank quality-passed player images for manual source-domain "
            "review. Scores are triage-only and never auto-reject samples."
        )
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("data/processed/glamour_examples_huiji_models_closed.jsonl"),
    )
    parser.add_argument(
        "--quality-manifest",
        type=Path,
        default=Path("data/processed/front_fullbody_quality_final_v46.jsonl"),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--queue-size", type=int, default=100)
    parser.add_argument(
        "--known-invalid-example-id",
        action="append",
        default=[],
        help="Optional manually known invalid example used only to report rank calibration.",
    )
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.queue_size < 1:
        parser.error("--batch-size and --queue-size must be positive")

    passed_ids = {
        str(row["example_id"])
        for row in read_jsonl(args.quality_manifest)
        if row.get("passed")
    }
    examples = [
        row
        for row in read_jsonl(args.examples)
        if str(row.get("example_id")) in passed_ids
    ]
    paths = []
    for row in examples:
        path = resolve_project_path(row.get("local_path"))
        if path is None or not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    probabilities = score_domains(
        paths,
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
    )
    rows = []
    for example, probability in zip(examples, probabilities, strict=True):
        rows.append(
            {
                "example_id": str(example["example_id"]),
                "glamour_id": str(example["glamour_id"]),
                "local_path": str(example["local_path"]),
                "title": str(example.get("title") or ""),
                "probabilities": probability,
                "non_game_risk": domain_risk(probability),
            }
        )
    ranked = sorted(rows, key=domain_sort_key)
    queue = []
    for rank, row in enumerate(ranked[: min(args.queue_size, len(ranked))], start=1):
        queue.append(
            {
                **row,
                "priority_rank": rank,
                "decision": None,
                "issue": None,
                "review_source": "manual_source_domain_triage",
            }
        )
    write_jsonl(args.scores_output, rows)
    write_jsonl(args.queue_output, queue)
    summary = {
        "version": 1,
        "model_id": args.model_id,
        "eligible_images": len(rows),
        "queued_images": len(queue),
        "known_invalid_ranks": {
            example_id: next(
                (
                    index
                    for index, row in enumerate(ranked, start=1)
                    if row["example_id"] == example_id
                ),
                None,
            )
            for example_id in args.known_invalid_example_id
        },
        "scores_output": portable_path(args.scores_output),
        "queue_output": portable_path(args.queue_output),
        "policy": (
            "Manual source-domain review priority only; CLIP scores must not "
            "automatically reject player images."
        ),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
