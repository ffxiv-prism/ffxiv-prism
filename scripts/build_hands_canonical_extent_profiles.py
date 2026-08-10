from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402
from ffxiv_prism.paths import CACHE_DIR, portable_path  # noqa: E402


CANONICAL_CROP = "both_hands"
ARM_LABELS = (14, 15)
# On a HANDS-only canonical render, non-skin person labels near an arm are a
# useful equipment signal even when metal is misclassified as shoes or a bag.
NON_EQUIPMENT_LABELS = frozenset((0, 2, 11, 14, 15))
EXTENT_CLASSES = ("short", "medium", "long")
SHORT_NAME_PRIOR = re.compile(
    r"\b(nails?|bracelets?|armbands?|wristbands?|chaplets?|runes?|"
    r"halfgloves?|shortgloves?)\b",
    re.IGNORECASE,
)
LONG_NAME_PRIOR = re.compile(
    r"\b(gauntlets?|vambraces?|armguards?|armlets?|kote|tekko|sleeves?)\b",
    re.IGNORECASE,
)
CLIP_PROMPTS = (
    "a bare forearm with only a tiny wrist bracelet or ring jewelry",
    "a hand wearing a short glove ending at the wrist",
    "a forearm covered from wrist toward elbow by a long sleeve or armored gauntlet",
)
ARM_CONTEXT_BOXES = (
    (0.0, 0.08, 0.58, 1.0),
    (0.42, 0.08, 1.0, 1.0),
)


def parser_extent_features(
    segmentation: np.ndarray,
    *,
    dilation: int = 61,
    minimum_arm_fraction: float = 0.02,
) -> tuple[dict[str, Any], np.ndarray]:
    if segmentation.ndim != 2:
        raise ValueError("segmentation must be a two-dimensional label map")
    if dilation < 1:
        raise ValueError("dilation must be positive")
    dilation = dilation if dilation % 2 == 1 else dilation + 1
    if not 0 <= minimum_arm_fraction <= 1:
        raise ValueError("minimum_arm_fraction must be in [0, 1]")

    person = segmentation != 0
    gear = person & ~np.isin(segmentation, tuple(NON_EQUIPMENT_LABELS))
    selected = np.zeros_like(gear, dtype=bool)
    side_rows = []
    for label in ARM_LABELS:
        arm = segmentation == label
        arm_fraction = float(arm.mean())
        if arm_fraction < minimum_arm_fraction:
            continue
        dilated = (
            np.asarray(
                Image.fromarray(arm.astype(np.uint8) * 255, mode="L").filter(
                    ImageFilter.MaxFilter(dilation)
                )
            )
            > 0
        )
        near_gear = gear & dilated
        support = arm | near_gear
        share = float(near_gear.sum() / max(int(support.sum()), 1))
        selected |= near_gear
        side_rows.append(
            {
                "arm_label": label,
                "arm_fraction": arm_fraction,
                "near_gear_fraction": float(near_gear.mean()),
                "gear_share": share,
            }
        )
    parser_score = (
        float(np.mean([row["gear_share"] for row in side_rows])) if side_rows else None
    )
    return (
        {
            "parser_extent_score": parser_score,
            "valid_arm_count": len(side_rows),
            "person_fraction": float(person.mean()),
            "gear_fraction": float(gear.mean()),
            "selected_gear_fraction": float(selected.mean()),
            "arm_sides": side_rows,
        },
        selected,
    )


def extent_class_from_features(
    item_name: str | None,
    clip_probabilities: Sequence[float],
    *,
    medium_threshold: float = 0.30,
    long_threshold: float = 0.45,
) -> tuple[str, float, str]:
    if len(clip_probabilities) != 3:
        raise ValueError("clip_probabilities must contain three values")
    if not 0 <= medium_threshold < long_threshold <= 1:
        raise ValueError("extent thresholds are invalid")
    name_prior = name_extent_prior(item_name)
    if name_prior == "short":
        return "short", 0.15, "calibrated_name_prior"
    if name_prior == "long":
        return "long", 0.75, "calibrated_name_prior"
    small, short_glove, long = (float(value) for value in clip_probabilities)
    score = 0.5 * short_glove + long
    source = "canonical_visual_clip"
    if score < medium_threshold:
        return "short", score, source
    if score < long_threshold:
        return "medium", score, source
    return "long", score, source


def name_extent_prior(item_name: str | None) -> str | None:
    name = str(item_name or "")
    # Explicit long terms win if a rare compound name contains both families.
    if LONG_NAME_PRIOR.search(name):
        return "long"
    if SHORT_NAME_PRIOR.search(name):
        return "short"
    return None


def select_profile_rows(
    metadata: Sequence[Mapping[str, Any]],
    *,
    allowed_piece_genders: set[tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    for row in metadata:
        if str(row.get("slot")) != "HANDS":
            continue
        if str(row.get("candidate_crop")) != CANONICAL_CROP:
            continue
        gender = str(row.get("model_gender") or "").lower()
        for piece in row.get("pieces") or []:
            key = (int(piece["piece_id"]), gender)
            if allowed_piece_genders is not None and key not in allowed_piece_genders:
                continue
            candidate = {
                "piece_id": key[0],
                "model_gender": gender,
                "piece_name": str(piece.get("name_en") or ""),
                "local_path": str(row["local_path"]),
                "race": str(row.get("race") or ""),
                "dye_state": str(row.get("dye_state") or ""),
            }
            current = selected.get(key)
            if current is None or (
                candidate["race"] != "hyur",
                candidate["race"],
                candidate["local_path"],
            ) < (
                current["race"] != "hyur",
                current["race"],
                current["local_path"],
            ):
                selected[key] = candidate
    return [selected[key] for key in sorted(selected)]


def arm_context_crops(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    return [
        image.crop(
            (
                round(left * width),
                round(top * height),
                round(right * width),
                round(bottom * height),
            )
        ).convert("RGB")
        for left, top, right, bottom in ARM_CONTEXT_BOXES
    ]


def clip_feature_tensor(output: Any) -> Any:
    """Normalize transformers versions that wrap CLIP features in outputs."""

    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output


def scoped_piece_genders(
    examples: Sequence[Mapping[str, Any]],
    eligibility: Mapping[str, set[str]],
    assignments: Mapping[str, str],
    *,
    fold_role: str,
) -> set[tuple[int, str]]:
    output = set()
    for example in examples:
        example_id = str(example.get("example_id"))
        if "HANDS" not in eligibility.get(example_id, set()):
            continue
        if assignments.get(str(example.get("glamour_id"))) != fold_role:
            continue
        label = (example.get("core_slot_labels") or {}).get("HANDS") or {}
        gender = str(example.get("model_gender") or "").lower()
        for piece_id in label.get("piece_ids") or []:
            output.add((int(piece_id), gender))
    return output


def save_qa(
    image: Image.Image,
    profile: Mapping[str, Any],
    output_path: Path,
) -> None:
    contexts = arm_context_crops(image)
    panels = [
        ImageOps.contain(panel, (340, 340), Image.Resampling.LANCZOS)
        for panel in (image, *contexts)
    ]
    canvas = Image.new("RGB", (1050, 390), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (6, 5),
        (
            f"{profile['piece_name']} | {profile['model_gender']} | "
            f"{profile['extent_class']} {profile['extent_score']:.3f} "
            f"({profile['extent_source']})"
        ),
        fill="black",
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 350 + 5, 34))
        panel.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    canvas.close()
    for context in contexts:
        context.close()


def stratified_qa_profiles(
    profiles: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[Mapping[str, Any]]:
    if limit < 0:
        raise ValueError("limit must not be negative")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in profiles:
        groups[(str(row["extent_class"]), str(row["extent_source"]))].append(row)
    for rows in groups.values():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"hands-extent-qa:{row['piece_id']}:{row['model_gender']}".encode(
                    "utf-8"
                )
            ).digest()
        )
    output = []
    keys = sorted(groups)
    while len(output) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(output) < limit:
                output.append(groups[key].pop())
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build HANDS coverage profiles from canonical equipment renders. "
            "Calibrated strong name terms handle known visual-language blind "
            "spots; canonical CLIP handles ambiguous gloves. Parser coverage "
            "is retained only as an auditable diagnostic."
        )
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--slot-eligibility-manifest", type=Path)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument(
        "--fold-role", choices=("train", "validation"), default="validation"
    )
    parser.add_argument("--parser-model-id", default="mattmdjaga/segformer_b2_clothes")
    parser.add_argument(
        "--include-parser-diagnostic",
        action="store_true",
        help=(
            "Run the slower clothes parser and retain its coverage metrics. "
            "These metrics never decide the extent class."
        ),
    )
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--name-priors-only",
        action="store_true",
        help=(
            "Emit only high-confidence calibrated name-prior profiles and "
            "skip all visual-model loading. This mode is intended for "
            "training gates after an independent visual audit of the terms."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dilation", type=int, default=61)
    parser.add_argument("--minimum-arm-fraction", type=float, default=0.02)
    parser.add_argument("--medium-threshold", type=float, default=0.30)
    parser.add_argument("--long-threshold", type=float, default=0.45)
    parser.add_argument("--qa-limit", type=int, default=60)
    args = parser.parse_args()
    scoped = args.examples is not None
    required_scope = (
        args.slot_eligibility_manifest is not None,
        args.fold_manifest is not None,
    )
    if scoped != all(required_scope):
        parser.error(
            "--examples requires --slot-eligibility-manifest and --fold-manifest"
        )
    if args.batch_size < 1 or args.qa_limit < 0:
        parser.error("batch-size must be positive and qa-limit non-negative")

    allowed = None
    if scoped:
        from ffxiv_prism.vision.evaluation_split import load_split_assignments
        from scripts.train_glamour_query_encoder import load_slot_eligibility

        allowed = scoped_piece_genders(
            read_jsonl(args.examples),
            load_slot_eligibility(args.slot_eligibility_manifest),
            load_split_assignments(args.fold_manifest),
            fold_role=args.fold_role,
        )
    rows = select_profile_rows(
        read_jsonl(args.index_dir / "metadata.jsonl"),
        allowed_piece_genders=allowed,
    )
    if args.name_priors_only:
        rows = [row for row in rows if name_extent_prior(row["piece_name"])]
    if not rows:
        raise ValueError("No canonical HANDS rows matched the requested scope")

    if args.name_priors_only:
        torch = None
        device = None
        human_parser = None
        clip_processor = None
        clip_model = None
        text_features = None
    else:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        from ffxiv_prism.torch_device import resolve_torch_device
        from ffxiv_prism.vision.human_parsing import HumanParser

        device = resolve_torch_device(args.device)
        human_parser = (
            HumanParser(args.parser_model_id, str(device))
            if args.include_parser_diagnostic
            else None
        )
        clip_processor = CLIPProcessor.from_pretrained(
            args.clip_model_id,
            cache_dir=CACHE_DIR / "huggingface" / "hub",
            local_files_only=True,
        )
        clip_model = (
            CLIPModel.from_pretrained(
                args.clip_model_id,
                cache_dir=CACHE_DIR / "huggingface" / "hub",
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        text_inputs = clip_processor(
            text=list(CLIP_PROMPTS), return_tensors="pt", padding=True
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.inference_mode():
            text_features = clip_feature_tensor(
                clip_model.get_text_features(**text_inputs)
            )
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

    output_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        images = []
        for row in batch:
            with Image.open(row["local_path"]) as source:
                images.append(source.convert("RGB"))
        segmentations = (
            human_parser.parse(images)
            if human_parser is not None
            else [None] * len(images)
        )
        if args.name_priors_only:
            clip_crops = []
            crop_probabilities = np.zeros((2 * len(images), 3), dtype=np.float32)
        else:
            assert clip_processor is not None
            assert clip_model is not None
            assert text_features is not None
            assert torch is not None
            clip_crops = [crop for image in images for crop in arm_context_crops(image)]
            clip_inputs = clip_processor(images=clip_crops, return_tensors="pt")
            clip_inputs = {key: value.to(device) for key, value in clip_inputs.items()}
            with torch.inference_mode():
                image_features = clip_feature_tensor(
                    clip_model.get_image_features(**clip_inputs)
                )
                image_features = torch.nn.functional.normalize(image_features, dim=-1)
                logits = clip_model.logit_scale.exp() * image_features @ text_features.T
                crop_probabilities = logits.softmax(dim=-1).cpu().numpy()
        for index, (row, image, segmentation) in enumerate(
            zip(batch, images, segmentations, strict=True)
        ):
            probabilities = crop_probabilities[2 * index : 2 * index + 2].mean(axis=0)
            if segmentation is None:
                parser_features = {
                    "parser_extent_score": None,
                    "valid_arm_count": 0,
                    "person_fraction": None,
                    "gear_fraction": None,
                    "selected_gear_fraction": None,
                    "arm_sides": [],
                }
                selected = np.zeros((image.height, image.width), dtype=bool)
            else:
                parser_features, selected = parser_extent_features(
                    segmentation,
                    dilation=args.dilation,
                    minimum_arm_fraction=args.minimum_arm_fraction,
                )
            extent_class, score, source = extent_class_from_features(
                row["piece_name"],
                probabilities,
                medium_threshold=args.medium_threshold,
                long_threshold=args.long_threshold,
            )
            profile = {
                **row,
                "extent_class": extent_class,
                "extent_score": score,
                "extent_source": source,
                "clip_probabilities": {
                    name: float(value)
                    for name, value in zip(
                        ("small_accessory", "short_glove", "long_forearm"),
                        probabilities,
                        strict=True,
                    )
                },
                **parser_features,
            }
            output_rows.append(profile)
            image.close()
        for crop in clip_crops:
            crop.close()
        print(
            f"profiles={min(start + args.batch_size, len(rows))}/{len(rows)}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "profiles.jsonl", output_rows)
    for index, profile in enumerate(
        stratified_qa_profiles(output_rows, limit=args.qa_limit)
    ):
        with Image.open(profile["local_path"]) as source:
            image = source.convert("RGB")
        save_qa(
            image,
            profile,
            args.output_dir / "qa" / f"{index:04d}.jpg",
        )
        image.close()
    class_counts = Counter(str(row["extent_class"]) for row in output_rows)
    source_counts = Counter(str(row["extent_source"]) for row in output_rows)
    summary = {
        "version": 1,
        "status": "awaiting_visual_calibration",
        "profile_count": len(output_rows),
        "extent_class_counts": dict(class_counts),
        "extent_source_counts": dict(source_counts),
        "scoped_to_fold": scoped,
        "fold_role": args.fold_role if scoped else None,
        "blind_test_accessed": False,
        "parameters": {
            "name_priors_only": args.name_priors_only,
            "dilation": args.dilation,
            "minimum_arm_fraction": args.minimum_arm_fraction,
            "medium_threshold": args.medium_threshold,
            "long_threshold": args.long_threshold,
            "clip_prompts": list(CLIP_PROMPTS),
            "include_parser_diagnostic": args.include_parser_diagnostic,
        },
        "sources": {
            "index_dir": portable_path(args.index_dir),
            "examples": portable_path(args.examples),
            "slot_eligibility_manifest": portable_path(args.slot_eligibility_manifest),
            "fold_manifest": portable_path(args.fold_manifest),
        },
        "profiles": portable_path(args.output_dir / "profiles.jsonl"),
        "qa_dir": portable_path(args.output_dir / "qa"),
        "interpretation_limit": (
            "Name-prior profiles are eligible for controlled training only after "
            "the exact term families pass an independent visual audit. Visual "
            "pseudo-labels and parser scores remain diagnostic because short wide "
            "cuffs and bare hands are frequently overclassified."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
