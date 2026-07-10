from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import CACHE_DIR, portable_path
from ffxiv_prism.torch_device import resolve_torch_device
from scripts.evaluate_glamour_embedding_model import DINO_MEAN, DINO_STD

DEFAULT_INPUT = Path("data/indexes/facebook-dinov2-base-color")
DEFAULT_OUTPUT = Path("data/indexes/facebook-dinov2-base-slot-aligned")
CORE_SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")



def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-embed canonical renders with slot-aligned crops so mannequin "
            "skin and unrelated body regions do not dominate matching."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a partially written output index using build_status.json "
            "and embeddings.npy."
        ),
    )
    parser.add_argument(
        "--body-crop-policy",
        choices=("upper", "multiscale"),
        default="upper",
        help=(
            "Use the historical upper-body crop, or add a tighter torso crop "
            "as a second clean-render candidate."
        ),
    )
    parser.add_argument(
        "--head-crop-policy",
        choices=("full", "upper", "face", "multiscale"),
        default="full",
        help="Optionally add upper-accessory and face-accessory HEAD views.",
    )
    parser.add_argument(
        "--hands-crop-policy",
        choices=("default", "context", "multiscale"),
        default="default",
        help=(
            "Use the historical HANDS crops, or add wider forearm/context "
            "views for small gloves and sleeve-like handwear."
        ),
    )
    parser.add_argument(
        "--legs-crop-policy",
        choices=("default", "context", "multiscale"),
        default="default",
        help="Optionally add side/thigh/boot-overlap LEGS candidate views.",
    )
    parser.add_argument(
        "--feet-crop-policy",
        choices=("default", "context", "multiscale"),
        default="default",
        help="Optionally add side and boot-focused FEET candidate views.",
    )
    parser.add_argument(
        "--slots",
        nargs="+",
        choices=CORE_SLOTS,
        default=list(CORE_SLOTS),
    )
    args = parser.parse_args()

    source_rows = read_jsonl(args.input_dir / "metadata.jsonl")
    selected_slots = set(args.slots)
    source_rows = [
        row for row in source_rows
        if str(row.get("slot")) in selected_slots
    ]
    rows = expand_slot_aligned_rows(
        source_rows,
        body_crop_policy=args.body_crop_policy,
        head_crop_policy=args.head_crop_policy,
        hands_crop_policy=args.hands_crop_policy,
        legs_crop_policy=args.legs_crop_policy,
        feet_crop_policy=args.feet_crop_policy,
    )
    device = resolve_torch_device(args.device)
    args.device = str(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.output_dir / "embeddings.npy"
    status_path = args.output_dir / "build_status.json"
    summary_path = args.output_dir / "summary.json"
    slot_policy = {
        "HEAD": [
            name
            for name, _ in crop_variants_for_slot(
                "HEAD",
                head_crop_policy=args.head_crop_policy,
            )
        ],
        "BODY": [
            name
            for name, _ in crop_variants_for_slot(
                "BODY",
                body_crop_policy=args.body_crop_policy,
            )
        ],
        "HANDS": [
            name
            for name, _ in crop_variants_for_slot(
                "HANDS",
                hands_crop_policy=args.hands_crop_policy,
            )
        ],
        "LEGS": [
            name
            for name, _ in crop_variants_for_slot(
                "LEGS",
                legs_crop_policy=args.legs_crop_policy,
            )
        ],
        "FEET": [
            name
            for name, _ in crop_variants_for_slot(
                "FEET",
                feet_crop_policy=args.feet_crop_policy,
            )
        ],
    }
    summary = {
        "source_index": portable_path(args.input_dir),
        "model_id": args.model_id,
        "source_image_count": len(source_rows),
        "aligned_candidate_count": len(rows),
        "slot_policy": slot_policy,
        "output": portable_path(args.output_dir),
        "completed": False,
    }
    write_jsonl(args.output_dir / "metadata.jsonl", rows)
    write_json(summary_path, summary)

    completed_rows = 0
    matrix: np.memmap | None = None
    if args.resume and status_path.is_file() and embeddings_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        completed_rows = int(status.get("completed_rows", 0))
        matrix = np.load(embeddings_path, mmap_mode="r+")
        if int(matrix.shape[0]) != len(rows):
            raise ValueError(
                "Cannot resume index with a different row count: "
                f"{matrix.shape[0]} != {len(rows)}"
            )
        completed_rows = min(completed_rows, len(rows))
        print(
            f"resuming completed={completed_rows}/{len(rows)} "
            f"dimension={matrix.shape[1]}",
            flush=True,
        )

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device).eval()
    for start in range(completed_rows, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        pixels = load_candidate_batch(batch).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            tokens = model(pixel_values=pixels).last_hidden_state
            features = torch.nn.functional.normalize(tokens[:, 0], dim=-1)
        batch_matrix = features.float().cpu().numpy().astype(
            np.float32,
            copy=False,
        )
        if matrix is None:
            matrix = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(rows), int(batch_matrix.shape[1])),
            )
        end = start + len(batch)
        matrix[start:end] = batch_matrix
        matrix.flush()
        write_json(
            status_path,
            {
                "completed_rows": end,
                "total_rows": len(rows),
                "dimension": int(matrix.shape[1]),
                "completed": end == len(rows),
            },
        )
        print(f"embedded={end}/{len(rows)}", flush=True)
    if matrix is None:
        matrix = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(0, 0),
        )
    summary["dimension"] = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    summary["completed"] = True
    write_json(summary_path, summary)
    write_json(
        status_path,
        {
            "completed_rows": len(rows),
            "total_rows": len(rows),
            "dimension": summary["dimension"],
            "completed": True,
        },
    )
    print(
        f"source={len(source_rows)} aligned={len(rows)} "
        f"dimension={summary['dimension']} output={args.output_dir}"
    )
    return 0


def expand_slot_aligned_rows(
    rows: Sequence[dict[str, Any]],
    *,
    body_crop_policy: str = "upper",
    head_crop_policy: str = "full",
    hands_crop_policy: str = "default",
    legs_crop_policy: str = "default",
    feet_crop_policy: str = "default",
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        slot = str(row.get("slot", ""))
        variants = crop_variants_for_slot(
            slot,
            body_crop_policy=body_crop_policy,
            head_crop_policy=head_crop_policy,
            hands_crop_policy=hands_crop_policy,
            legs_crop_policy=legs_crop_policy,
            feet_crop_policy=feet_crop_policy,
        )
        for name, box in variants:
            aligned = dict(row)
            aligned["candidate_crop"] = name
            aligned["candidate_crop_box_normalized"] = list(box)
            aligned["source_image_row_id"] = row.get("image_row_id")
            aligned["image_row_id"] = len(output)
            output.append(aligned)
    return output


def crop_variants_for_slot(
    slot: str,
    *,
    body_crop_policy: str = "upper",
    head_crop_policy: str = "full",
    hands_crop_policy: str = "default",
    legs_crop_policy: str = "default",
    feet_crop_policy: str = "default",
) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    if slot == "HEAD":
        full = ("full", (0.0, 0.0, 1.0, 1.0))
        if head_crop_policy == "full":
            return (full,)
        if head_crop_policy == "upper":
            return (("head_upper_accessory", (0.0, 0.0, 1.0, 0.68)),)
        if head_crop_policy == "face":
            return (("head_face_accessory", (0.08, 0.20, 0.92, 0.80)),)
        if head_crop_policy == "multiscale":
            return (
                full,
                ("head_upper_accessory", (0.0, 0.0, 1.0, 0.68)),
                ("head_face_accessory", (0.08, 0.20, 0.92, 0.80)),
            )
        raise ValueError(f"unsupported HEAD crop policy: {head_crop_policy}")

    if slot == "BODY":
        upper = ("upper_body", (0.0, 0.0, 1.0, 0.72))
        if body_crop_policy == "upper":
            return (upper,)
        if body_crop_policy == "multiscale":
            return (
                upper,
                ("torso_core", (0.06, 0.14, 0.94, 0.62)),
            )
        raise ValueError(f"unsupported BODY crop policy: {body_crop_policy}")
    if slot == "HANDS":
        default = (
            # Most canonical hand renders use a crossed-arms pose.  A full-width
            # variant is required because each hand and sleeve crosses the body
            # centre and is truncated by either one-sided crop.
            ("both_hands", (0.0, 0.12, 1.0, 1.0)),
            ("left_arm_tight", (0.0, 0.20, 0.45, 1.0)),
            ("right_arm_tight", (0.55, 0.12, 1.0, 0.94)),
        )
        if hands_crop_policy == "default":
            return default
        context = (
            ("left_forearm_context", (0.0, 0.08, 0.58, 1.0)),
            ("right_forearm_context", (0.42, 0.08, 1.0, 1.0)),
            ("hands_upper_context", (0.0, 0.0, 1.0, 0.70)),
        )
        if hands_crop_policy == "context":
            return context
        if hands_crop_policy == "multiscale":
            return default + context
        raise ValueError(f"unsupported HANDS crop policy: {hands_crop_policy}")
    if slot == "LEGS":
        full = ("full", (0.0, 0.0, 1.0, 1.0))
        context = (
            ("legs_left", (0.0, 0.0, 0.55, 1.0)),
            ("legs_right", (0.45, 0.0, 1.0, 1.0)),
            ("legs_thighs", (0.04, 0.0, 0.96, 0.58)),
            ("legs_boot_overlap", (0.04, 0.42, 0.96, 1.0)),
        )
        if legs_crop_policy == "default":
            return (full,)
        if legs_crop_policy == "context":
            return context
        if legs_crop_policy == "multiscale":
            return (full,) + context
        raise ValueError(f"unsupported LEGS crop policy: {legs_crop_policy}")
    if slot == "FEET":
        full = ("full", (0.0, 0.0, 1.0, 1.0))
        context = (
            ("feet_left", (0.0, 0.0, 0.56, 1.0)),
            ("feet_right", (0.44, 0.0, 1.0, 1.0)),
            ("feet_boot_left", (0.0, 0.0, 0.62, 1.0)),
            ("feet_boot_right", (0.38, 0.0, 1.0, 1.0)),
        )
        if feet_crop_policy == "default":
            return (full,)
        if feet_crop_policy == "context":
            return context
        if feet_crop_policy == "multiscale":
            return (full,) + context
        raise ValueError(f"unsupported FEET crop policy: {feet_crop_policy}")
    return (("full", (0.0, 0.0, 1.0, 1.0)),)


def load_candidate_batch(rows: Sequence[dict[str, Any]]) -> torch.Tensor:
    arrays = []
    for row in rows:
        with Image.open(str(row["local_path"])) as source:
            image = source.convert("RGB")
        left, top, right, bottom = row["candidate_crop_box_normalized"]
        box = (
            int(round(left * image.width)),
            int(round(top * image.height)),
            int(round(right * image.width)),
            int(round(bottom * image.height)),
        )
        crop = image.crop(box)
        image.close()
        resized = crop.resize((224, 224), Image.Resampling.BICUBIC)
        crop.close()
        array = np.asarray(resized, dtype=np.float32) / 255.0
        resized.close()
        arrays.append(
            np.transpose(
                (array - DINO_MEAN) / DINO_STD,
                (2, 0, 1),
            )
        )
    return torch.from_numpy(np.stack(arrays))


if __name__ == "__main__":
    raise SystemExit(main())
