from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import (
    CACHE_DIR,
    PROCESSED_DIR,
    portable_path,
    resolve_project_path,
)
from ffxiv_prism.torch_device import resolve_torch_device
from scripts.build_slot_aligned_candidate_index import load_candidate_batch


DEFAULT_BASE_INDEX = Path("data/indexes/facebook-dinov2-base-slot-aligned-v2")
DEFAULT_OUTPUT = Path(
    "data/indexes/facebook-dinov2-base-slot-aligned-v2-body-previews"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add source-backed full-set front previews as supplemental BODY "
            "candidates. Each preview is linked only to the BODY piece declared "
            "by the same Eorzea Collection gearset."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=PROCESSED_DIR / "eorzea.sqlite",
    )
    parser.add_argument("--base-index", type=Path, default=DEFAULT_BASE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--races",
        nargs="+",
        help="Optional normalized race allowlist. The default keeps every race.",
    )
    parser.add_argument(
        "--genders",
        nargs="+",
        help="Optional normalized gender allowlist. The default keeps both.",
    )
    parser.add_argument(
        "--supplement-only",
        action="store_true",
        help="Do not prepend the existing slot-specific candidate index.",
    )
    parser.add_argument("--limit", type=int, help="Debug-only preview row limit.")
    args = parser.parse_args()

    discovered_rows = load_full_preview_body_rows(
        args.db,
        races=args.races,
        genders=args.genders,
        limit=args.limit,
    )
    rows, missing_rows = filter_existing_rows(discovered_rows)
    if not rows:
        raise ValueError("No eligible full-preview BODY candidates were found")
    if missing_rows:
        print(
            f"skipped_missing_local_files={len(missing_rows)}",
            file=sys.stderr,
            flush=True,
        )

    device = resolve_torch_device(args.device)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR / "huggingface" / "hub",
        local_files_only=True,
    ).to(device).eval()
    supplemental_embeddings = embed_rows(
        model,
        rows,
        batch_size=args.batch_size,
        device=device,
    )

    if args.supplement_only:
        output_rows = rows
        output_embeddings = supplemental_embeddings
        base_count = 0
    else:
        base_rows = read_jsonl(args.base_index / "metadata.jsonl")
        base_embeddings = np.load(
            args.base_index / "embeddings.npy",
            mmap_mode="r",
        )
        if len(base_rows) != len(base_embeddings):
            raise ValueError("Base index metadata and embeddings differ in length")
        output_rows = [*base_rows, *rows]
        output_embeddings = np.concatenate(
            [
                np.asarray(base_embeddings, dtype=np.float32),
                supplemental_embeddings,
            ],
            axis=0,
        )
        base_count = len(base_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_dir / "embeddings.npy",
        output_embeddings.astype(np.float32, copy=False),
        allow_pickle=False,
    )
    write_jsonl(args.output_dir / "metadata.jsonl", output_rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "model_id": args.model_id,
            "source_db": portable_path(args.db),
            "base_index": (
                None if args.supplement_only else portable_path(args.base_index)
            ),
            "base_candidate_count": base_count,
            "discovered_supplemental_candidate_count": len(discovered_rows),
            "supplemental_candidate_count": len(rows),
            "missing_local_file_count": len(missing_rows),
            "candidate_count": len(output_rows),
            "supplemental_policy": {
                "image_part": "front",
                "slot": "BODY",
                "piece_link": "same_gearset_exact_slot",
                "candidate_crop": "upper_body_from_full_preview",
                "candidate_crop_box_normalized": [0.0, 0.0, 1.0, 0.72],
                "races": sorted(
                    {str(row.get("race") or "") for row in rows}
                ),
                "genders": sorted(
                    {str(row.get("model_gender") or "") for row in rows}
                ),
            },
            "output": portable_path(args.output_dir),
        },
    )
    print(
        f"base={base_count} supplemental={len(rows)} "
        f"total={len(output_rows)} output={args.output_dir}"
    )
    return 0


def filter_existing_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing = []
    missing = []
    for row in rows:
        path = resolve_project_path(row.get("local_path"))
        if path is None or not path.is_file():
            missing.append(row)
            continue
        existing.append(row)
    return existing, missing


def load_full_preview_body_rows(
    db_path: Path,
    *,
    races: Sequence[str] | None = None,
    genders: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    allowed_races = {value.strip().lower() for value in races or ()}
    allowed_genders = {value.strip().lower() for value in genders or ()}
    sql = """
        SELECT
            i.image_row_id,
            i.source_image_url,
            i.local_path,
            i.image_part,
            i.model_gender,
            i.race,
            i.dye_state,
            i.width,
            i.height,
            gp.piece_id,
            gp.gearset_slug,
            gp.name_en,
            gp.name_zh
        FROM images i
        JOIN gear_pieces gp
          ON gp.gearset_slug = i.gearset_slug
         AND gp.slot = 'BODY'
        WHERE i.qa_status = 'ready'
          AND i.image_part = 'front'
          AND COALESCE(i.dye_state, 'normal') != 'dyed'
        ORDER BY i.image_row_id, gp.piece_id
    """
    grouped: dict[int, dict[str, Any]] = {}
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        for result in connection.execute(sql):
            payload = dict(result)
            race = str(payload.get("race") or "").lower()
            gender = str(payload.get("model_gender") or "").lower()
            if allowed_races and race not in allowed_races:
                continue
            if allowed_genders and gender not in allowed_genders:
                continue
            image_row_id = int(payload["image_row_id"])
            if image_row_id not in grouped:
                grouped[image_row_id] = {
                    key: payload[key]
                    for key in (
                        "image_row_id",
                        "source_image_url",
                        "local_path",
                        "image_part",
                        "model_gender",
                        "race",
                        "dye_state",
                        "width",
                        "height",
                    )
                }
                grouped[image_row_id].update(
                    slot="BODY",
                    candidate_crop="upper_body_from_full_preview",
                    candidate_crop_box_normalized=[0.0, 0.0, 1.0, 0.72],
                    supplemental_candidate_source="full_preview",
                    pieces=[],
                )
            piece = {
                "piece_id": int(payload["piece_id"]),
                "gearset_slug": payload["gearset_slug"],
                "name_en": payload["name_en"],
                "name_zh": payload["name_zh"],
                "link_type": "same_gearset_exact_slot",
            }
            if piece["piece_id"] not in {
                existing["piece_id"]
                for existing in grouped[image_row_id]["pieces"]
            }:
                grouped[image_row_id]["pieces"].append(piece)
    finally:
        connection.close()

    rows = list(grouped.values())
    return rows[:limit] if limit is not None else rows


def embed_rows(
    model: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    batches = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        pixels = load_candidate_batch(batch).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            tokens = model(pixel_values=pixels).last_hidden_state
            features = torch.nn.functional.normalize(tokens[:, 0], dim=-1)
        batches.append(features.float().cpu().numpy())
        completed = min(start + batch_size, len(rows))
        if completed == len(rows) or (start // batch_size + 1) % 25 == 0:
            print(f"embedded={completed}/{len(rows)}", flush=True)
    return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


if __name__ == "__main__":
    raise SystemExit(main())
