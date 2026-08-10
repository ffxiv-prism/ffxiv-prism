from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402
from ffxiv_prism.paths import CACHE_DIR, portable_path, resolve_project_path  # noqa: E402
from ffxiv_prism.torch_device import resolve_torch_device  # noqa: E402
from ffxiv_prism.vision.retrieval import load_candidate_images  # noqa: E402
from scripts.build_slot_aligned_candidate_index import (  # noqa: E402
    expand_slot_aligned_rows,
    load_candidate_batch,
)
from scripts.evaluate_glamour_embedding_model import (  # noqa: E402
    build_encoder,
    pool_dino_tokens,
)


DEFAULT_DB = Path("data/processed/eorzea.sqlite")
DINO_V2 = "facebook/dinov2-base"
DINO_V3 = "var/models/dinov3-vitb16-pretrain-lvd1689m"
SIGLIP2 = "google/siglip2-base-patch16-224"


@dataclass(frozen=True)
class Target:
    name: str
    source: Path
    output: Path
    model_id: str
    encoder_group: str
    slots: tuple[str, ...]
    expected_new_rows: int | None
    full_image_encoding: bool = False
    body_crop_policy: str = "upper"
    head_crop_policy: str = "full"
    hands_crop_policy: str = "default"
    legs_crop_policy: str = "default"
    feet_crop_policy: str = "default"
    pooling: str = "cls"


def target_plan(source_suffix: str, output_suffix: str) -> list[Target]:
    root = Path("data/indexes")

    def source(name: str) -> Path:
        return root / f"{name}-{source_suffix}"

    def output(name: str) -> Path:
        return root / f"{name}-{output_suffix}"

    return [
        Target(
            name="body_dinov2",
            source=source(
                "facebook-dinov2-base-slot-aligned-v3-body-multiscale"
            ),
            output=output(
                "facebook-dinov2-base-slot-aligned-v3-body-multiscale"
            ),
            model_id=DINO_V2,
            encoder_group="dinov2",
            slots=("HEAD", "BODY", "HANDS", "LEGS", "FEET"),
            expected_new_rows=None,
            body_crop_policy="multiscale",
        ),
        Target(
            name="body_dinov3",
            source=source("facebook-dinov3-vitb16-body-multiscale-v3"),
            output=output("facebook-dinov3-vitb16-body-multiscale-v3"),
            model_id=DINO_V3,
            encoder_group="dinov3",
            slots=("BODY",),
            expected_new_rows=None,
            body_crop_policy="multiscale",
        ),
        Target(
            name="head_dinov2",
            source=source("facebook-dinov2-base-head-multiscale-v4"),
            output=output("facebook-dinov2-base-head-multiscale-v4"),
            model_id=DINO_V2,
            encoder_group="dinov2",
            slots=("HEAD",),
            expected_new_rows=None,
            head_crop_policy="multiscale",
        ),
        Target(
            name="head_siglip2",
            source=source(
                "google-siglip2-base-patch16-224-head-multiscale-v4"
            ),
            output=output(
                "google-siglip2-base-patch16-224-head-multiscale-v4"
            ),
            model_id=SIGLIP2,
            encoder_group="siglip2",
            slots=("HEAD",),
            expected_new_rows=None,
            full_image_encoding=True,
            head_crop_policy="multiscale",
        ),
        Target(
            name="head_dinov3",
            source=source("facebook-dinov3-vitb16-head-multiscale-v4"),
            output=output("facebook-dinov3-vitb16-head-multiscale-v4"),
            model_id=DINO_V3,
            encoder_group="dinov3",
            slots=("HEAD",),
            expected_new_rows=None,
            full_image_encoding=True,
            head_crop_policy="multiscale",
        ),
        Target(
            name="hands_dinov2",
            source=source("facebook-dinov2-base-hands-multiscale-v5"),
            output=output("facebook-dinov2-base-hands-multiscale-v5"),
            model_id=DINO_V2,
            encoder_group="dinov2",
            slots=("HANDS",),
            expected_new_rows=None,
            hands_crop_policy="multiscale",
        ),
        Target(
            name="legs_dinov2",
            source=source("facebook-dinov2-base-legs-multiscale-v6"),
            output=output("facebook-dinov2-base-legs-multiscale-v6"),
            model_id=DINO_V2,
            encoder_group="dinov2",
            slots=("LEGS",),
            expected_new_rows=None,
            legs_crop_policy="multiscale",
        ),
        Target(
            name="legs_dinov3",
            source=source("facebook-dinov3-vitb16-legs-multiscale-v6"),
            output=output("facebook-dinov3-vitb16-legs-multiscale-v6"),
            model_id=DINO_V3,
            encoder_group="dinov3",
            slots=("LEGS",),
            expected_new_rows=None,
            legs_crop_policy="multiscale",
        ),
        Target(
            name="feet_dinov2",
            source=source(
                "facebook-dinov2-base-feet-multiscale-v7-clean"
            ),
            output=output(
                "facebook-dinov2-base-feet-multiscale-v7-clean"
            ),
            model_id=DINO_V2,
            encoder_group="dinov2",
            slots=("FEET",),
            expected_new_rows=None,
            feet_crop_policy="multiscale",
        ),
        Target(
            name="feet_siglip2",
            source=source(
                "google-siglip2-base-patch16-224-feet-multiscale-v7-clean"
            ),
            output=output(
                "google-siglip2-base-patch16-224-feet-multiscale-v7-clean"
            ),
            model_id=SIGLIP2,
            encoder_group="siglip2",
            slots=("FEET",),
            expected_new_rows=None,
            full_image_encoding=True,
            feet_crop_policy="multiscale",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Version and incrementally extend all active candidate indexes "
            "without modifying their existing metadata or embeddings."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--base-index",
        type=Path,
        default=None,
        help="Optional legacy base index metadata; SQLite is used by default.",
    )
    parser.add_argument("--source-suffix", default="20260730")
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--expected-gearset-slug", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources and additions without loading encoders or writing indexes.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help=(
            "Optional target names to build. Use this to resume after completed "
            "versioned targets have already been written."
        ),
    )
    args = parser.parse_args()

    base_rows = (
        read_jsonl(args.base_index / "metadata.jsonl")
        if args.base_index is not None
        else load_candidate_images(db_path=args.db)
    )
    targets = target_plan(args.source_suffix, args.suffix)
    if args.targets:
        known_names = {target.name for target in targets}
        unknown_names = sorted(set(args.targets) - known_names)
        if unknown_names:
            raise ValueError(f"Unknown targets: {unknown_names}")
        selected_names = set(args.targets)
        targets = [
            target for target in targets if target.name in selected_names
        ]
    validate_output_dirs(targets)
    additions = {
        target.name: build_additions(
            target,
            base_rows,
            expected_gearset_slug=args.expected_gearset_slug,
        )
        for target in targets
    }
    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "target": target.name,
                        "source": portable_path(target.source),
                        "output": portable_path(target.output),
                        "new_candidate_count": len(additions[target.name]),
                    }
                    for target in targets
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    encoders: dict[str, Any] = {}
    results = []
    for target in targets:
        if target.encoder_group not in encoders:
            print(
                f"loading encoder={target.encoder_group} model={target.model_id}",
                flush=True,
            )
            if target.encoder_group in {"dinov2", "dinov3"}:
                encoders[target.encoder_group] = DinoRowEncoder(
                    target.model_id,
                    args.device,
                    target.pooling,
                )
            else:
                encoders[target.encoder_group] = build_encoder(
                    target.model_id,
                    "clip",
                    args.device,
                    "color",
                    target.pooling,
                )

        rows = additions[target.name]
        encoder = encoders[target.encoder_group]
        if target.encoder_group in {"dinov2", "dinov3"}:
            new_embeddings = encoder.encode_rows(
                rows,
                batch_size=args.batch_size,
                full_image=target.full_image_encoding,
            )
        else:
            new_embeddings = encode_full_image_rows(
                encoder,
                rows,
                batch_size=args.batch_size,
            )
        result = extend_index(target, rows, new_embeddings)
        results.append(result)
        print(
            f"completed target={target.name} "
            f"rows={result['candidate_count']}",
            flush=True,
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


class DinoRowEncoder:
    def __init__(self, model_id: str, device: str, pooling: str) -> None:
        from transformers import AutoModel

        self.model_id = model_id
        self.device = resolve_torch_device(device)
        self.pooling = pooling
        self.model = (
            AutoModel.from_pretrained(
                model_id,
                cache_dir=CACHE_DIR / "huggingface" / "hub",
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )

    def encode_rows(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        batch_size: int,
        full_image: bool,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(rows), batch_size):
            batch = [dict(row) for row in rows[start : start + batch_size]]
            if full_image:
                for row in batch:
                    row["candidate_crop_box_normalized"] = [0.0, 0.0, 1.0, 1.0]
            pixels = load_candidate_batch(batch).to(self.device)
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                tokens = self.model(pixel_values=pixels).last_hidden_state
                features = torch.nn.functional.normalize(
                    pool_dino_tokens(tokens, self.pooling),
                    dim=-1,
                )
            outputs.append(
                features.float().cpu().numpy().astype(np.float32, copy=False)
            )
        return (
            np.concatenate(outputs, axis=0)
            if outputs
            else np.empty((0, 0), dtype=np.float32)
        )


def encode_full_image_rows(
    encoder: Any,
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
) -> np.ndarray:
    paths = [resolve_project_path(row.get("local_path")) for row in rows]
    missing = [
        row.get("local_path")
        for row, path in zip(rows, paths)
        if path is None or not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Candidate images are missing: {missing[:5]}")
    return encoder.encode_paths(
        [path for path in paths if path is not None],
        batch_size,
    )


def build_additions(
    target: Target,
    base_rows: list[dict[str, Any]],
    *,
    expected_gearset_slug: str,
) -> list[dict[str, Any]]:
    source_rows = read_jsonl(target.source / "metadata.jsonl")
    known_images = {canonical_image_key(row) for row in source_rows}
    new_base_rows = [
        row
        for row in base_rows
        if str(row.get("slot")) in set(target.slots)
        and canonical_image_key(row) not in known_images
    ]
    unexpected_slugs = sorted(
        {
            str(piece.get("gearset_slug"))
            for row in new_base_rows
            for piece in row.get("pieces") or []
            if piece.get("gearset_slug") != expected_gearset_slug
        }
    )
    if unexpected_slugs:
        raise ValueError(
            f"{target.name}: unexpected new gearsets {unexpected_slugs}"
        )

    additions = expand_slot_aligned_rows(
        new_base_rows,
        body_crop_policy=target.body_crop_policy,
        head_crop_policy=target.head_crop_policy,
        hands_crop_policy=target.hands_crop_policy,
        legs_crop_policy=target.legs_crop_policy,
        feet_crop_policy=target.feet_crop_policy,
    )
    for offset, row in enumerate(additions):
        row["image_row_id"] = len(source_rows) + offset
    if (
        target.expected_new_rows is not None
        and len(additions) != target.expected_new_rows
    ):
        raise ValueError(
            f"{target.name}: expected {target.expected_new_rows} additions, "
            f"found {len(additions)} from {len(new_base_rows)} images"
        )
    return additions


def extend_index(
    target: Target,
    additions: list[dict[str, Any]],
    new_embeddings: np.ndarray,
) -> dict[str, Any]:
    source_rows = read_jsonl(target.source / "metadata.jsonl")
    source_embeddings = np.load(target.source / "embeddings.npy", mmap_mode="r")
    if source_embeddings.ndim != 2 or source_embeddings.shape[0] != len(
        source_rows
    ):
        raise ValueError(
            f"{target.name}: source embedding/metadata mismatch "
            f"{source_embeddings.shape} vs {len(source_rows)}"
        )
    expected_shape = (len(additions), int(source_embeddings.shape[1]))
    if new_embeddings.shape != expected_shape:
        raise ValueError(
            f"{target.name}: new embedding shape {new_embeddings.shape} "
            f"does not match {expected_shape}"
        )

    target.output.mkdir(parents=True, exist_ok=True)
    output_embeddings_path = target.output / "embeddings.npy"
    matrix = np.lib.format.open_memmap(
        output_embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(source_rows) + len(additions),
            int(source_embeddings.shape[1]),
        ),
    )
    matrix[: len(source_rows)] = source_embeddings
    matrix[len(source_rows) :] = new_embeddings
    matrix.flush()
    output_rows = [*source_rows, *additions]
    write_jsonl(target.output / "metadata.jsonl", output_rows)

    new_gearset_slugs = sorted(
        {
            str(piece.get("gearset_slug"))
            for row in additions
            for piece in row.get("pieces") or []
            if piece.get("gearset_slug")
        }
    )
    source_manifest_path = target.source / "manifest.json"
    if source_manifest_path.is_file():
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        write_json(
            target.output / "manifest.json",
            {
                **source_manifest,
                "embedding_count": len(output_rows),
                "embedding_dimension": int(matrix.shape[1]),
                "embeddings": portable_path(output_embeddings_path),
                "metadata": portable_path(target.output / "metadata.jsonl"),
                "extended_from": portable_path(target.source),
                "new_candidate_count": len(additions),
                "new_gearset_slugs": new_gearset_slugs,
            },
        )

    source_summary_path = target.source / "summary.json"
    source_summary = (
        json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary_path.is_file()
        else {}
    )
    write_json(
        target.output / "summary.json",
        {
            **source_summary,
            "output": portable_path(target.output),
            "extended_from": portable_path(target.source),
            "source_candidate_count": len(source_rows),
            "new_candidate_count": len(additions),
            "aligned_candidate_count": len(output_rows),
            "candidate_count": len(output_rows),
            "source_image_count": len(
                {
                    canonical_image_key(row)
                    for row in output_rows
                }
            ),
            "new_gearset_slugs": new_gearset_slugs,
            "embedding_shape": list(matrix.shape),
            "preserved_source_rows": True,
            "completed": True,
        },
    )
    return {
        "target": target.name,
        "source_candidate_count": len(source_rows),
        "new_candidate_count": len(additions),
        "candidate_count": len(output_rows),
        "output": portable_path(target.output),
    }


def canonical_image_key(row: dict[str, Any]) -> tuple[str, str]:
    source_url = str(row.get("source_image_url") or "").strip()
    if source_url:
        return ("source_image_url", source_url)
    local_path = str(row.get("local_path") or "").strip()
    if local_path:
        return ("local_path", local_path)
    raise ValueError("Candidate row has no image identity")


def validate_output_dirs(targets: Sequence[Target]) -> None:
    nonempty = [
        str(target.output)
        for target in targets
        if target.output.exists() and any(target.output.iterdir())
    ]
    if nonempty:
        raise FileExistsError(
            "Versioned output directories must be empty: " + ", ".join(nonempty)
        )


if __name__ == "__main__":
    raise SystemExit(main())
