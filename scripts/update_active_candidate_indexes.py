from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
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
    parser.add_argument("--source-suffix", default=None, help="Legacy explicit source version; otherwise use the active selection.")
    parser.add_argument("--source-selection", type=Path, default=Path("data/current_best_selection.json"))
    parser.add_argument("--selection", type=Path, default=Path("data/current_best_selection.json"))
    parser.add_argument("--activate", action="store_true", help="Atomically activate all validated outputs after every branch succeeds.")
    parser.add_argument("--sample-manifest", type=Path, help="Only append eligible images from this audited manifest.")
    parser.add_argument("--summary", type=Path, help="Write this run's candidate counts and activation result.")
    parser.add_argument("--suffix", required=True)
    expected_group = parser.add_mutually_exclusive_group(required=True)
    expected_group.add_argument(
        "--expected-gearset-slug",
        action="append",
        dest="expected_gearset_slugs",
        help=(
            "Expected new gearset slug. May be repeated when one release adds "
            "multiple gearsets."
        ),
    )
    expected_group.add_argument(
        "--allow-any-new-gearsets",
        action="store_true",
        help=(
            "Accept every gearset newly present in SQLite. Intended for the "
            "audited sample-library lifecycle pipeline."
        ),
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse target outputs that have a completed, structurally valid "
            "summary and build only missing targets."
        ),
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not create a versioned copy for targets with no new rows.",
    )
    args = parser.parse_args()

    base_rows = (
        read_jsonl(args.base_index / "metadata.jsonl")
        if args.base_index is not None
        else load_candidate_images(db_path=args.db)
    )
    selection = json.loads(args.source_selection.read_text(encoding="utf-8"))
    targets = active_target_plan(selection, args.suffix, source_suffix=args.source_suffix)
    if args.sample_manifest:
        if not args.sample_manifest.is_file():
            raise FileNotFoundError(args.sample_manifest)
        allowed = {row["source_image_url"] for row in read_jsonl(args.sample_manifest)
                   if row.get("eligibility_status") == "eligible"}
        base_rows = [row for row in base_rows if row.get("source_image_url") in allowed]
    if args.activate and args.targets:
        raise ValueError("Activation requires all candidate branches; omit --targets")
    if args.targets:
        known_names = {target.name for target in targets}
        unknown_names = sorted(set(args.targets) - known_names)
        if unknown_names:
            raise ValueError(f"Unknown targets: {unknown_names}")
        selected_names = set(args.targets)
        targets = [
            target for target in targets if target.name in selected_names
        ]
    expected_gearset_slugs = (
        None
        if args.allow_any_new_gearsets
        else frozenset(args.expected_gearset_slugs or [])
    )
    additions = {
        target.name: build_additions(
            target,
            base_rows,
            expected_gearset_slugs=expected_gearset_slugs,
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

    writable_targets = [
        target
        for target in targets
        if additions[target.name] or not args.skip_empty
    ]
    encoders: dict[str, Any] = {}
    results = []
    resumed_names: set[str] = set()
    if args.resume:
        for target in writable_targets:
            if target.output.exists() and any(target.output.iterdir()):
                result = load_completed_target(
                    target,
                    additions[target.name],
                )
                results.append(result)
                resumed_names.add(target.name)
                print(
                    f"reused target={target.name} rows={result['candidate_count']}",
                    flush=True,
                )
    pending_targets = [
        target for target in writable_targets if target.name not in resumed_names
    ]
    validate_output_dirs(pending_targets)

    for target in targets:
        rows = additions[target.name]
        if target.name in resumed_names:
            continue
        if not rows and args.skip_empty:
            result = {
                "target": target.name,
                "source_candidate_count": len(
                    read_jsonl(target.source / "metadata.jsonl")
                ),
                "new_candidate_count": 0,
                "candidate_count": len(
                    read_jsonl(target.source / "metadata.jsonl")
                ),
                "output": portable_path(target.source),
                "skipped": True,
                "reason": "no_new_candidates",
            }
            results.append(result)
            print(
                f"skipped target={target.name} reason=no_new_candidates",
                flush=True,
            )
            continue
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

    activated = False
    if args.activate:
        activated = activate_targets(targets, results, selection, args.selection)
    summary = {
        "activated": activated,
        "new_indexed_images": len({row["source_image_url"] for rows in additions.values() for row in rows}),
        "new_candidate_rows": sum(len(rows) for rows in additions.values()),
        "indexes": results,
    }
    if args.summary:
        write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def active_target_plan(selection: dict[str, Any], suffix: str, *, source_suffix: str | None = None) -> list[Target]:
    base_suffix = source_suffix or str(selection["candidate_suffix"])
    targets = target_plan(base_suffix, suffix)
    if source_suffix:
        return targets
    overrides = selection.get("candidate_indexes") or {}
    return [replace(target, source=Path(overrides.get(
        target.source.name.removesuffix("-" + base_suffix), target.source))) for target in targets]


def activate_targets(targets: list[Target], results: list[dict[str, Any]],
                     source_selection: dict[str, Any], selection_path: Path) -> bool:
    """Commit the selection only after every branch is complete; preserve weights."""
    if len(results) != len(targets) or {r["target"] for r in results} != {t.name for t in targets}:
        raise ValueError("Cannot activate an incomplete candidate update")
    destinations = {r["target"]: Path(r["output"]) for r in results}
    overrides = dict(source_selection.get("candidate_indexes") or {})
    for target in targets:
        destination = destinations[target.name]
        rows = read_jsonl(destination / "metadata.jsonl")
        matrix = np.load(destination / "embeddings.npy", mmap_mode="r")
        if not rows or matrix.ndim != 2 or len(rows) != matrix.shape[0]:
            raise ValueError(f"{target.name}: invalid activation metadata/embedding shape")
        if destination != target.source:
            summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            if summary.get("completed") is not True:
                raise ValueError(f"{target.name}: output is not complete")
        # Output suffix can contain dashes; derive the stable prefix from the plan.
        template = next(t for t in target_plan("source", "output") if t.name == target.name)
        prefix = template.output.name.removesuffix("-output")
        overrides[prefix] = portable_path(destination)
    current = json.loads(selection_path.read_text(encoding="utf-8"))
    if current.get("candidate_indexes") == overrides:
        return True  # A resumed, already committed update.
    if current != source_selection:
        raise ValueError("Active selection changed during update; refusing to overwrite it")
    if not any(not row.get("skipped") for row in results):
        return False
    updated = {**current, "candidate_indexes": overrides,
               "candidate_indexes_updated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    # Existing metrics describe the previous candidate corpus, not this expansion.
    metrics_path = selection_path.parent / "processed/current_recognition_metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.update(metric_status="refresh_required", candidate_update_reason="image_library_expanded")
        write_json(metrics_path, metrics)
    write_json(selection_path, updated)
    return True


def load_completed_target(
    target: Target,
    additions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summary_path = target.output / "summary.json"
    metadata_path = target.output / "metadata.jsonl"
    embeddings_path = target.output / "embeddings.npy"
    required = (summary_path, metadata_path, embeddings_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            f"{target.name}: incomplete resume output; missing {missing}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_rows = read_jsonl(target.source / "metadata.jsonl")
    output_rows = read_jsonl(metadata_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    expected_count = len(source_rows) + len(additions)
    problems: list[str] = []
    if summary.get("completed") is not True:
        problems.append("summary is not completed")
    if int(summary.get("source_candidate_count", -1)) != len(source_rows):
        problems.append("source candidate count changed")
    if int(summary.get("new_candidate_count", -1)) != len(additions):
        problems.append("new candidate count changed")
    if int(summary.get("candidate_count", -1)) != expected_count:
        problems.append("total candidate count changed")
    if len(output_rows) != expected_count:
        problems.append("metadata row count mismatch")
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_count:
        problems.append("embedding row count mismatch")
    if len(output_rows) == expected_count:
        if output_rows[: len(source_rows)] != source_rows:
            problems.append("source metadata prefix changed")
        if additions and output_rows[len(source_rows) :] != list(additions):
            problems.append("appended metadata changed")
    if problems:
        raise ValueError(
            f"{target.name}: existing output cannot be resumed: "
            + "; ".join(problems)
        )
    return {
        "target": target.name,
        "source_candidate_count": len(source_rows),
        "new_candidate_count": len(additions),
        "candidate_count": expected_count,
        "output": portable_path(target.output),
        "reused": True,
    }


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
    expected_gearset_slugs: frozenset[str] | None = None,
    expected_gearset_slug: str | None = None,
) -> list[dict[str, Any]]:
    # ``expected_gearset_slug`` remains as a compatibility shim for callers
    # outside the CLI while the lifecycle pipeline supports multi-gearset
    # releases.
    if expected_gearset_slug is not None:
        if expected_gearset_slugs is not None:
            raise ValueError(
                "Use expected_gearset_slug or expected_gearset_slugs, not both"
            )
        expected_gearset_slugs = frozenset((expected_gearset_slug,))
    source_rows = read_jsonl(target.source / "metadata.jsonl")
    known_images = {canonical_image_key(row) for row in source_rows}
    new_base_rows = [
        row
        for row in base_rows
        if str(row.get("slot")) in set(target.slots)
        and canonical_image_key(row) not in known_images
    ]
    new_gearset_slugs = {
        str(piece.get("gearset_slug"))
        for row in new_base_rows
        for piece in row.get("pieces") or []
        if piece.get("gearset_slug")
    }
    unexpected_slugs = sorted(
        new_gearset_slugs - expected_gearset_slugs
        if expected_gearset_slugs is not None
        else ()
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


def extend_index(target: Target, additions: list[dict[str, Any]], new_embeddings: np.ndarray) -> dict[str, Any]:
    # A stopped build leaves only a private staging directory, never a published
    # half-built version that prevents --resume from trying this branch again.
    validate_output_dirs([target])
    target.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".partial-{target.output.name}-", dir=target.output.parent))
    result = _write_extended_index(replace(target, output=staging), additions, new_embeddings)
    for filename in ("summary.json", "manifest.json"):
        path = staging / filename
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if filename == "summary.json":
                payload["output"] = portable_path(target.output)
            else:
                payload["embeddings"] = portable_path(target.output / "embeddings.npy")
                payload["metadata"] = portable_path(target.output / "metadata.jsonl")
            write_json(path, payload)
    if target.output.exists():
        target.output.rmdir()  # Only an empty directory is permitted.
    staging.rename(target.output)
    result["output"] = portable_path(target.output)
    return result


def _write_extended_index(
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
    if not np.isfinite(new_embeddings).all():
        raise ValueError(f"{target.name}: non-finite embeddings")
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
