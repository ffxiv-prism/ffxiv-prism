"""Build/replay an experimental local HEAD branch without changing production.

Saved pose landmarks define query crops. Every HEAD item receives the same
canonical lower-face/ear crop policy. Optional truth IDs are used only after
retrieval, to report ranks; they never select candidates or views.
"""

# Windows BLAS must initialize before CUDA Torch.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

np.linalg.inv(np.eye(2, dtype=np.float32))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ffxiv_prism.cli.match import (
    _load_appearance_equivalence,
    _load_appearance_equivalence_catalog,
)
from ffxiv_prism.jsonl import read_jsonl, write_jsonl
from ffxiv_prism.matching.equipment_slots import (
    filter_candidates_for_slot,
    load_official_slots,
)
from ffxiv_prism.matching.inference import (
    attach_appearance_equivalent_candidates,
    collapse_appearance_equivalent_candidates,
    SlotInferenceConfig,
    crop_records,
    current_best_configs,
    fuse_crop_results,
    slot_result,
    write_html_report,
    write_json_report,
)
from ffxiv_prism.paths import portable_path, resolve_project_path
from ffxiv_prism.vision.outfit_regions import (
    Keypoint,
    build_outfit_regions,
    save_region_crops,
    render_crop_contact_sheet,
    region_manifest,
)
from scripts.build_slot_aligned_candidate_index import crop_variants_for_slot
from scripts.evaluate_glamour_embedding_model import build_encoder, search_crops


def render_comparison(output_dir, rows, ranks):
    selected = ("head_upper_accessory", "head_face_accessory", "head_lower_face")
    paths = {r["name"]: r["path"] for r in rows}
    canvas = Image.new("RGB", (1080, 425), "#20232b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.text(
        (12, 10),
        "Same SigLIP model and catalogue: zero-shot crop comparison",
        fill="white",
        font=font,
    )
    for i, name in enumerate(selected):
        x = i * 360
        with Image.open(paths[name]) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (340, 320))
        canvas.paste(
            thumb, (x + (360 - thumb.width) // 2, 44 + (320 - thumb.height) // 2)
        )
        draw.text((x + 12, 374), name, fill="white", font=font)
        rank = ranks[name]
        draw.text(
            (x + 12, 398),
            f"Target rank: {rank if rank is not None else 'n/a'}",
            fill="#9fceff",
            font=font,
        )
    canvas.save(output_dir / "crop-comparison.jpg", quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--truth-piece-ids", nargs="*", type=int, default=[])
    args = parser.parse_args()
    if args.report_dir.resolve() == args.output_dir.resolve():
        parser.error("Use a separate output directory; preserve the original report.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose = json.loads((args.report_dir / "regions.json").read_text("utf-8"))
    source = resolve_project_path(pose["source_image"])
    points = [Keypoint(p["x"], p["y"], p["score"]) for p in pose["keypoints"]]
    with Image.open(source) as image:
        regions = [
            r for r in build_outfit_regions(points, image.size) if r.slot == "HEAD"
        ]
        paths = save_region_crops(image, regions, args.output_dir / "crops", "source")
        render_crop_contact_sheet(image, regions).save(
            args.output_dir / "head-crops.jpg", quality=94
        )
        image_info = {
            "path": portable_path(source),
            "width": image.width,
            "height": image.height,
        }
    manifest = {
        **region_manifest(points, tuple(pose["person_box"]), regions),
        "source_image": portable_path(source),
        "crop_paths": [portable_path(p) for p in paths],
    }
    write_json_report(args.output_dir / "regions.json", manifest)
    query = crop_records(regions, paths)
    config = current_best_configs()["HEAD"]
    model_id = "google/siglip2-base-patch16-224"
    existing_index = config.ensemble_index_dirs[
        config.ensemble_model_ids.index(model_id)
    ]
    metadata_path = existing_index / "metadata.jsonl"
    source_rows = [
        r for r in read_jsonl(metadata_path) if r.get("candidate_crop") == "full"
    ]
    variants = crop_variants_for_slot("HEAD", head_crop_policy="lower-face")
    contract = {
        "version": 1,
        "model_id": model_id,
        "preprocess": "color",
        "source_metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
        "variants": [[n, list(b)] for n, b in variants],
    }
    index = args.index_dir or args.output_dir / "candidate-index"
    index.mkdir(parents=True, exist_ok=True)
    contract_path = index / "contract.json"
    encoder = build_encoder(model_id, "auto", args.device, "color")
    if contract_path.is_file():
        if json.loads(contract_path.read_text("utf-8")) != contract:
            raise ValueError(
                "Candidate crop/model contract changed; use a new index directory"
            )
        metadata = read_jsonl(index / "metadata.jsonl")
        embeddings = np.load(index / "embeddings.npy", mmap_mode="r")
        if len(metadata) != len(source_rows) * len(variants) or embeddings.shape[
            0
        ] != len(metadata):
            raise ValueError("Incomplete local HEAD index")
    else:
        metadata, matrices = [], []
        original_prepare = encoder.prepare_image
        for name, box in variants:

            def prepare(path, box=box):
                image = original_prepare(path)
                result = image.crop(
                    tuple(
                        round(v * (image.width if i % 2 == 0 else image.height))
                        for i, v in enumerate(box)
                    )
                )
                image.close()
                return result

            encoder.prepare_image = prepare
            matrices.append(
                encoder.encode_paths(
                    [resolve_project_path(r["local_path"]) for r in source_rows],
                    args.batch_size,
                )
            )
            for row in source_rows:
                metadata.append(
                    dict(
                        row,
                        candidate_crop=name,
                        candidate_crop_box_normalized=list(box),
                        image_row_id=len(metadata),
                    )
                )
        encoder.prepare_image = original_prepare
        embeddings = np.concatenate(matrices)
        np.save(index / "embeddings.npy", embeddings, allow_pickle=False)
        write_jsonl(index / "metadata.jsonl", metadata)
        write_json_report(contract_path, contract)
    vectors = dict(
        zip(map(str, paths), encoder.encode_paths(paths, args.batch_size), strict=True)
    )
    np.savez_compressed(
        args.output_dir / "queries.npz",
        paths=np.asarray(list(vectors)),
        embeddings=np.asarray(list(vectors.values())),
    )
    routes = {name: (name,) for name, _ in variants}
    rows = [
        dict(r, role="auxiliary")
        for r in query["query"]
        if r["name"] in routes
        and r["visibility"] == "visible"
        and r["confidence"] >= 0.55
    ]
    searches = search_crops(
        metadata,
        embeddings,
        {"query": rows},
        vectors,
        2000,
        device=args.device,
        candidate_crop_routes=routes,
    )
    # Isolate the effect of query geometry from the separate candidate-policy
    # experiment: all saved HEAD views also search the unmodified index.
    existing_searches = search_crops(
        read_jsonl(metadata_path),
        np.load(existing_index / "embeddings.npy", mmap_mode="r"),
        query,
        vectors,
        2000,
        device=args.device,
    )
    eq = _load_appearance_equivalence()
    catalog, official_slots = (
        _load_appearance_equivalence_catalog(),
        load_official_slots(),
    )

    def report_candidates(ranking):
        ranking = attach_appearance_equivalent_candidates(
            ranking, eq, piece_catalog=catalog, max_equivalents_per_anchor=None
        )
        return collapse_appearance_equivalent_candidates(
            filter_candidates_for_slot(ranking, "HEAD", official_slots), eq
        )

    candidates = report_candidates(
        fuse_crop_results(rows, searches, top_k=2000, fusion="rrf", rrf_k=1)
    )
    # Evaluation labels enter only after all crop geometry, encoding and ranks.
    truth = set(args.truth_piece_ids)
    truth |= {v for p in list(truth) for v in eq.get(p, ())}
    metrics = {
        "candidate_rows": len(metadata),
        "candidate_piece_count": len(
            {p["piece_id"] for r in metadata for p in r["pieces"]}
        ),
        "query_only_raw_view_ranks": {
            r["name"]: next(
                (
                    v["rank"]
                    for v in existing_searches[str(r["path"])]
                    if v["piece_id"] in truth
                ),
                None,
            )
            for r in query["query"]
        },
        "raw_view_ranks": {
            r["name"]: next(
                (v["rank"] for v in searches[str(r["path"])] if v["piece_id"] in truth),
                None,
            )
            for r in rows
        },
        "appearance_rank": next(
            (r["rank"] for r in candidates if r["piece_id"] in truth), None
        ),
    }
    render_comparison(
        args.output_dir, query["query"], metrics["query_only_raw_view_ranks"]
    )
    experiment_config = SlotInferenceConfig(
        slot="HEAD",
        index_dir=index,
        query_checkpoint=None,
        query_crop_names=tuple(routes),
        candidate_crop_routes=routes,
        fusion="rrf",
        rrf_k=1,
        benchmark_top1=None,
        benchmark_top10=None,
        benchmark_scope="Experimental zero-shot local HEAD branch; not promoted",
    )
    result = slot_result(experiment_config, rows, candidates[:20])
    result["method"]["model_id"] = model_id
    payload = {
        "input": image_info,
        "profile": "experimental-head-local-crops",
        "device": args.device,
        "top_k": 20,
        "warnings": ["实验性耳侧/下半脸检索；未经过独立身份评测，不替换默认候选。"],
        "artifacts": {
            "contact_sheet": portable_path(args.output_dir / "head-crops.jpg")
        },
        "slots": [result],
        "evaluation": metrics,
    }
    write_json_report(args.output_dir / "results.json", payload)
    write_json_report(args.output_dir / "all-candidates.json", candidates)
    write_json_report(
        args.output_dir / "query-only-candidates.json",
        {r["name"]: existing_searches[str(r["path"])] for r in query["query"]},
    )
    write_html_report(args.output_dir / "report.html", payload)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
