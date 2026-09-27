from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl
from ffxiv_prism.paths import portable_path
from scripts.evaluate_compact_rank_ensemble import (
    _fuse_all_examples,
    _rank_cases,
    load_compact_piece_ranks,
)
from scripts.evaluate_retrieval_funnel import build_fold_context


QUERY_CROPS = ("hands_forearm_left", "hands_forearm_right")
CANDIDATE_CROP_ORDER = {
    "both_hands": 0,
    "left_arm_tight": 1,
    "right_arm_tight": 2,
    "left_hand": 3,
    "right_hand": 4,
    "arm_context": 5,
}


def select_candidate_row(
    rows: Sequence[dict[str, Any]], model_gender: str | None
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Candidate piece has no metadata rows")
    gender = str(model_gender or "").casefold()
    return min(
        rows,
        key=lambda row: (
            str(row.get("model_gender") or "").casefold() != gender,
            CANDIDATE_CROP_ORDER.get(str(row.get("candidate_crop")), 99),
            int(row.get("image_row_id", 0)),
        ),
    )


def crop_candidate(row: dict[str, Any]) -> Image.Image:
    with Image.open(Path(str(row["local_path"]))) as source:
        image = source.convert("RGB")
    normalized = row.get("candidate_crop_box_normalized")
    if not normalized:
        return image
    left, top, right, bottom = (float(value) for value in normalized)
    box = (
        max(0, min(image.width - 1, round(left * image.width))),
        max(0, min(image.height - 1, round(top * image.height))),
        max(1, min(image.width, round(right * image.width))),
        max(1, min(image.height, round(bottom * image.height))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return image
    cropped = image.crop(box)
    image.close()
    return cropped


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (238, 238, 238))
    canvas.paste(
        fitted,
        ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2),
    )
    fitted.close()
    return canvas


def load_fitted(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return fit_image(image, size)


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    *,
    width: int = 38,
    fill: tuple[int, int, int] = (20, 20, 20),
) -> None:
    draw.multiline_text(
        xy,
        "\n".join(textwrap.wrap(label, width=width) or [""]),
        fill=fill,
        font=ImageFont.load_default(),
        spacing=3,
    )


def render_case(
    example: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    top1_row: dict[str, Any],
    correct_row: dict[str, Any],
    top1_name: str,
    correct_name: str,
    correct_rank: int,
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 470), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    example_id = str(example["example_id"])
    draw_label(
        draw,
        (12, 10),
        f"{example_id}  {example.get('model_gender')} / {example.get('model_race')}  "
        f"correct rank={correct_rank}",
        width=100,
    )

    source = load_fitted(Path(str(example["local_path"])), (250, 360))
    canvas.paste(source, (10, 65))
    source.close()
    draw_label(draw, (10, 430), "player source", width=25)

    by_name = {str(row.get("name")): row for row in records}
    for index, crop_name in enumerate(QUERY_CROPS):
        row = by_name.get(crop_name)
        if row is None:
            continue
        crop = load_fitted(Path(str(row["path"])), (220, 170))
        y = 65 + index * 185
        canvas.paste(crop, (275, y))
        crop.close()
        draw_label(draw, (275, y + 173), crop_name, width=28)

    top1 = crop_candidate(top1_row)
    top1_fitted = fit_image(top1, (350, 300))
    top1.close()
    canvas.paste(top1_fitted, (515, 65))
    top1_fitted.close()
    draw_label(draw, (515, 373), f"WRONG TOP-1: {top1_name}", fill=(170, 25, 25))

    correct = crop_candidate(correct_row)
    correct_fitted = fit_image(correct, (350, 300))
    correct.close()
    canvas.paste(correct_fitted, (900, 65))
    correct_fitted.close()
    draw_label(
        draw,
        (900, 373),
        f"CORRECT TOP-{correct_rank}: {correct_name}",
        fill=(15, 115, 45),
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build visual HANDS reviews for near-miss retrieval failures."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--search-cache", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--crop-rrf-k", type=int, default=1)
    parser.add_argument("--min-rank", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    metadata = [
        row for row in read_jsonl(args.metadata) if str(row.get("slot")) == "HANDS"
    ]
    examples = read_jsonl(args.examples)
    quality = {
        str(row["example_id"]): bool(row.get("passed"))
        for row in read_jsonl(args.quality_manifest)
    }
    context = build_fold_context(
        fold=args.fold,
        fold_manifest=args.fold_manifest,
        all_examples=examples,
        quality_by_id=quality,
        regions_dir=args.regions_dir,
        slot="HANDS",
    )
    paths, piece_ranks = load_compact_piece_ranks(
        args.search_cache, candidate_k=args.candidate_k
    )
    rankings = _fuse_all_examples(
        context["records"],
        {path: index for index, path in enumerate(paths)},
        piece_ranks,
        crop_rrf_k=args.crop_rrf_k,
        candidate_k=args.candidate_k,
    )
    cases = _rank_cases(context["examples"], rankings, metadata, slot="HANDS")
    selected_cases = [
        row
        for row in cases
        if row.get("status") == "eligible"
        and row.get("rank") is not None
        and args.min_rank <= int(row["rank"]) <= args.max_rank
    ][: args.limit]
    examples_by_id = {str(row["example_id"]): row for row in context["examples"]}

    rows_by_piece: dict[int, list[dict[str, Any]]] = defaultdict(list)
    name_by_piece: dict[int, str] = {}
    for row in metadata:
        for piece in row.get("pieces", []):
            piece_id = int(piece["piece_id"])
            rows_by_piece[piece_id].append(row)
            name_by_piece.setdefault(piece_id, str(piece.get("name_en") or piece_id))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in selected_cases:
        example_id = str(case["example_id"])
        example = examples_by_id[example_id]
        ranked_ids = rankings[example_id]
        correct_rank = int(case["rank"])
        top1_id = int(ranked_ids[0])
        correct_id = int(ranked_ids[correct_rank - 1])
        top1_row = select_candidate_row(
            rows_by_piece[top1_id], example.get("model_gender")
        )
        correct_row = select_candidate_row(
            rows_by_piece[correct_id], example.get("model_gender")
        )
        sheet = render_case(
            example,
            context["records"].get(example_id, []),
            top1_row=top1_row,
            correct_row=correct_row,
            top1_name=name_by_piece[top1_id],
            correct_name=name_by_piece[correct_id],
            correct_rank=correct_rank,
        )
        output_path = args.output_dir / f"{example_id.replace(':', '-')}.jpg"
        sheet.save(output_path, quality=94)
        sheet.close()
        manifest.append(
            {
                "example_id": example_id,
                "correct_rank": correct_rank,
                "top1_piece_id": top1_id,
                "top1_name": name_by_piece[top1_id],
                "correct_piece_id": correct_id,
                "correct_name": name_by_piece[correct_id],
                "review_path": portable_path(output_path),
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": portable_path(args.output_dir),
                "manifest": portable_path(manifest_path),
                "review_count": len(manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
