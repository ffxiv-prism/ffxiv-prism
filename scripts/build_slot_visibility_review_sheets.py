from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import PROCESSED_DIR, portable_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build labeled contact sheets for per-image slot visibility review."
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROCESSED_DIR / "glamour_examples.jsonl",
    )
    parser.add_argument("--quality-manifest", type=Path)
    parser.add_argument(
        "--slot",
        choices=("HEAD", "BODY", "HANDS", "LEGS", "FEET"),
    )
    parser.add_argument("--glamour-ids", nargs="+")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-sheet", type=int, default=5)
    parser.add_argument("--cell-width", type=int, default=220)
    parser.add_argument("--cell-height", type=int, default=260)
    args = parser.parse_args()

    if not args.glamour_ids and not args.sample_size:
        parser.error("provide --glamour-ids or --sample-size")
    passed_ids = (
        {
            str(row["example_id"])
            for row in read_jsonl(args.quality_manifest)
            if row.get("passed")
        }
        if args.quality_manifest is not None
        else None
    )
    by_glamour = defaultdict(list)
    for example in read_jsonl(args.examples):
        example_id = str(example["example_id"])
        if passed_ids is not None and example_id not in passed_ids:
            continue
        if args.slot and not (
            ((example.get("core_slot_labels") or {}).get(args.slot) or {}).get(
                "piece_ids"
            )
        ):
            continue
        by_glamour[str(example["glamour_id"])].append(example)
    if args.glamour_ids:
        wanted = [str(value) for value in args.glamour_ids]
    else:
        available = sorted(by_glamour)
        if args.sample_size is None or args.sample_size <= 0:
            parser.error("--sample-size must be positive")
        if args.sample_size > len(available):
            parser.error(
                f"--sample-size {args.sample_size} exceeds {len(available)} "
                "eligible glamours"
            )
        wanted = random.Random(args.seed).sample(available, args.sample_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ordered_ids = [value for value in wanted if value in by_glamour]
    outputs = []
    review_rows = []
    for start in range(0, len(ordered_ids), args.rows_per_sheet):
        glamour_ids = ordered_ids[start : start + args.rows_per_sheet]
        max_columns = max(len(by_glamour[value]) for value in glamour_ids)
        sheet = Image.new(
            "RGB",
            (
                max_columns * args.cell_width,
                len(glamour_ids) * args.cell_height,
            ),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for row_index, glamour_id in enumerate(glamour_ids):
            examples = sorted(
                by_glamour[glamour_id],
                key=lambda row: int(row["image_position"]),
            )
            for column_index, example in enumerate(examples):
                path = Path(str(example["local_path"]))
                with Image.open(path) as source:
                    image = source.convert("RGB")
                thumbnail = ImageOps.contain(
                    image,
                    (args.cell_width - 8, args.cell_height - 28),
                    Image.Resampling.LANCZOS,
                )
                image.close()
                x = column_index * args.cell_width
                y = row_index * args.cell_height
                sheet.paste(
                    thumbnail,
                    (
                        x + (args.cell_width - thumbnail.width) // 2,
                        y + 24,
                    ),
                )
                thumbnail.close()
                draw.rectangle(
                    (x, y, x + args.cell_width - 1, y + args.cell_height - 1),
                    outline=(90, 90, 90),
                )
                draw.text(
                    (x + 4, y + 4),
                    str(example["example_id"]),
                    fill=(0, 0, 0),
                )
                review_rows.append(
                    {
                        "example_id": str(example["example_id"]),
                        "glamour_id": glamour_id,
                        "local_path": str(example["local_path"]),
                        "slot": args.slot,
                        "item_name": str(
                            (
                                (
                                    example.get("core_slot_labels") or {}
                                ).get(args.slot or "")
                                or {}
                            ).get("item_name")
                            or ""
                        ),
                        "visibility": None,
                        "occlusion_reason": None,
                        "review_source": None,
                    }
                )
        output = args.output_dir / f"visibility-sheet-{start // args.rows_per_sheet + 1:03d}.jpg"
        sheet.save(output, quality=92)
        sheet.close()
        outputs.append(portable_path(output))
    write_json(
        args.output_dir / "manifest.json",
        {
            "glamour_ids": ordered_ids,
            "slot": args.slot,
            "seed": args.seed,
            "quality_manifest": (
                portable_path(args.quality_manifest)
                if args.quality_manifest is not None
                else None
            ),
            "review_example_count": len(review_rows),
            "sheet_count": len(outputs),
            "outputs": outputs,
        },
    )
    write_jsonl(args.output_dir / "review-template.jsonl", review_rows)
    print(f"glamours={len(ordered_ids)} sheets={len(outputs)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
