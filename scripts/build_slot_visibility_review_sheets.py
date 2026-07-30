from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--exclude-reviews",
        type=Path,
        help="JSONL reviews whose example IDs should not be sampled again.",
    )
    parser.add_argument(
        "--exclude-reviewed-glamours",
        action="store_true",
        help="Exclude every glamour represented in --exclude-reviews.",
    )
    parser.add_argument(
        "--one-per-glamour",
        action="store_true",
        help="Select one deterministic image per glamour for independent sampling.",
    )
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
    excluded_example_ids: set[str] = set()
    excluded_glamour_ids: set[str] = set()
    if args.exclude_reviews is not None:
        for row in read_jsonl(args.exclude_reviews):
            excluded_example_ids.add(str(row.get("example_id")))
            if row.get("glamour_id") is not None:
                excluded_glamour_ids.add(str(row.get("glamour_id")))
        if not excluded_glamour_ids:
            glamour_by_example = {
                str(example["example_id"]): glamour_id
                for glamour_id, examples in by_glamour.items()
                for example in examples
            }
            excluded_glamour_ids = {
                glamour_by_example[example_id]
                for example_id in excluded_example_ids
                if example_id in glamour_by_example
            }
        for glamour_id in list(by_glamour):
            if (
                args.exclude_reviewed_glamours
                and glamour_id in excluded_glamour_ids
            ):
                del by_glamour[glamour_id]
                continue
            by_glamour[glamour_id] = [
                example
                for example in by_glamour[glamour_id]
                if str(example["example_id"]) not in excluded_example_ids
            ]
            if not by_glamour[glamour_id]:
                del by_glamour[glamour_id]
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
    selected_by_glamour = {
        glamour_id: (
            [
                deterministic_example(
                    by_glamour[glamour_id],
                    args.seed,
                    glamour_id,
                )
            ]
            if args.one_per_glamour
            else by_glamour[glamour_id]
        )
        for glamour_id in ordered_ids
    }
    outputs = []
    review_rows = []
    for start in range(0, len(ordered_ids), args.rows_per_sheet):
        glamour_ids = ordered_ids[start : start + args.rows_per_sheet]
        max_columns = max(
            len(selected_by_glamour[value]) for value in glamour_ids
        )
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
                selected_by_glamour[glamour_id],
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
            "one_per_glamour": args.one_per_glamour,
            "excluded_reviewed_examples": len(excluded_example_ids),
            "excluded_reviewed_glamours": (
                len(excluded_glamour_ids)
                if args.exclude_reviewed_glamours
                else 0
            ),
            "sheet_count": len(outputs),
            "outputs": outputs,
        },
    )
    write_jsonl(args.output_dir / "review-template.jsonl", review_rows)
    print(f"glamours={len(ordered_ids)} sheets={len(outputs)} output={args.output_dir}")
    return 0


def deterministic_example(
    examples: list[dict],
    seed: int,
    glamour_id: str,
) -> dict:
    if not examples:
        raise ValueError("Cannot select from an empty glamour")
    ordered = sorted(
        examples,
        key=lambda row: (
            int(row.get("image_position") or 0),
            str(row.get("example_id")),
        ),
    )
    digest = hashlib.sha256(
        f"{seed}:{glamour_id}".encode("utf-8")
    ).digest()
    index = int.from_bytes(digest[:8], "big") % len(ordered)
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
