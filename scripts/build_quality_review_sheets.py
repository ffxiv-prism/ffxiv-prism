from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build boundary-case review sheets for a quality manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help="Exclude example IDs already covered by an earlier manifest.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count-per-group", type=int, default=20)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument(
        "--all-passing",
        action="store_true",
        help="Also emit numbered sheets covering every passing example.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if args.exclude_manifest is not None:
        excluded = {
            str(row["example_id"])
            for row in read_jsonl(args.exclude_manifest)
        }
        rows = [
            row
            for row in rows
            if str(row["example_id"]) not in excluded
        ]
    passed = sorted(
        (row for row in rows if row.get("passed")),
        key=lambda row: float(row.get("quality_score", 0.0)),
    )[: args.count_per_group]
    rejected = sorted(
        (row for row in rows if not row.get("passed")),
        key=lambda row: float(row.get("quality_score", 0.0)),
        reverse=True,
    )[: args.count_per_group]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_sheet(passed, args.output_dir / "lowest-passing.jpg", args.columns)
    build_sheet(rejected, args.output_dir / "highest-rejected.jpg", args.columns)
    all_passing = [row for row in rows if row.get("passed")]
    if args.all_passing:
        for start in range(0, len(all_passing), args.count_per_group):
            build_sheet(
                all_passing[start : start + args.count_per_group],
                args.output_dir
                / f"all-passing-{start // args.count_per_group + 1:03d}.jpg",
                args.columns,
            )
    print(
        f"passing={len(passed)} rejected={len(rejected)} "
        f"all_passing={len(all_passing)} output={args.output_dir}"
    )
    return 0


def build_sheet(rows: list[dict], output: Path, columns: int) -> None:
    if not rows:
        return
    cell_width, cell_height = 230, 300
    row_count = (len(rows) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * cell_width, row_count * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        with Image.open(str(row["source_image"])) as source:
            image = source.convert("RGB")
        thumbnail = ImageOps.contain(
            image,
            (cell_width - 8, cell_height - 58),
            Image.Resampling.LANCZOS,
        )
        image.close()
        sheet.paste(
            thumbnail,
            (
                x + (cell_width - thumbnail.width) // 2,
                y + 52,
            ),
        )
        thumbnail.close()
        status = "PASS" if row.get("passed") else "FAIL"
        draw.text(
            (x + 4, y + 4),
            f"{row['example_id']} {status} {row.get('quality_score', 0):.3f}",
            fill=(0, 100, 0) if status == "PASS" else (150, 0, 0),
        )
        reasons = ",".join(row.get("rejection_reasons", []))
        draw.text((x + 4, y + 22), reasons[:36], fill=(0, 0, 0))
        draw.text((x + 4, y + 36), reasons[36:72], fill=(0, 0, 0))
        draw.rectangle(
            (x, y, x + cell_width - 1, y + cell_height - 1),
            outline=(100, 100, 100),
        )
    sheet.save(output, quality=92)
    sheet.close()


if __name__ == "__main__":
    raise SystemExit(main())
