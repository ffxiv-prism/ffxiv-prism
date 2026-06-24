from __future__ import annotations

import argparse
import json
from pathlib import Path

from ffxiv_prism.vision.retrieval import DEFAULT_INDEX_DIR, ClipImageEncoder, load_index, search_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Search visually similar FFXIV gear for one armor slot.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gender", choices=("female", "male"))
    parser.add_argument("--race", help="Optional indexed race slug, for example hyur.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    index = load_index(args.index_dir)
    encoder = ClipImageEncoder(index.manifest["model_id"], args.device)
    results = search_index(
        index,
        encoder.encode_path(args.image),
        args.slot,
        args.top_k,
        model_gender=args.gender,
        race=args.race,
    )
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for row in results:
            name = row.get("name_zh") or row.get("name_en") or str(row["piece_id"])
            print(f"{row['rank']:>2}. {row['score']:.4f}  {name}  [{row.get('name_en')}]")
            print(f"    {row['local_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
