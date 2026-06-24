from __future__ import annotations

import argparse
from pathlib import Path

from ffxiv_prism.paths import INDEXES_DIR, PROCESSED_DIR
from ffxiv_prism.vision.retrieval import DEFAULT_MODEL_ID, VALID_SLOTS, ClipImageEncoder, build_index, load_candidate_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local single-slot visual embedding index.")
    parser.add_argument("--db", type=Path, default=PROCESSED_DIR / "eorzea.sqlite")
    parser.add_argument("--output-dir", type=Path, default=INDEXES_DIR / "clip-vit-base-patch32")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--slots", default=",".join(VALID_SLOTS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-dyed", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Build a bounded smoke-test index.")
    args = parser.parse_args()

    slots = [value.strip().upper() for value in args.slots.split(",") if value.strip()]
    rows = load_candidate_images(args.db, slots, include_dyed=args.include_dyed, limit=args.limit)
    print(f"candidate_images={len(rows)} model={args.model}")
    encoder = ClipImageEncoder(args.model, args.device)
    manifest = build_index(encoder, rows, args.output_dir, args.batch_size)
    print(f"index={args.output_dir} embeddings={manifest['embedding_count']} dim={manifest['embedding_dimension']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
