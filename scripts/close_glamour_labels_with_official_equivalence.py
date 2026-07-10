from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Close every glamour label over an official piece-equivalence "
            "graph, including labels found through name-variant fallback."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--equivalence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    graph = load_graph(args.equivalence)
    rows = read_jsonl(args.input)
    changed_labels = 0
    added_piece_ids = 0
    for row in rows:
        for label in (row.get("core_slot_labels") or {}).values():
            before = {
                int(value)
                for value in label.get("piece_ids", [])
            }
            if not before:
                continue
            after = close_piece_ids(before, graph)
            if after != before:
                changed_labels += 1
                added_piece_ids += len(after - before)
            label["piece_ids"] = sorted(after)
            label["appearance_equivalence_rule"] = (
                "official_huiji_same_model"
                if len(after) > 1
                else None
            )
    write_jsonl(args.output, rows)
    summary = {
        "input": args.input.as_posix(),
        "equivalence": args.equivalence.as_posix(),
        "output": args.output.as_posix(),
        "records": len(rows),
        "changed_labels": changed_labels,
        "added_piece_ids": added_piece_ids,
        "inference_used": False,
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def load_graph(path: Path) -> dict[int, set[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(piece_id): {int(value) for value in values}
        for piece_id, values in payload["direct_equivalents"].items()
    }


def close_piece_ids(
    piece_ids: set[int],
    graph: dict[int, set[int]],
) -> set[int]:
    output = set(piece_ids)
    pending = list(piece_ids)
    while pending:
        piece_id = pending.pop()
        for equivalent in graph.get(piece_id, set()):
            if equivalent not in output:
                output.add(equivalent)
                pending.append(equivalent)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
