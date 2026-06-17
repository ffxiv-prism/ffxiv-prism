# 功能：提供 JSONL 读写工具，用于保存可复跑的采集清单和原始元数据。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append rows to a JSONL file and return the number of written rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file. Missing files return an empty list."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

