from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from ffxiv_prism.paths import portable_payload


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Replace a JSONL file and return the number of written rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(portable_payload(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Replace a JSONL file only after its complete replacement is written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        count = write_jsonl(temporary_path, rows)
        temporary_path.replace(path)
        return count
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(portable_payload(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON file only after its complete replacement is written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        write_json(temporary_path, payload)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
