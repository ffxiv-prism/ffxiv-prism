from __future__ import annotations

import json
import os
import tempfile
import time
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
    """Atomically replace a JSONL file and return the number of written rows."""

    return write_jsonl_atomic(path, rows)


def _write_jsonl_file(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write JSONL content directly to a new or temporary path."""

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
        count = _write_jsonl_file(temporary_path, rows)
        replace_with_retry(temporary_path, path)
        return count
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON file with stable formatting."""

    write_json_atomic(path, payload)


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON content directly to a new or temporary path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(portable_payload(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
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
        _write_json_file(temporary_path, payload)
        replace_with_retry(temporary_path, path)
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


def replace_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 40,
    delay_seconds: float = 0.25,
) -> None:
    """Replace a file while tolerating short-lived Windows reader locks."""

    for attempt in range(max(1, attempts)):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 >= max(1, attempts):
                raise
            time.sleep(max(0.0, delay_seconds))
