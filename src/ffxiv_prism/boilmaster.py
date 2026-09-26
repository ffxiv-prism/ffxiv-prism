from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_API_ROOT = "https://xivapi-v2.xivcdn.com/api/sheet"
USER_AGENT = "ffxiv-prism/0.1 game-data-cache"


def fetch_sheet_page(
    base_url: str,
    *,
    fields: tuple[str, ...],
    after: int | None,
    limit: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    """Fetch and validate one Boilmaster/XIVAPI v2 sheet page."""

    query: dict[str, str] = {
        "fields": ",".join(fields),
        "limit": str(limit),
    }
    if after is not None:
        query["after"] = str(after)
    url = f"{base_url}?{urlencode(query)}"
    attempts = max(retries, 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("Boilmaster response must be a JSON object")
            if not isinstance(payload.get("rows"), list):
                raise TypeError("Boilmaster response rows must be a list")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise RuntimeError(f"Failed to fetch Boilmaster sheet page: {url}") from last_error


def row_id(row: object) -> int:
    if not isinstance(row, dict):
        raise TypeError("Boilmaster sheet row must be a JSON object")
    value = row.get("row_id")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Invalid Boilmaster row_id: {value!r}") from exc
    if normalized < 0:
        raise ValueError(f"Boilmaster row_id must be non-negative: {normalized}")
    return normalized


def row_fields(row: object) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError("Boilmaster sheet row must be a JSON object")
    fields = row.get("fields")
    if not isinstance(fields, dict):
        raise TypeError("Boilmaster sheet row fields must be a JSON object")
    return fields


def next_after(rows: list[object], previous_after: int | None) -> int:
    """Return the next exclusive row cursor and reject stalled pagination."""

    if not rows:
        raise ValueError("Cannot advance Boilmaster pagination from an empty page")
    normalized_ids = [row_id(row) for row in rows]
    current_after = max(normalized_ids)
    if previous_after is not None and current_after <= previous_after:
        raise RuntimeError(
            "Boilmaster pagination did not advance: "
            f"previous_after={previous_after}, current_after={current_after}"
        )
    return current_after
