"""Bounded incremental list discovery without assuming a stable remote sort order."""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Any
from urllib.parse import parse_qs, urlparse

from .image_url import parse_gearsets_image_url
from .parser import parse_html


def inspect_list_page(html: str, url: str) -> dict[str, Any]:
    parsed = parse_html(html, url)
    title = (parsed.title or "").casefold()
    links = list(dict.fromkeys(link.href for link in parsed.links
                               if urlparse(link.href).path.startswith("/gearset/")))
    if any(marker in title for marker in ("just a moment", "attention required", "access denied")):
        raise ValueError("Source returned a verification page, not a gearset list")
    if not links and "gearsets" not in title:
        raise ValueError("Unrecognized or empty source page; refusing to advance discovery")
    seeds = sorted({image_url for image in parsed.images for image_url in image.urls
                    if parse_gearsets_image_url(image_url)})
    query = parse_qs(urlparse(url).query)
    pages = []
    for link in parsed.links:
        target = urlparse(link.href)
        target_query = parse_qs(target.query)
        if target.path.rstrip("/") == "/gearsets" and target_query.get("model") == query.get("model"):
            try:
                pages.append(int(target_query["page"][0]))
            except (KeyError, ValueError):
                pass
    fingerprint = hashlib.sha256(json.dumps([sorted(links), seeds]).encode()).hexdigest()
    return {"links": links, "seeds": seeds, "fingerprint": fingerprint,
            "last_page_hint": max(pages, default=0)}


def scan_model(*, model: str, state: dict, known_links: set[str], known_seeds: set[str],
               fetch: Callable[[int], tuple[str, bool]], checkpoint: Callable[[int, dict, bool], None],
               full: bool, max_pages: int = 500, head_pages: int = 3,
               max_head_pages: int = 8, known_stop: int = 2,
               tail_pages: int = 3, empty_stop: int = 2) -> dict:
    """Checkpoint after each valid page; rotate the historical cursor across runs."""
    checked: list[int] = []
    remote = cached = 0
    signatures = state.setdefault("pages", {})

    def visit(page: int) -> tuple[dict, bool]:
        nonlocal remote, cached
        html, from_cache = fetch(page)
        url = f"https://ffxiv.eorzeacollection.com/gearsets?page={page}&model={model}"
        parsed = inspect_list_page(html, url)
        parsed["from_cache"] = from_cache
        previous = signatures.get(str(page), {})
        stable = bool(parsed["links"]) and (
            previous.get("fingerprint") == parsed["fingerprint"] or
            (set(parsed["links"]) <= known_links and set(parsed["seeds"]) <= known_seeds))
        signatures[str(page)] = {"fingerprint": parsed["fingerprint"], "links": len(parsed["links"])}
        state["last_page_hint"] = min(max_pages, max(int(state.get("last_page_hint", 0)), parsed["last_page_hint"], page))
        checked.append(page)
        cached += int(from_cache)
        remote += int(not from_cache)
        known_links.update(parsed["links"])
        known_seeds.update(parsed["seeds"])
        return parsed, stable

    complete = False
    if full:
        start = int(state.get("full_next_page", 1))
        if start > max_pages:
            raise ValueError("Full discovery reached --max-pages; increase the limit to finish initialization")
        seen = set(state.get("full_seen", []))
        empty = int(state.get("full_empty", 0))
        for page in range(start, max_pages + 1):
            parsed, _ = visit(page)
            if not seen and not parsed["links"]:
                raise ValueError("Initial discovery returned no gearsets; refusing to accept an empty baseline")
            empty = empty + 1 if not (set(parsed["links"]) - seen) else 0
            seen.update(parsed["links"])
            complete = empty >= empty_stop
            state.update(full_next_page=page + 1, full_seen=sorted(seen), full_empty=empty, full_complete=complete)
            checkpoint(page, parsed, parsed["from_cache"])
            if complete:
                break
        if complete:
            state.pop("full_seen", None)
            state["tail_cursor"] = head_pages + 1
    else:
        stable_streak = 0
        for page in range(1, min(max_head_pages, max_pages) + 1):
            parsed, stable = visit(page)
            stable_streak = stable_streak + 1 if stable else 0
            checkpoint(page, parsed, parsed["from_cache"])
            if not parsed["links"] or (page >= head_pages and stable_streak >= known_stop):
                break
        # Even when the head is unchanged, historical pages get a bounded turn.
        # When no reliable last-page link exists, advance until a real empty page.
        cursor = max(head_pages + 1, int(state.get("tail_cursor", head_pages + 1)))
        history_empty = int(state.get("tail_empty", 0))
        for _ in range(tail_pages):
            while cursor in checked:
                cursor += 1
            if cursor > max_pages:
                cursor = head_pages + 1
                if cursor in checked:
                    break
            page = cursor
            parsed, _ = visit(page)
            # Repeated terminal pages also count as the end, like full discovery.
            fingerprint = parsed["fingerprint"]
            repeated = state.get("tail_last_fingerprint") == fingerprint
            history_empty = history_empty + 1 if not parsed["links"] or repeated else 0
            state["tail_last_fingerprint"] = fingerprint
            cursor = head_pages + 1 if history_empty >= empty_stop else page + 1
            if history_empty >= empty_stop:
                history_empty = 0
                state.pop("tail_last_fingerprint", None)
                state["rotation_cycles"] = int(state.get("rotation_cycles", 0)) + 1
            state.update(tail_cursor=cursor, tail_empty=history_empty)
            checkpoint(page, parsed, parsed["from_cache"])
            if cursor == head_pages + 1:
                break
    return {"mode": "full" if full else "incremental", "pages_checked": checked,
            "remote_pages": remote, "cached_pages": cached, "full_complete": complete,
            "head_budget": max_head_pages, "history_budget": tail_pages,
            "next_history_page": state.get("tail_cursor", head_pages + 1)}
