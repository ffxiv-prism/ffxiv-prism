from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.jsonl import read_jsonl, write_json
from ffxiv_prism.paths import EORZEA_GLAMOUR_METADATA_DIR, INTERIM_DIR, portable_path
from ffxiv_prism.vision.evaluation_split import stable_new_group_role
from scripts.collect_eorzea_glamours import create_http_session


BASE_URL = "https://ffxiv.eorzeacollection.com"
DEFAULT_EXISTING = (
    EORZEA_GLAMOUR_METADATA_DIR / "glamour_examples_raw.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Discover source-backed glamour IDs from Eorzea Collection's "
            "equipment-specific glamour listing pages."
        )
    )
    parser.add_argument("item_ids", nargs="+", type=int)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument(
        "--output",
        type=Path,
        default=INTERIM_DIR / "glamour_item_queue.txt",
    )
    parser.add_argument(
        "--exclude-file",
        action="append",
        type=Path,
        default=[],
        help="Previously attempted queue file; may be passed more than once.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="Also write numbered queue parts of at most this many ids.",
    )
    parser.add_argument(
        "--limit-per-item",
        type=int,
        default=0,
        help="Optional per-source-item cap applied before the global limit.",
    )
    parser.add_argument("--max-pages-per-item", type=int, default=2)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["train"],
        choices=(
            "train",
            "validation",
            "blind_test",
            "development_test",
            "confirmation_test",
        ),
    )
    args = parser.parse_args()

    existing = {
        int(row["glamour_id"])
        for row in read_jsonl(args.existing)
        if row.get("glamour_id")
    }
    excluded = load_excluded_ids(args.exclude_file)
    session = create_http_session()
    sources: dict[int, set[int]] = defaultdict(set)
    fetched_pages = []
    errors = []
    try:
        for item_id in args.item_ids:
            start_url = f"{BASE_URL}/glamours/body/{item_id}"
            queue = [start_url]
            visited = set()
            while queue and len(visited) < args.max_pages_per_item:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)
                try:
                    response = session.get(url, timeout=45)
                    response.raise_for_status()
                    html = response.text
                    glamour_ids, pagination = extract_listing_links(
                        html,
                        response.url,
                        item_id,
                    )
                    for glamour_id in glamour_ids:
                        sources[glamour_id].add(item_id)
                    for page_url in pagination:
                        if page_url not in visited and page_url not in queue:
                            queue.append(page_url)
                    fetched_pages.append(
                        {
                            "item_id": item_id,
                            "url": response.url,
                            "glamour_count": len(glamour_ids),
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "item_id": item_id,
                            "url": url,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if queue:
                    time.sleep(max(0.0, args.delay))
            time.sleep(max(0.0, args.delay))
    finally:
        session.close()

    allowed_roles = set(args.roles)
    candidates = [
        glamour_id
        for glamour_id in sources
        if glamour_id not in existing
        and glamour_id not in excluded
        and stable_new_group_role(glamour_id) in allowed_roles
    ]
    candidates.sort(
        key=lambda glamour_id: (
            len(sources[glamour_id]),
            glamour_id,
        ),
        reverse=True,
    )
    selected = select_balanced_candidates(
        candidates,
        sources,
        args.item_ids,
        limit=max(0, args.limit),
        limit_per_item=max(0, args.limit_per_item),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{glamour_id}\n" for glamour_id in selected),
        encoding="utf-8",
    )
    shard_paths = write_queue_shards(
        selected,
        args.output,
        shard_size=max(0, args.shard_size),
    )
    summary = {
        "item_ids": sorted(set(args.item_ids)),
        "roles": sorted(allowed_roles),
        "existing_glamour_count": len(existing),
        "excluded_attempted_count": len(excluded),
        "fetched_page_count": len(fetched_pages),
        "fetched_pages": fetched_pages,
        "errors": errors,
        "discovered_glamour_count": len(sources),
        "eligible_new_count": len(candidates),
        "limit_per_item": args.limit_per_item,
        "selected_count": len(selected),
        "shards": [portable_path(path) for path in shard_paths],
        "selected": [
            {
                "glamour_id": glamour_id,
                "role": stable_new_group_role(glamour_id),
                "source_item_ids": sorted(sources[glamour_id]),
            }
            for glamour_id in selected
        ],
        "output": portable_path(args.output),
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(
        f"pages={len(fetched_pages)} discovered={len(sources)} "
        f"eligible={len(candidates)} selected={len(selected)} "
        f"errors={len(errors)} output={args.output}"
    )
    return 1 if errors and not fetched_pages else 0


def load_excluded_ids(paths: list[Path]) -> set[int]:
    excluded = set()
    for path in paths:
        if not path.is_file():
            continue
        excluded.update(
            int(value.strip())
            for value in path.read_text(encoding="utf-8").splitlines()
            if value.strip().isdigit()
        )
    return excluded


def write_queue_shards(
    selected: list[int],
    output: Path,
    *,
    shard_size: int,
) -> list[Path]:
    if shard_size <= 0:
        return []
    shard_paths = []
    for start in range(0, len(selected), shard_size):
        shard_path = output.with_name(
            f"{output.stem}.part{start // shard_size + 1:03d}{output.suffix}"
        )
        shard_path.write_text(
            "".join(
                f"{glamour_id}\n"
                for glamour_id in selected[start : start + shard_size]
            ),
            encoding="utf-8",
        )
        shard_paths.append(shard_path)
    return shard_paths


def select_balanced_candidates(
    candidates: list[int],
    sources: dict[int, set[int]],
    item_ids: list[int],
    *,
    limit: int,
    limit_per_item: int,
) -> list[int]:
    if limit_per_item <= 0:
        return candidates[:limit]
    candidate_set = set(candidates)
    selected = []
    seen = set()
    ordered_item_ids = list(dict.fromkeys(item_ids))
    candidates_by_item = {
        item_id: sorted(
            (
                glamour_id
                for glamour_id in candidate_set
                if item_id in sources[glamour_id]
            ),
            reverse=True,
        )
        for item_id in ordered_item_ids
    }
    for offset in range(limit_per_item):
        for item_id in ordered_item_ids:
            item_candidates = candidates_by_item[item_id]
            if offset >= len(item_candidates):
                continue
            glamour_id = item_candidates[offset]
            if glamour_id in seen:
                continue
            selected.append(glamour_id)
            seen.add(glamour_id)
            if len(selected) >= limit:
                return selected
    return selected


def extract_listing_links(
    html: str,
    page_url: str,
    item_id: int,
) -> tuple[set[int], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    glamour_ids = set()
    pagination = []
    expected_path = f"/glamours/body/{item_id}"
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        match = re.search(r"/glamour/(\d+)(?:/|$)", href)
        if match:
            glamour_ids.add(int(match.group(1)))
        absolute = urljoin(page_url, href)
        parsed = urlparse(absolute)
        if parsed.path == expected_path and parsed.query:
            pagination.append(absolute)
    return glamour_ids, list(dict.fromkeys(pagination))


if __name__ == "__main__":
    raise SystemExit(main())
