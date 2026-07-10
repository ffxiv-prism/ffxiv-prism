from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag


SAME_MODEL_HEADING = "同模型装备"
EXACT_MODEL_HEADING = "模型完全相同物品"
MAIN_MODEL_HEADING = "主模型相同物品"
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


@dataclass(frozen=True)
class HuijiItemLink:
    name_zh: str
    item_url: str


@dataclass(frozen=True)
class HuijiSameModelPage:
    page_title: str | None
    section_found: bool
    exact_same_model: tuple[HuijiItemLink, ...]
    main_same_model: tuple[HuijiItemLink, ...]


def parse_same_model_page(
    html: str,
    page_url: str,
) -> HuijiSameModelPage:
    """Parse HuijiWiki's explicit same-model section without visual inference."""
    soup = BeautifulSoup(html, "html.parser")
    page_title = extract_page_title(soup)
    section = find_same_model_section(soup)
    if section is None:
        return HuijiSameModelPage(page_title, False, (), ())

    root_level = heading_level(section)
    current_group: str | None = None
    exact: list[HuijiItemLink] = []
    main: list[HuijiItemLink] = []
    seen_by_group: dict[str, set[str]] = {
        "exact": set(),
        "main": set(),
    }
    current_url = normalize_item_url(page_url)

    for node in section.find_all_next():
        if not isinstance(node, Tag):
            continue
        if node is not section and node.name in HEADING_TAGS:
            level = heading_level(node)
            text = normalized_text(node)
            if level <= root_level:
                break
            if EXACT_MODEL_HEADING in text:
                current_group = "exact"
                continue
            if MAIN_MODEL_HEADING in text:
                current_group = "main"
                continue

        direct_text = " ".join(
            str(value).strip()
            for value in node.find_all(string=True, recursive=False)
            if str(value).strip()
        )
        if EXACT_MODEL_HEADING in direct_text:
            current_group = "exact"
        elif MAIN_MODEL_HEADING in direct_text:
            current_group = "main"

        if node.name != "a" or current_group is None:
            continue
        href = str(node.get("href") or "").strip()
        item_url = normalize_item_url(urljoin(page_url, href))
        if item_url is None or item_url == current_url:
            continue
        name_zh = clean_item_name(
            node.get_text(" ", strip=True)
            or str(node.get("title") or "")
            or item_name_from_url(item_url)
        )
        if not name_zh or item_url in seen_by_group[current_group]:
            continue
        seen_by_group[current_group].add(item_url)
        link = HuijiItemLink(name_zh=name_zh, item_url=item_url)
        (exact if current_group == "exact" else main).append(link)

    return HuijiSameModelPage(
        page_title=page_title,
        section_found=True,
        exact_same_model=tuple(exact),
        main_same_model=tuple(main),
    )


def build_piece_equivalence(
    pieces: Iterable[dict[str, Any]],
    crawl_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Map Huiji item relations to local piece IDs and close equivalence groups."""
    piece_rows = [normalize_piece(row) for row in pieces]
    pieces_by_en: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pieces_by_zh: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in piece_rows:
        if row["name_en"]:
            pieces_by_en[normalize_name(row["name_en"])].append(row)
        if row["name_zh"]:
            pieces_by_zh[normalize_name(row["name_zh"])].append(row)

    exact_edges: set[tuple[int, int]] = set()
    broad_edges: set[tuple[int, int]] = set()
    edge_rows: list[dict[str, Any]] = []
    unmatched_links: set[tuple[str, str, str]] = set()
    slot_mismatches: set[tuple[str, str, str]] = set()
    matched_records = 0

    for record in crawl_records:
        if record.get("status") != "matched":
            continue
        seeds = lookup_seed_pieces(record, pieces_by_en, pieces_by_zh)
        if not seeds:
            continue
        matched_records += 1
        for relation, field in (
            ("exact_model", "exact_same_model"),
            ("main_model", "main_same_model"),
        ):
            for link in record.get(field, []):
                target_name = clean_item_name(str(link.get("name_zh") or ""))
                targets = pieces_by_zh.get(normalize_name(target_name), [])
                if not targets:
                    unmatched_links.add(
                        (
                            relation,
                            target_name,
                            str(link.get("item_url") or ""),
                        )
                    )
                    continue
                relation_had_edge = False
                for seed in seeds:
                    compatible = [
                        target
                        for target in targets
                        if target["slot"] == seed["slot"]
                    ]
                    if not compatible:
                        slot_mismatches.add(
                            (relation, seed["name_en"], target_name)
                        )
                        continue
                    for target in compatible:
                        pair = sorted_pair(seed["piece_id"], target["piece_id"])
                        if pair is None:
                            continue
                        broad_edges.add(pair)
                        if relation == "exact_model":
                            exact_edges.add(pair)
                        relation_had_edge = True
                if relation_had_edge:
                    edge_rows.append(
                        {
                            "relation": relation,
                            "seed_name_en": str(record.get("seed_name_en") or ""),
                            "seed_name_zh": str(record.get("seed_name_zh") or ""),
                            "target_name_zh": target_name,
                            "target_item_url": str(link.get("item_url") or ""),
                        }
                    )

    exact_equivalents = equivalence_closure(piece_rows, exact_edges)
    broad_equivalents = equivalence_closure(piece_rows, broad_edges)
    return {
        "schema_version": 1,
        "source": "ff14.huijiwiki.com explicit 同模型装备",
        "method": "page_relation",
        "inference_used": False,
        "default_relation": "same_model_including_main_model",
        "direct_equivalents": stringify_map(broad_equivalents),
        "same_model_equivalents": stringify_map(broad_equivalents),
        "exact_model_equivalents": stringify_map(exact_equivalents),
        "edge_count": len(broad_edges),
        "exact_model_edge_count": len(exact_edges),
        "main_model_edge_count": len(broad_edges - exact_edges),
        "matched_crawl_records": matched_records,
        "edges": edge_rows,
        "unmatched_relation_links": [
            {
                "relation": relation,
                "name_zh": name,
                "item_url": url,
            }
            for relation, name, url in sorted(unmatched_links)
        ],
        "slot_mismatches": [
            {
                "relation": relation,
                "seed_name_en": seed_name,
                "target_name_zh": target_name,
            }
            for relation, seed_name, target_name in sorted(slot_mismatches)
        ],
    }


def find_same_model_section(soup: BeautifulSoup) -> Tag | None:
    for heading in soup.find_all(HEADING_TAGS):
        if SAME_MODEL_HEADING in normalized_text(heading):
            return heading
    marker = soup.find(id=re.compile(r"同模型装备|same.?model", re.I))
    if isinstance(marker, Tag):
        heading = marker.find_parent(HEADING_TAGS)
        return heading if isinstance(heading, Tag) else marker
    return None


def normalize_item_url(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.netloc and parts.netloc.casefold() != "ff14.huijiwiki.com":
        return None
    if "/wiki/" not in parts.path:
        return None
    title = unquote(parts.path.rsplit("/wiki/", 1)[-1])
    if not (title.startswith("物品:") or title.startswith("Item:")):
        return None
    scheme = parts.scheme or "https"
    netloc = parts.netloc or "ff14.huijiwiki.com"
    return urlunsplit((scheme, netloc, parts.path, "", ""))


def item_name_from_url(url: str) -> str:
    title = unquote(urlsplit(url).path.rsplit("/wiki/", 1)[-1])
    return clean_item_name(title)


def clean_item_name(value: str) -> str:
    value = " ".join(value.replace("_", " ").strip().split())
    for prefix in ("物品:", "Item:"):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
    return value


def extract_page_title(soup: BeautifulSoup) -> str | None:
    node = (
        soup.select_one("h1#firstHeading")
        or soup.select_one(".mw-page-title-main")
        or soup.select_one("meta[property='og:title']")
    )
    if node is None:
        return None
    value = (
        str(node.get("content") or "")
        if node.name == "meta"
        else node.get_text(" ", strip=True)
    )
    return clean_item_name(value.split(" - ", 1)[0])


def heading_level(node: Tag) -> int:
    if node.name in HEADING_TAGS:
        return int(node.name[1])
    return 6


def normalized_text(node: Tag) -> str:
    return " ".join(node.get_text(" ", strip=True).split())


def normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def normalize_piece(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "piece_id": int(row["piece_id"]),
        "slot": str(row.get("slot") or ""),
        "name_en": str(row.get("name_en") or row.get("name") or "").strip(),
        "name_zh": clean_item_name(str(row.get("name_zh") or "")),
    }


def lookup_seed_pieces(
    record: dict[str, Any],
    pieces_by_en: dict[str, list[dict[str, Any]]],
    pieces_by_zh: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    values: dict[int, dict[str, Any]] = {}
    name_en = normalize_name(str(record.get("seed_name_en") or ""))
    name_zh = normalize_name(clean_item_name(str(record.get("seed_name_zh") or "")))
    for row in pieces_by_en.get(name_en, []):
        values[row["piece_id"]] = row
    for row in pieces_by_zh.get(name_zh, []):
        values[row["piece_id"]] = row
    return list(values.values())


def sorted_pair(left: int, right: int) -> tuple[int, int] | None:
    if left == right:
        return None
    return (left, right) if left < right else (right, left)


def equivalence_closure(
    pieces: list[dict[str, Any]],
    edges: set[tuple[int, int]],
) -> dict[int, list[int]]:
    parent = {row["piece_id"]: row["piece_id"] for row in pieces}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        if left in parent and right in parent:
            union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for piece_id in parent:
        components[find(piece_id)].append(piece_id)
    output: dict[int, list[int]] = {}
    for values in components.values():
        if len(values) < 2:
            continue
        values = sorted(values)
        for piece_id in values:
            output[piece_id] = [
                other for other in values if other != piece_id
            ]
    return output


def stringify_map(values: dict[int, list[int]]) -> dict[str, list[int]]:
    return {
        str(piece_id): equivalents
        for piece_id, equivalents in sorted(values.items())
    }
