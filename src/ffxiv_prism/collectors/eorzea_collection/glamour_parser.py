from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag


GLAMOUR_BASE_URL = "https://ffxiv.eorzeacollection.com"
CORE_SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")


def parse_glamour_page(html: str, page_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    canonical_url = meta_content(soup, "property", "og:url") or page_url
    glamour_id = extract_glamour_id(canonical_url) or extract_glamour_id(page_url)
    if glamour_id is None:
        raise ValueError(f"Could not determine glamour id from {page_url}")

    canonical_path = urlparse(canonical_url).path.rstrip("/")
    slug = canonical_path.rsplit("/", 1)[-1] if "/" in canonical_path else None
    title_node = soup.select_one("main h1.title span.has-text-primary")
    title = title_node.get_text(" ", strip=True) if title_node else None

    record = {
        "glamour_id": glamour_id,
        "slug": slug,
        "title": title,
        "page_url": f"{GLAMOUR_BASE_URL}/glamour/{glamour_id}",
        "canonical_url": canonical_url,
        "cover_image_url": meta_content(soup, "property", "og:image"),
        "image_urls": extract_glamour_images(soup, glamour_id),
        "submitted_at": extract_submitted_at(soup),
        "tags": extract_tags(soup),
        "description": extract_section_box_text(soup, "Description"),
        "creator": extract_creator(soup),
        "model": extract_about_value(soup, "Model"),
        "intended_for": extract_about_value(soup, "Intended for"),
        "level_to_equip": parse_int(extract_about_value(soup, "Level to Equip")),
        "fits": extract_about_value(soup, "Fits"),
        "equipment": extract_equipment(soup),
        "related_glamour_urls": extract_related_glamours(soup, glamour_id),
    }
    model = record.get("model") or ""
    if model:
        parts = model.rsplit(" ", 1)
        record["model_race"] = parts[0] if len(parts) == 2 else model
        record["model_gender"] = parts[1].lower() if len(parts) == 2 else None
    else:
        record["model_race"] = None
        record["model_gender"] = None
    return record


def enrich_equipment_from_sqlite(
    record: dict[str, Any],
    db_path: Path,
    appearance_equivalence: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        all_rows = conn.execute(
            """
            SELECT piece_id, slot, name_en, occupied_slots_json, blocked_slots_json
            FROM gear_pieces
            ORDER BY piece_id
            """
        ).fetchall()
        for item in record.get("equipment", []):
            name = item.get("item_name")
            if not name:
                set_piece_matches(item, [], "empty")
                continue

            rows = [row for row in all_rows if row["name_en"] == name]
            match_type = "exact_item"
            equivalence_rule = None
            if rows:
                exact_ids = {int(row["piece_id"]) for row in rows}
                equivalent_ids = set(exact_ids)
                for piece_id in exact_ids:
                    equivalent_ids.update((appearance_equivalence or {}).get(piece_id, []))
                if equivalent_ids != exact_ids:
                    rows = [
                        row
                        for row in all_rows
                        if int(row["piece_id"]) in equivalent_ids
                        and row["slot"] == item.get("slot")
                    ]
                    equivalence_rule = "canonical_embedding_similarity"
            else:
                key = appearance_name_key(name)
                rows = [
                    row
                    for row in all_rows
                    if row["slot"] == item.get("slot")
                    and appearance_name_key(row["name_en"]) == key
                ]
                match_type = "appearance_equivalent" if rows else "unmatched"
                equivalence_rule = "normalized_variant_name" if rows else None

            set_piece_matches(item, rows, match_type, equivalence_rule)
    finally:
        conn.close()
    return record


def set_piece_matches(
    item: dict[str, Any],
    rows: list[sqlite3.Row],
    match_type: str,
    equivalence_rule: str | None = None,
) -> None:
    item["piece_match_type"] = match_type
    item["appearance_equivalence_rule"] = equivalence_rule
    item["matched_piece_ids"] = [int(row["piece_id"]) for row in rows]
    item["matched_piece_names"] = sorted(
        {str(row["name_en"]) for row in rows if row["name_en"]}
    )
    item["matched_piece_slots"] = sorted({row["slot"] for row in rows})
    item["occupied_slots"] = sorted(
        {
            slot
            for row in rows
            for slot in parse_json_list(row["occupied_slots_json"])
        }
    )
    item["blocked_slots"] = sorted(
        {
            slot
            for row in rows
            for slot in parse_json_list(row["blocked_slots_json"])
        }
    )


def appearance_name_key(value: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9+]+", " ", (value or "").lower()).strip()
    variant_prefixes = ("augmented ", "antiquated ", "anemos ", "replica ")
    changed = True
    while changed:
        changed = False
        for prefix in variant_prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                changed = True
    normalized = re.sub(r"\bcostume\b", "", normalized)
    normalized = re.sub(r"\s*\+\d+$", "", normalized)
    return " ".join(normalized.split())

def extract_equipment(soup: BeautifulSoup) -> list[dict[str, Any]]:
    equipment: list[dict[str, Any]] = []
    for slot_node in soup.select("main .gear-icon-box-slot-name"):
        box = slot_node.find_parent(class_="list")
        if not isinstance(box, Tag):
            continue
        slot = slot_node.get_text(" ", strip=True).upper()
        if slot not in CORE_SLOTS:
            continue
        item_image = box.select_one(".gear-icon-image")
        item_name = item_image.get("alt") if isinstance(item_image, Tag) else None
        lodestone = box.select_one('a.eorzeadb_link[href*="/db/item/"]')
        lodestone_url = lodestone.get("href") if isinstance(lodestone, Tag) else None
        site_item_id = extract_site_item_id(box, slot)
        equipment.append(
            {
                "slot": slot,
                "is_core_slot": True,
                "is_empty": item_name is None,
                "item_name": item_name,
                "item_icon_url": item_image.get("src") if isinstance(item_image, Tag) else None,
                "lodestone_url": lodestone_url,
                "lodestone_item_hash": lodestone_url.rstrip("/").rsplit("/", 1)[-1] if lodestone_url else None,
                "site_item_id": site_item_id,
                "dyes": extract_dyes(box),
                "online_store_exclusive": "Online Store Exclusive" in box.get_text(" ", strip=True),
            }
        )
    return equipment


def extract_dyes(box: Tag) -> list[str]:
    dyes: list[str] = []
    for tag in box.select(".list-item-description .tag"):
        text = tag.get_text(" ", strip=True)
        if "Online Store Exclusive" in text:
            continue
        text = text.replace("⬤", "").replace("◯", "").strip()
        if text and text not in dyes:
            dyes.append(text)
    return dyes


def extract_site_item_id(box: Tag, slot: str) -> int | None:
    slot_slug = slot.lower()
    for anchor in box.select("a[href]"):
        href = anchor.get("href") or ""
        match = re.search(rf"/glamours/{re.escape(slot_slug)}/(\d+)", href, re.I)
        if match:
            return int(match.group(1))
        match = re.search(rf"[?&]{re.escape(slot_slug)}Piece=(\d+)", href, re.I)
        if match:
            return int(match.group(1))
    return None


def extract_glamour_images(soup: BeautifulSoup, glamour_id: int) -> list[str]:
    prefix = f"https://glamours.eorzeacollection.com/{glamour_id}/"
    return unique(
        image.get("src")
        for image in soup.select("main img[src]")
        if (image.get("src") or "").startswith(prefix)
    )


def extract_related_glamours(soup: BeautifulSoup, glamour_id: int) -> list[str]:
    urls = []
    for anchor in soup.select("main article.collection.glamour a.glamour-image[href]"):
        href = anchor.get("href") or ""
        related_id = extract_glamour_id(href)
        if related_id is not None and related_id != glamour_id:
            urls.append(urljoin(GLAMOUR_BASE_URL, href))
    return unique(urls)


def extract_creator(soup: BeautifulSoup) -> dict[str, Any] | None:
    divider = find_divider(soup, "Creator")
    box = divider.find_next_sibling(class_="box") if divider else None
    if not isinstance(box, Tag):
        return None
    name = box.select_one("h3")
    server = box.select_one("h4")
    link = box.select_one('a[href^="/creator/"]')
    return {
        "name": name.get_text(" ", strip=True) if name else None,
        "server": clean_server(server.get_text(" ", strip=True)) if server else None,
        "profile_url": urljoin(GLAMOUR_BASE_URL, link.get("href")) if link else None,
    }


def extract_about_value(soup: BeautifulSoup, label: str) -> str | None:
    divider = find_divider(soup, "About this glamour")
    box = divider.find_next_sibling(class_="box") if divider else None
    if not isinstance(box, Tag):
        return None
    for row in box.select(".columns"):
        label_node = row.select_one(".subtitle")
        if label_node and label_node.get_text(" ", strip=True) == label:
            columns = row.select(":scope > .column")
            if len(columns) >= 2:
                return columns[1].get_text(" ", strip=True)
    return None


def extract_submitted_at(soup: BeautifulSoup) -> str | None:
    label = soup.find(string=lambda value: value and value.strip() == "Submitted on:")
    if not label:
        return None
    parent = label.parent.parent if label.parent else None
    tag = parent.select_one(".tag") if isinstance(parent, Tag) else None
    text = tag.get_text(" ", strip=True) if tag else None
    if not text:
        return None
    normalized = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text)
    try:
        return datetime.strptime(normalized, "%b %d, %Y").date().isoformat()
    except ValueError:
        return text


def extract_tags(soup: BeautifulSoup) -> list[str]:
    label = soup.find(string=lambda value: value and value.strip() == "Tags:")
    parent = label.parent.parent if label and label.parent else None
    return [anchor.get_text(" ", strip=True) for anchor in parent.select("a.tag")] if isinstance(parent, Tag) else []


def extract_section_box_text(soup: BeautifulSoup, heading: str) -> str | None:
    divider = find_divider(soup, heading)
    box = divider.find_next_sibling(class_="box") if divider else None
    return box.get_text("\n", strip=True) if isinstance(box, Tag) else None


def find_divider(soup: BeautifulSoup, text: str) -> Tag | None:
    for node in soup.select("main .divider"):
        if node.get_text(" ", strip=True) == text:
            return node
    return None


def meta_content(soup: BeautifulSoup, attribute: str, value: str) -> str | None:
    node = soup.find("meta", attrs={attribute: value})
    return node.get("content") if isinstance(node, Tag) else None


def extract_glamour_id(url: str) -> int | None:
    match = re.search(r"/glamour/(\d+)", url)
    return int(match.group(1)) if match else None


def parse_int(value: str | None) -> int | None:
    match = re.search(r"\d+", value or "")
    return int(match.group()) if match else None


def clean_server(value: str) -> str:
    return value.replace("⧫", "").replace("«", "").replace("»", "").strip().title()


def parse_json_list(value: str) -> list[str]:
    import json

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def unique(values: Any) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
