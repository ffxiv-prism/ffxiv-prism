# 功能：解析 Eorzea Collection 列表页、套装详情页和单件使用案例页。

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from ffxiv_prism.collectors.eorzea_collection.models import (
    GearPieceRecord,
    GearsetRecord,
    ImageCandidate,
)
from ffxiv_prism.collectors.eorzea_collection.image_url import (
    DYE_STATES,
    GEAR_IMAGE_PARTS,
    GENDER_SLUGS,
    RACE_SLUGS,
    filter_same_family_urls,
    generate_static_image_urls,
    is_gearsets_image_url,
    page_exposed_model_range,
    parse_gearsets_image_url,
)
from ffxiv_prism.collectors.eorzea_collection.url_utils import (
    absolute_url,
    extract_slug,
)


STANDARD_GEAR_SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
GEAR_SLOTS = set(STANDARD_GEAR_SLOTS)
SLOT_TO_IMAGE_PART = {
    "HEAD": "head",
    "BODY": "body",
    "HANDS": "hands",
    "LEGS": "legs",
    "FEET": "feet",
}
IMAGE_PART_TO_SLOT = {image_part: slot for slot, image_part in SLOT_TO_IMAGE_PART.items()}
GENDERS = ("Female", "Male")
RACES = (
    "Hyur Midlander",
    "Midlander",
    "Highlander",
    "Elezen",
    "Miqo'te",
    "Lalafell",
    "Roegadyn",
    "Au Ra",
    "Viera",
    "Hrothgar",
)
JOBS = (
    "PLD",
    "WAR",
    "DRK",
    "GNB",
    "WHM",
    "SCH",
    "AST",
    "SGE",
    "MNK",
    "DRG",
    "NIN",
    "SAM",
    "RPR",
    "VPR",
    "BRD",
    "MCH",
    "DNC",
    "BLM",
    "SMN",
    "RDM",
    "PCT",
    "BLU",
    "CRP",
    "BSM",
    "ARM",
    "GSM",
    "LTW",
    "WVR",
    "ALC",
    "CUL",
    "MIN",
    "BTN",
    "FSH",
)


@dataclass
class LinkEvent:
    href: str
    text: str


@dataclass
class ImageEvent:
    urls: list[str]
    alt: str | None


@dataclass
class TextEvent:
    text: str


@dataclass
class ParsedHtml:
    title: str | None
    text: str
    links: list[LinkEvent] = field(default_factory=list)
    images: list[ImageEvent] = field(default_factory=list)
    events: list[LinkEvent | ImageEvent | TextEvent] = field(default_factory=list)


class EventHtmlParser(HTMLParser):
    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.events: list[LinkEvent | ImageEvent | TextEvent] = []
        self.links: list[LinkEvent] = []
        self.images: list[ImageEvent] = []
        self._anchor_stack: list[dict[str, str]] = []
        self._title_parts: list[str] = []
        self._in_title = False
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key.lower(): value or "" for key, value in attrs}
        if tag == "title":
            self._in_title = True
        if tag == "a" and attr.get("href"):
            self._anchor_stack.append({"href": absolute_url(attr["href"], self.page_url), "text": ""})
        if tag == "img":
            urls = image_urls_from_attrs(attr, self.page_url)
            if urls:
                image = ImageEvent(urls=urls, alt=clean_text(attr.get("alt", "")) or None)
                self.images.append(image)
                self.events.append(image)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._anchor_stack:
            anchor = self._anchor_stack.pop()
            link = LinkEvent(href=anchor["href"], text=clean_text(anchor["text"]))
            self.links.append(link)
            self.events.append(link)

    def handle_data(self, data: str) -> None:
        text = clean_text(data)
        if not text:
            return
        self._text_parts.append(text)
        if self._in_title:
            self._title_parts.append(text)
        if self._anchor_stack:
            self._anchor_stack[-1]["text"] = f"{self._anchor_stack[-1]['text']} {text}".strip()
        self.events.append(TextEvent(text=text))

    def parsed(self) -> ParsedHtml:
        return ParsedHtml(
            title=clean_text(" ".join(self._title_parts)) or None,
            text=clean_text(" ".join(self._text_parts)),
            links=self.links,
            images=self.images,
            events=self.events,
        )


def parse_html(html_text: str, page_url: str) -> ParsedHtml:
    parser = EventHtmlParser(page_url)
    parser.feed(html_text)
    return parser.parsed()


def extract_gearset_links(html_text: str, page_url: str) -> list[str]:
    parsed = parse_html(html_text, page_url)
    links: list[str] = []
    seen: set[str] = set()
    for link in parsed.links:
        path = urlparse(link.href).path
        if path.startswith("/gearset/") and link.href not in seen:
            links.append(link.href)
            seen.add(link.href)
    return links


def extract_pagination_links(html_text: str, page_url: str, path_prefix: str) -> list[str]:
    parsed = parse_html(html_text, page_url)
    links: list[str] = []
    seen: set[str] = set()
    for link in parsed.links:
        parsed_url = urlparse(link.href)
        if parsed_url.path.startswith(path_prefix) and "page=" in parsed_url.query and link.href not in seen:
            links.append(link.href)
            seen.add(link.href)
    return links


def parse_gearset_detail(
    html_text: str,
    page_url: str,
    expand_missing_model_variants: bool = False,
    races: tuple[str, ...] | list[str] | None = None,
    genders: tuple[str, ...] | list[str] | None = None,
    parts: tuple[str, ...] | list[str] | None = None,
    dye_states: tuple[str, ...] | list[str] | None = None,
) -> GearsetRecord:
    parsed = parse_html(html_text, page_url)
    slug = extract_slug(page_url, "gearset")
    title = title_from_text(parsed.title, suffix="| Eorzea Collection")
    if title == "Eorzea Collection":
        title = None

    pieces: list[GearPieceRecord] = []
    page_static_urls: list[str] = []
    seen_page_static_urls: set[str] = set()
    in_equipment = False
    current_slot: str | None = None

    for event in parsed.events:
        if isinstance(event, TextEvent):
            text = normalize_label(event.text)
            if text == "EQUIPMENT" or text.endswith(" EQUIPMENT"):
                in_equipment = True
                current_slot = None
                continue
            if in_equipment and text in {"HOW TO OBTAIN", "MINIMUM LEVEL", "ITEM LEVEL", "WEARABLE BY", "FITS"}:
                current_slot = None
            if in_equipment and text in GEAR_SLOTS:
                current_slot = text
                continue
            if in_equipment and text in {
                "MAIN HAND", "OFF HAND", "MAINHAND", "OFFHAND", "WEAPON", "WEAPONS",
                "ACCESSORIES", "EARRINGS", "EARS", "NECKLACE", "NECK",
                "BRACELETS", "WRISTS", "RING", "RINGS",
            }:
                current_slot = None

        if isinstance(event, LinkEvent):
            if not in_equipment:
                continue
            link_text = clean_text(event.text)
            if current_slot and is_lodestone_item_url(event.href) and link_text:
                pieces.append(
                    GearPieceRecord(
                        slot=current_slot,
                        name=link_text,
                        source_item_url=event.href,
                        occupied_slots=[current_slot],
                    )
                )

        if isinstance(event, ImageEvent):
            for image_url in event.urls:
                if not should_keep_image(image_url):
                    continue
                meta = parse_gearsets_image_url(image_url)
                if not meta:
                    continue
                if image_url not in seen_page_static_urls:
                    page_static_urls.append(image_url)
                    seen_page_static_urls.add(image_url)

    html_pieces = extract_equipment_pieces(html_text, page_url)
    if html_pieces or re.search(
        r"\bc-gear-slot-(?:head|body|hands|legs|feet)\b|gear-icon-box-slot-name",
        html_text,
    ):
        pieces = html_pieces
    present_slots = unique([piece.slot for piece in dedupe_pieces(pieces)])
    missing_slots = [slot for slot in STANDARD_GEAR_SLOTS if slot not in present_slots]
    pieces = infer_equipment_conflicts(dedupe_pieces(pieces), missing_slots, page_static_urls, slug)

    images = build_static_gearset_images(
        page_static_urls,
        page_url,
        slug,
        expand_missing_model_variants=expand_missing_model_variants,
        races=races,
        genders=genders,
        parts=parts,
        dye_states=dye_states,
    )

    profile = infer_profile(parsed.text)
    return GearsetRecord(
        source_page_url=page_url,
        slug=slug,
        title=title,
        dyeable=infer_dyeable(parsed.text),
        roles=infer_roles(parsed.text),
        model_genders=profile["genders"],
        present_slots=present_slots,
        missing_slots=missing_slots,
        pieces=pieces,
        images=images,
    )


def build_static_gearset_images(
    page_static_urls: list[str],
    page_url: str,
    current_gearset_slug: str,
    expand_missing_model_variants: bool = False,
    races: tuple[str, ...] | list[str] | None = None,
    genders: tuple[str, ...] | list[str] | None = None,
    parts: tuple[str, ...] | list[str] | None = None,
    dye_states: tuple[str, ...] | list[str] | None = None,
) -> list[ImageCandidate]:
    """Build static image records for all same-series role gearsets exposed by the page."""

    same_family_urls = filter_same_family_urls(page_static_urls, current_gearset_slug)
    if not same_family_urls:
        return []

    if expand_missing_model_variants:
        races = tuple(races or RACE_SLUGS)
        genders = tuple(genders or GENDER_SLUGS)
    elif races is None or genders is None:
        exposed_races, exposed_genders = page_exposed_model_range(same_family_urls)
        races = tuple(races or exposed_races or RACE_SLUGS)
        genders = tuple(genders or exposed_genders or GENDER_SLUGS)
    parts = tuple(parts or GEAR_IMAGE_PARTS)
    dye_states = tuple(dye_states or DYE_STATES)

    candidate_urls = generate_static_image_urls(
        same_family_urls,
        races=tuple(races),
        genders=tuple(genders),
        parts=parts,
        dye_states=dye_states,
    )

    images: list[ImageCandidate] = []
    seen: set[str] = set()
    for image_url in candidate_urls:
        if image_url in seen:
            continue
        meta = parse_gearsets_image_url(image_url)
        if not meta:
            continue
        images.append(
            ImageCandidate(
                source_image_url=image_url,
                source_page_url=page_url,
                image_kind="gearset_static",
                slot=meta.slot,
                gearset_slug=meta.gearset_slug,
                model_gender=meta.model_gender,
                race=meta.race,
                dye_state=meta.dye_state,
                source_type=meta.source_type,
                source_path=meta.source_path,
                collection_slug=meta.collection_slug,
                role_slug=meta.role_slug,
                family_slug=meta.family_slug,
                image_part=meta.image_part,
            )
        )
        seen.add(image_url)
    return images


def extract_equipment_pieces(html_text: str, page_url: str) -> list[GearPieceRecord]:
    equipment_html = extract_equipment_html(html_text)
    if not equipment_html:
        return extract_modern_equipment_pieces(html_text, page_url)

    # Non-armour slot markers also terminate the previous slot's HTML range.
    # Otherwise an empty FEET slot consumes the following weapon/accessory.
    marker_pattern = re.compile(r"\bc-gear-slot-(?!item\b)([a-z]+(?:-[a-z]+)*)\b", re.I)
    markers = [(match.group(1).upper(), match.start()) for match in marker_pattern.finditer(equipment_html)]
    if not markers:
        return extract_modern_equipment_pieces(html_text, page_url)

    pieces: list[GearPieceRecord] = []
    for index, (slot, start) in enumerate(markers):
        if slot not in GEAR_SLOTS:
            continue
        end = markers[index + 1][1] if index + 1 < len(markers) else len(equipment_html)
        chunk = equipment_html[start:end]
        item_url = first_lodestone_item_url(chunk)
        if not item_url:
            continue
        name = first_item_name(chunk)
        if not name:
            continue
        pieces.append(
            GearPieceRecord(
                slot=slot,
                name=name,
                source_item_url=item_url,
                occupied_slots=[slot],
            )
        )
    return dedupe_pieces(pieces)


def extract_modern_equipment_pieces(
    html_text: str,
    page_url: str,
) -> list[GearPieceRecord]:
    """Parse the v7 list-card equipment markup used by current gearset pages."""

    soup = BeautifulSoup(html_text, "html.parser")
    pieces: list[GearPieceRecord] = []
    for box in soup.select("div.list.box"):
        slot_node = box.select_one(".gear-icon-box-slot-name")
        if slot_node is None:
            continue
        slot = normalize_label(slot_node.get_text(" ", strip=True))
        if slot not in GEAR_SLOTS:
            continue
        for item in box.select(".list-item"):
            name_node = item.select_one(".list-item-title")
            if name_node is None:
                continue
            name = clean_text(name_node.get_text(" ", strip=True))
            if not name:
                continue
            source_node = item.select_one(
                f'a[href*="/glamours/{slot.casefold()}/"]'
            )
            if source_node is None:
                source_node = item.select_one('a[href*="/glamours/"]')
            source_url = None
            if source_node is not None and source_node.get("href"):
                source_url = absolute_url(str(source_node["href"]), page_url)
            pieces.append(
                GearPieceRecord(
                    slot=slot,
                    name=name,
                    source_item_url=source_url,
                    occupied_slots=[slot],
                )
            )
    return dedupe_pieces(pieces)


def extract_equipment_html(html_text: str) -> str:
    match = re.search(
        r"<h4[^>]*>\s*Equipment\s*</h4>(.*?)(?:<div class=\"u-no-sidebar-only\"|<h4[^>]*>\s*Wearable by\s*</h4>|</section>)",
        html_text,
        re.I | re.S,
    )
    return match.group(1) if match else ""


def first_lodestone_item_url(html_chunk: str) -> str | None:
    match = re.search(r'href="([^"]*finalfantasyxiv\.com/lodestone/playguide/db/item/[^"]*)"', html_chunk, re.I)
    if not match:
        return None
    return html.unescape(match.group(1))


def first_item_name(html_chunk: str) -> str | None:
    match = re.search(r'<span[^>]*class="[^"]*\bc-gear-slot-item-name\b[^"]*"[^>]*>(.*?)</span>', html_chunk, re.I | re.S)
    if not match:
        return None
    return clean_text(strip_tags(match.group(1)))


def infer_equipment_conflicts(
    pieces: list[GearPieceRecord],
    missing_slots: list[str],
    page_static_urls: list[str],
    current_gearset_slug: str,
) -> list[GearPieceRecord]:
    if not pieces or not missing_slots:
        return pieces

    static_slots = {
        meta.slot
        for url in page_static_urls
        if (meta := parse_gearsets_image_url(url))
        and meta.gearset_slug == current_gearset_slug
        and meta.slot in missing_slots
    }
    if not static_slots:
        return pieces

    owner_index = next((index for index, piece in enumerate(pieces) if piece.slot == "BODY"), None)
    if owner_index is None and len(pieces) == 1:
        owner_index = 0
    if owner_index is None:
        return pieces

    inferred_blocked_slots = [slot for slot in STANDARD_GEAR_SLOTS if slot in static_slots and slot != pieces[owner_index].slot]
    if pieces[owner_index].slot == "BODY":
        inferred_blocked_slots = [slot for slot in inferred_blocked_slots if slot in {"HANDS", "LEGS", "FEET"}]
    if not inferred_blocked_slots:
        return pieces

    updated: list[GearPieceRecord] = []
    for index, piece in enumerate(pieces):
        if index != owner_index:
            updated.append(piece)
            continue
        occupied_slots = unique([piece.slot, *piece.occupied_slots, *inferred_blocked_slots])
        updated.append(
            GearPieceRecord(
                slot=piece.slot,
                name=piece.name,
                source_item_url=piece.source_item_url,
                occupied_slots=occupied_slots,
                blocked_slots=inferred_blocked_slots,
                equip_conflict_note=(
                    "Inferred multi-slot item: static images exist for empty equipment slots "
                    f"{', '.join(inferred_blocked_slots)}."
                ),
                equip_conflict_source="empty_equipment_slots_with_static_part_images",
            )
        )
    return updated


def strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_label(value: str) -> str:
    value = clean_text(value).upper()
    value = value.strip("-—– ")
    value = re.sub(r"\s+", " ", value)
    return value


def image_urls_from_attrs(attrs: dict[str, str], page_url: str) -> list[str]:
    urls: list[str] = []
    for key in ("src", "data-src", "data-original", "data-lazy-src"):
        if attrs.get(key):
            urls.append(absolute_url(attrs[key], page_url))
    if attrs.get("srcset"):
        for candidate in attrs["srcset"].split(","):
            url = candidate.strip().split(" ")[0]
            if url:
                urls.append(absolute_url(url, page_url))
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def should_keep_image(url: str) -> bool:
    return is_gearsets_image_url(url)


def is_lodestone_item_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.endswith("finalfantasyxiv.com") and "/lodestone/playguide/db/item/" in parsed.path


def title_from_text(value: str | None, suffix: str) -> str | None:
    if not value:
        return None
    title = value.replace(suffix, "").strip()
    title = re.sub(r"(\|\s*)?Eorzea Collection(\s+Eorzea Collection)*$", "", title).strip()
    return clean_text(title) or None


def infer_profile(text: str) -> dict[str, list[str]]:
    return {
        "genders": unique([gender for gender in GENDERS if re.search(rf"\b{re.escape(gender)}\b", text, re.I)]),
        "races": unique([race for race in RACES if re.search(re.escape(race), text, re.I)]),
        "jobs": unique([job for job in JOBS if re.search(rf"\b{re.escape(job)}\b", text)]),
    }


def infer_dyeable(text: str) -> bool | None:
    if re.search(r"\bUndyeable\b", text, re.I):
        return False
    if re.search(r"\bDyeable\b", text, re.I):
        return True
    return None


def infer_roles(text: str) -> list[str]:
    roles = ("Tanks", "Healers", "Melee DPS", "Ranged DPS", "Caster DPS", "Crafters", "Gatherers")
    return unique([role for role in roles if re.search(re.escape(role), text, re.I)])


def dedupe_pieces(pieces: list[GearPieceRecord]) -> list[GearPieceRecord]:
    deduped: list[GearPieceRecord] = []
    seen: set[tuple[str, str]] = set()
    for piece in pieces:
        key = (piece.slot, piece.name)
        if key in seen:
            continue
        deduped.append(piece)
        seen.add(key)
    return deduped


def unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
