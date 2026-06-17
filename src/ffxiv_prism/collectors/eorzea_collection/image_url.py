# 功能：解析和生成 Eorzea Collection gearsets 静态装备展示图片 URL。

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


GEARSETS_IMAGE_HOST = "gearsets.eorzeacollection.com"
RACE_SLUGS = ("hyur", "elezen", "miqote", "lalafell", "roegadyn", "aura", "viera", "hrothgar")
GENDER_SLUGS = ("male", "female")
GEAR_IMAGE_PARTS = ("front", "back", "head", "body", "hands", "legs", "feet")
DYE_STATES = ("normal", "dyed")
ROLE_SUFFIXES = (
    "fending",
    "maiming",
    "striking",
    "scouting",
    "aiming",
    "casting",
    "healing",
    "crafting",
    "gathering",
)


@dataclass(frozen=True)
class GearsetImageMeta:
    source_type: str
    source_path: str
    collection_slug: str
    gearset_slug: str
    role_slug: str | None
    family_slug: str
    race: str
    model_gender: str
    image_part: str
    slot: str | None
    dye_state: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def is_gearsets_image_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == GEARSETS_IMAGE_HOST and parsed.path.lower().endswith(".png")


def parse_gearsets_image_url(url: str) -> GearsetImageMeta | None:
    if not is_gearsets_image_url(url):
        return None
    parsed = urlparse(url)
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 3:
        return None
    filename = parts[-1]
    stem = Path(filename).stem
    name_parts = stem.split("-")
    if len(name_parts) < 3:
        return None

    race = name_parts[0]
    gender = name_parts[1]
    if race not in RACE_SLUGS or gender not in GENDER_SLUGS:
        return None

    remainder = name_parts[2:]
    dye_state = "normal"
    if remainder and remainder[0] == "dyed":
        dye_state = "dyed"
        remainder = remainder[1:]
    if len(remainder) != 1 or remainder[0] not in GEAR_IMAGE_PARTS:
        return None

    image_part = remainder[0]
    source_type = parts[0]
    gearset_slug = parts[-2]
    collection_slug = parts[-3] if len(parts) >= 3 else gearset_slug
    source_path = "/".join(parts[:-2])
    return GearsetImageMeta(
        source_type=source_type,
        source_path=source_path,
        collection_slug=collection_slug,
        gearset_slug=gearset_slug,
        role_slug=role_suffix(gearset_slug),
        family_slug=family_slug(gearset_slug),
        race=race,
        model_gender=gender,
        image_part=image_part,
        slot=image_part.upper() if image_part in {"head", "body", "hands", "legs", "feet"} else None,
        dye_state=dye_state,
    )


def generate_static_image_urls(
    seed_urls: list[str],
    races: tuple[str, ...] | list[str] | None = None,
    genders: tuple[str, ...] | list[str] | None = None,
    parts: tuple[str, ...] | list[str] | None = None,
    dye_states: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Generate fixed-range race/gender/part/dye URLs for discovered static image folders."""

    bases: set[str] = set()
    for url in seed_urls:
        meta = parse_gearsets_image_url(url)
        if not meta:
            continue
        parsed = urlparse(url)
        folder = parsed.path.rsplit("/", 1)[0].strip("/")
        bases.add(f"https://{GEARSETS_IMAGE_HOST}/{folder}")

    urls: list[str] = []
    seen: set[str] = set()
    races = tuple(races or RACE_SLUGS)
    genders = tuple(genders or GENDER_SLUGS)
    parts = tuple(parts or GEAR_IMAGE_PARTS)
    dye_states = tuple(dye_states or DYE_STATES)
    validate_slug_range("race", races, RACE_SLUGS)
    validate_slug_range("gender", genders, GENDER_SLUGS)
    validate_slug_range("part", parts, GEAR_IMAGE_PARTS)
    validate_slug_range("dye_state", dye_states, DYE_STATES)
    for base in sorted(bases):
        for race in races:
            for gender in genders:
                for dye_state in dye_states:
                    for part in parts:
                        dye_piece = "-dyed" if dye_state == "dyed" else ""
                        url = f"{base}/{race}-{gender}{dye_piece}-{part}.png"
                        if url not in seen:
                            urls.append(url)
                            seen.add(url)
    return urls


def filter_same_family_urls(seed_urls: list[str], current_gearset_slug: str) -> list[str]:
    """Keep static image URLs that belong to the current gearset family and source path."""

    current_family = family_slug(current_gearset_slug)
    current_source_paths = {
        meta.source_path
        for url in seed_urls
        if (meta := parse_gearsets_image_url(url)) and meta.gearset_slug == current_gearset_slug
    }
    if not current_source_paths:
        current_source_paths = {
            meta.source_path
            for url in seed_urls
            if (meta := parse_gearsets_image_url(url)) and meta.family_slug == current_family
        }

    filtered: list[str] = []
    seen: set[str] = set()
    for url in seed_urls:
        meta = parse_gearsets_image_url(url)
        if not meta:
            continue
        if meta.family_slug != current_family:
            continue
        if current_source_paths and meta.source_path not in current_source_paths:
            continue
        if url not in seen:
            filtered.append(url)
            seen.add(url)
    return filtered


def page_exposed_model_range(seed_urls: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return race and gender ranges explicitly exposed by a page's static image URLs."""

    races: list[str] = []
    genders: list[str] = []
    for url in seed_urls:
        meta = parse_gearsets_image_url(url)
        if not meta:
            continue
        if meta.race not in races:
            races.append(meta.race)
        if meta.model_gender not in genders:
            genders.append(meta.model_gender)
    return tuple(races), tuple(genders)


def family_slug(gearset_slug: str) -> str:
    for suffix in ROLE_SUFFIXES:
        marker = f"-{suffix}"
        if gearset_slug.endswith(marker):
            return gearset_slug[: -len(marker)]
    return gearset_slug


def role_suffix(gearset_slug: str) -> str | None:
    for suffix in ROLE_SUFFIXES:
        if gearset_slug.endswith(f"-{suffix}"):
            return suffix
    return None


def validate_slug_range(label: str, selected: tuple[str, ...], allowed: tuple[str, ...]) -> None:
    invalid = [value for value in selected if value not in allowed]
    if invalid:
        raise ValueError(f"Invalid {label} values: {', '.join(invalid)}")


def generate_same_gearset_variant_urls(seed_urls: list[str]) -> list[str]:
    """Backward-compatible wrapper for older callers."""

    return generate_static_image_urls(seed_urls)
