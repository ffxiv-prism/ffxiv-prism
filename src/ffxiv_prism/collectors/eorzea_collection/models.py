# 功能：定义 Eorzea Collection 采集过程中使用的结构化记录。

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GearPieceRecord:
    slot: str
    name: str
    source_item_url: str | None = None
    glamours_using_piece_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageCandidate:
    source_image_url: str
    source_page_url: str
    image_kind: str
    slot: str | None = None
    gearset_slug: str | None = None
    glamour_slug: str | None = None
    piece_name: str | None = None
    model_gender: str | None = None
    race: str | None = None
    dye_state: str | None = None
    source_type: str | None = None
    source_path: str | None = None
    collection_slug: str | None = None
    role_slug: str | None = None
    family_slug: str | None = None
    image_part: str | None = None
    alt_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GearsetRecord:
    source_page_url: str
    slug: str
    title: str | None = None
    content_type: str | None = None
    item_level: int | None = None
    equip_level: int | None = None
    dyeable: bool | None = None
    roles: list[str] = field(default_factory=list)
    model_genders: list[str] = field(default_factory=list)
    pieces: list[GearPieceRecord] = field(default_factory=list)
    images: list[ImageCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pieces"] = [piece.to_dict() for piece in self.pieces]
        payload["images"] = [image.to_dict() for image in self.images]
        return payload


@dataclass(frozen=True)
class GlamourVariantRecord:
    source_page_url: str
    slug: str
    title: str | None
    genders: list[str]
    races: list[str]
    jobs: list[str]
    images: list[ImageCandidate]
    related_piece_name: str | None = None
    related_piece_slot: str | None = None
    related_piece_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["images"] = [image.to_dict() for image in self.images]
        return payload


@dataclass(frozen=True)
class DownloadedImageRecord:
    source_image_url: str
    source_page_url: str
    local_path: str
    sha256: str
    byte_size: int
    from_cache: bool
    image_kind: str
    slot: str | None = None
    gearset_slug: str | None = None
    glamour_slug: str | None = None
    piece_name: str | None = None
    model_gender: str | None = None
    race: str | None = None
    dye_state: str | None = None
    source_type: str | None = None
    source_path: str | None = None
    collection_slug: str | None = None
    role_slug: str | None = None
    family_slug: str | None = None
    image_part: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
