# 功能：提供 Eorzea Collection URL 标准化、slug 提取和本地文件名生成工具。

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse


BASE_URL = "https://ffxiv.eorzeacollection.com"


def absolute_url(url: str, base_url: str = BASE_URL) -> str:
    return urljoin(base_url, url)


def extract_slug(url: str, prefix: str) -> str:
    parsed = urlparse(url)
    pattern = rf"/{re.escape(prefix)}/([^/?#]+)"
    match = re.search(pattern, parsed.path)
    if match:
        return sanitize_slug(match.group(1))
    return short_hash(url)


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def sanitize_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or short_hash(value)


def extension_from_url(url: str, default: str = ".bin") -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".html", ".json"}:
        return suffix
    return default

