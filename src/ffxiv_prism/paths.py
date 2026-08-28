from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR_NAME = PROJECT_ROOT.name
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
GAME_DATA_RAW_DIR = RAW_DIR / "game_data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
APPEARANCE_EQUIVALENCE_PATH = (
    PROCESSED_DIR / "piece_appearance_equivalence_huiji_exact_models.json"
)
INDEXES_DIR = DATA_DIR / "indexes"
SAMPLES_DIR = DATA_DIR / "samples"
VAR_DIR = PROJECT_ROOT / "var"
CACHE_DIR = VAR_DIR / "cache"
LOGS_DIR = VAR_DIR / "logs"

EORZEA_RAW_DIR = RAW_DIR / "eorzea_collection"
EORZEA_HTML_DIR = EORZEA_RAW_DIR / "html"
EORZEA_IMAGE_DIR = EORZEA_RAW_DIR / "images"
EORZEA_METADATA_DIR = EORZEA_RAW_DIR / "metadata"
EORZEA_GLAMOUR_DIR = EORZEA_RAW_DIR / "glamours"
EORZEA_GLAMOUR_HTML_DIR = EORZEA_GLAMOUR_DIR / "html"
EORZEA_GLAMOUR_IMAGE_DIR = EORZEA_GLAMOUR_DIR / "images"
EORZEA_GLAMOUR_METADATA_DIR = EORZEA_GLAMOUR_DIR / "metadata"


def ensure_project_dirs() -> None:
    """Create all local data directories expected by the collectors."""

    for path in (
        DATA_DIR,
        RAW_DIR,
        GAME_DATA_RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        INDEXES_DIR,
        SAMPLES_DIR,
        VAR_DIR,
        CACHE_DIR,
        LOGS_DIR,
        EORZEA_RAW_DIR,
        EORZEA_HTML_DIR,
        EORZEA_IMAGE_DIR,
        EORZEA_METADATA_DIR,
        EORZEA_GLAMOUR_DIR,
        EORZEA_GLAMOUR_HTML_DIR,
        EORZEA_GLAMOUR_IMAGE_DIR,
        EORZEA_GLAMOUR_METADATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def portable_path(path: str | Path | None) -> str | None:
    """Return a stable project-relative path when the value is inside this repo."""

    if path is None:
        return None
    path_text = str(path)
    if not path_text:
        return path_text
    if "://" in path_text:
        return path_text

    project_relative = path_text_after_project_dir(path_text)
    if project_relative is not None:
        return project_relative

    try:
        resolved = Path(path_text).resolve()
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return path_text


def resolve_project_path(path: str | Path | None) -> Path | None:
    """Resolve current or historical project-local path strings on this machine."""

    if path is None:
        return None
    path_text = str(path)
    if not path_text:
        return None
    if "://" in path_text:
        return Path(path_text)

    project_relative = path_text_after_project_dir(path_text)
    if project_relative is not None:
        return PROJECT_ROOT / project_relative

    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def portable_payload(value: Any) -> Any:
    """Normalize project-local paths inside JSON payloads before writing."""

    if isinstance(value, dict):
        return {key: portable_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_payload(item) for item in value]
    if isinstance(value, tuple):
        return [portable_payload(item) for item in value]
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, str):
        return portable_path(value)
    return value


def path_text_after_project_dir(path_text: str) -> str | None:
    normalized = path_text.replace("\\", "/")
    marker = f"/{PROJECT_DIR_NAME.lower()}/"
    lowered = normalized.lower()
    index = lowered.find(marker)
    if index < 0:
        return None
    relative = normalized[index + len(marker) :]
    return relative.lstrip("/")
