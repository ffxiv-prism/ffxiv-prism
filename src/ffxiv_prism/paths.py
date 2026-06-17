# 功能：集中管理项目根目录、数据目录、缓存目录和日志目录路径。

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"
SAMPLES_DIR = DATA_DIR / "samples"
VAR_DIR = PROJECT_ROOT / "var"
CACHE_DIR = VAR_DIR / "cache"
LOGS_DIR = VAR_DIR / "logs"

EORZEA_RAW_DIR = RAW_DIR / "eorzea_collection"
EORZEA_HTML_DIR = EORZEA_RAW_DIR / "html"
EORZEA_IMAGE_DIR = EORZEA_RAW_DIR / "images"
EORZEA_METADATA_DIR = EORZEA_RAW_DIR / "metadata"


def ensure_project_dirs() -> None:
    """Create all local data directories expected by the collectors."""

    for path in (
        DATA_DIR,
        RAW_DIR,
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
    ):
        path.mkdir(parents=True, exist_ok=True)

