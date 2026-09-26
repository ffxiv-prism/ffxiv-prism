from __future__ import annotations

import json
import os
import tempfile
import sqlite3
from pathlib import Path
from typing import Any

from ffxiv_prism.jsonl import read_jsonl, write_json_atomic, replace_with_retry
from ffxiv_prism.paths import EORZEA_METADATA_DIR, PROCESSED_DIR, ensure_project_dirs, portable_path, resolve_project_path
from ffxiv_prism.collectors.eorzea_collection.url_utils import sanitize_slug


DEFAULT_DB_PATH = PROCESSED_DIR / "eorzea.sqlite"
DEFAULT_GEARSETS_PATH = EORZEA_METADATA_DIR / "gearset_details_raw.jsonl"
DEFAULT_GEAR_PIECES_PATH = EORZEA_METADATA_DIR / "gear_pieces_raw.jsonl"
DEFAULT_IMAGES_QA_PATH = PROCESSED_DIR / "eorzea_images_qa_index.jsonl"
DEFAULT_SUMMARY_PATH = PROCESSED_DIR / "eorzea_sqlite_summary.json"


def canonical_gearset_slug(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    return sanitize_slug(str(value))


def build_eorzea_sqlite(
    db_path: Path | None = None,
    gearsets_path: Path | None = None,
    gear_pieces_path: Path | None = None,
    images_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    ensure_project_dirs()
    db_path = db_path or DEFAULT_DB_PATH
    gearsets_path = gearsets_path or DEFAULT_GEARSETS_PATH
    gear_pieces_path = gear_pieces_path or DEFAULT_GEAR_PIECES_PATH
    images_path = images_path or DEFAULT_IMAGES_QA_PATH
    summary_path = summary_path or DEFAULT_SUMMARY_PATH

    gearsets = dedupe_rows(read_jsonl(gearsets_path), "slug")
    gear_pieces = read_jsonl(gear_pieces_path)
    images = read_jsonl(images_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{db_path.name}.", suffix=".tmp", dir=db_path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    conn = sqlite3.connect(temporary)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        create_schema(conn)
        gearset_ids = insert_gearsets(conn, gearsets)
        piece_ids = insert_gear_pieces(conn, gear_pieces, gearset_ids)
        image_ids = insert_images(conn, images, gearset_ids)
        link_count = insert_gear_piece_images(conn, gear_pieces, images, piece_ids, image_ids)
        create_indexes(conn)
        summary = summarize_sqlite(conn)
        summary.update(
            {
                "database": portable_path(db_path),
                "gearsets_source": portable_path(gearsets_path),
                "gear_pieces_source": portable_path(gear_pieces_path),
                "images_source": portable_path(images_path),
                "gear_piece_image_links": link_count,
            }
        )
        conn.commit()
    except BaseException:
        conn.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        conn.close()
    try:
        replace_with_retry(temporary, db_path)
    finally:
        temporary.unlink(missing_ok=True)

    # The database is replaced immediately before this summary is written.
    # On Windows, opening the existing summary in place can briefly fail with
    # EINVAL while filesystem scanners observe the adjacent database replace.
    # Writing a sibling temporary and atomically replacing the small summary
    # avoids that transient lifecycle failure.
    write_json_atomic(summary_path, summary)
    return summary


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE gearsets (
            gearset_id INTEGER PRIMARY KEY,
            slug TEXT NOT NULL UNIQUE,
            title TEXT,
            source_page_url TEXT,
            content_type TEXT,
            item_level INTEGER,
            equip_level INTEGER,
            dyeable INTEGER,
            present_slots_json TEXT NOT NULL,
            missing_slots_json TEXT NOT NULL
        );

        CREATE TABLE gearset_roles (
            gearset_id INTEGER NOT NULL REFERENCES gearsets(gearset_id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            PRIMARY KEY (gearset_id, role)
        );

        CREATE TABLE gearset_model_genders (
            gearset_id INTEGER NOT NULL REFERENCES gearsets(gearset_id) ON DELETE CASCADE,
            model_gender TEXT NOT NULL,
            PRIMARY KEY (gearset_id, model_gender)
        );

        CREATE TABLE gear_pieces (
            piece_id INTEGER PRIMARY KEY,
            gearset_id INTEGER NOT NULL REFERENCES gearsets(gearset_id) ON DELETE CASCADE,
            gearset_slug TEXT NOT NULL,
            slot TEXT NOT NULL,
            name TEXT NOT NULL,
            name_en TEXT,
            name_zh TEXT,
            name_zh_source TEXT,
            name_zh_match_status TEXT,
            source_item_url TEXT,
            game_item_id INTEGER,
            game_item_ids_json TEXT NOT NULL,
            name_zh_candidates_json TEXT NOT NULL,
            occupied_slots_json TEXT NOT NULL,
            blocked_slots_json TEXT NOT NULL,
            official_primary_slots_json TEXT NOT NULL,
            official_occupied_slots_json TEXT NOT NULL,
            official_blocked_slots_json TEXT NOT NULL,
            official_slot_category_id INTEGER,
            official_item_ui_category TEXT,
            official_equip_slot_source TEXT,
            equip_conflict_note TEXT,
            equip_conflict_source TEXT,
            UNIQUE (gearset_slug, slot, source_item_url, name)
        );

        CREATE TABLE images (
            image_row_id INTEGER PRIMARY KEY,
            gearset_id INTEGER REFERENCES gearsets(gearset_id) ON DELETE SET NULL,
            gearset_slug TEXT,
            source_image_url TEXT NOT NULL UNIQUE,
            source_page_url TEXT,
            local_path TEXT NOT NULL,
            sha256 TEXT,
            byte_size INTEGER,
            actual_byte_size INTEGER,
            width INTEGER,
            height INTEGER,
            image_format TEXT,
            image_kind TEXT,
            image_part TEXT,
            slot TEXT,
            model_gender TEXT,
            race TEXT,
            dye_state TEXT,
            source_type TEXT,
            source_path TEXT,
            collection_slug TEXT,
            family_slug TEXT,
            role_slug TEXT,
            qa_status TEXT,
            quality_flags_json TEXT NOT NULL,
            file_exists INTEGER NOT NULL,
            readable_image INTEGER NOT NULL,
            sha_duplicate_count INTEGER,
            path_reference_count INTEGER
        );

        CREATE TABLE gear_piece_images (
            piece_id INTEGER NOT NULL REFERENCES gear_pieces(piece_id) ON DELETE CASCADE,
            image_row_id INTEGER NOT NULL REFERENCES images(image_row_id) ON DELETE CASCADE,
            link_type TEXT NOT NULL,
            PRIMARY KEY (piece_id, image_row_id)
        );
        """
    )


def insert_gearsets(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, int]:
    gearset_ids: dict[str, int] = {}
    for row in rows:
        slug = canonical_gearset_slug(row.get("slug"))
        if not slug:
            continue
        cursor = conn.execute(
            """
            INSERT INTO gearsets (
                slug, title, source_page_url, content_type, item_level, equip_level,
                dyeable, present_slots_json, missing_slots_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                slug,
                row.get("title"),
                row.get("source_page_url"),
                row.get("content_type"),
                row.get("item_level"),
                row.get("equip_level"),
                bool_to_int(row.get("dyeable")),
                json_text(row.get("present_slots") or []),
                json_text(row.get("missing_slots") or []),
            ),
        )
        gearset_id = int(cursor.lastrowid)
        gearset_ids[slug] = gearset_id
        conn.executemany(
            "INSERT OR IGNORE INTO gearset_roles (gearset_id, role) VALUES (?, ?)",
            [(gearset_id, role) for role in row.get("roles") or []],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO gearset_model_genders (gearset_id, model_gender) VALUES (?, ?)",
            [(gearset_id, gender) for gender in row.get("model_genders") or []],
        )
    return gearset_ids


def insert_gear_pieces(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    gearset_ids: dict[str, int],
) -> dict[tuple[str, str, str | None, str], int]:
    piece_ids: dict[tuple[str, str, str | None, str], int] = {}
    for row in rows:
        gearset_slug = canonical_gearset_slug(row.get("gearset_slug"))
        slot = row.get("slot")
        name = row.get("name") or row.get("name_en")
        gearset_id = gearset_ids.get(gearset_slug)
        if not gearset_id or not gearset_slug or not slot or not name:
            continue
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO gear_pieces (
                gearset_id, gearset_slug, slot, name, name_en, name_zh, name_zh_source,
                name_zh_match_status, source_item_url, game_item_id, game_item_ids_json,
                name_zh_candidates_json, occupied_slots_json, blocked_slots_json,
                official_primary_slots_json, official_occupied_slots_json,
                official_blocked_slots_json, official_slot_category_id,
                official_item_ui_category, official_equip_slot_source,
                equip_conflict_note, equip_conflict_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gearset_id,
                gearset_slug,
                slot,
                name,
                row.get("name_en"),
                row.get("name_zh"),
                row.get("name_zh_source"),
                row.get("name_zh_match_status"),
                row.get("source_item_url"),
                row.get("game_item_id"),
                json_text(row.get("game_item_ids") or []),
                json_text(row.get("name_zh_candidates") or []),
                json_text(row.get("occupied_slots") or []),
                json_text(row.get("blocked_slots") or []),
                json_text(row.get("official_primary_slots") or []),
                json_text(row.get("official_occupied_slots") or []),
                json_text(row.get("official_blocked_slots") or []),
                row.get("official_slot_category_id"),
                row.get("official_item_ui_category"),
                row.get("official_equip_slot_source"),
                row.get("equip_conflict_note"),
                row.get("equip_conflict_source"),
            ),
        )
        piece_id = int(cursor.lastrowid) if cursor.lastrowid else find_piece_id(conn, gearset_slug, slot, row.get("source_item_url"), name)
        piece_ids[(gearset_slug, slot, row.get("source_item_url"), name)] = piece_id
    return piece_ids


def insert_images(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    gearset_ids: dict[str, int],
) -> dict[str, int]:
    image_ids: dict[str, int] = {}
    for row in rows:
        source_image_url = row.get("source_image_url")
        local_path = portable_path(resolve_project_path(row.get("local_path")))
        gearset_slug = canonical_gearset_slug(row.get("gearset_slug"))
        if not source_image_url or not local_path:
            continue
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO images (
                gearset_id, gearset_slug, source_image_url, source_page_url, local_path,
                sha256, byte_size, actual_byte_size, width, height, image_format,
                image_kind, image_part, slot, model_gender, race, dye_state,
                source_type, source_path, collection_slug, family_slug, role_slug,
                qa_status, quality_flags_json, file_exists, readable_image,
                sha_duplicate_count, path_reference_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gearset_ids.get(gearset_slug),
                gearset_slug,
                source_image_url,
                row.get("source_page_url"),
                local_path,
                row.get("sha256"),
                row.get("byte_size"),
                row.get("actual_byte_size"),
                row.get("width"),
                row.get("height"),
                row.get("image_format"),
                row.get("image_kind"),
                row.get("image_part"),
                row.get("slot"),
                row.get("model_gender"),
                row.get("race"),
                row.get("dye_state"),
                row.get("source_type"),
                row.get("source_path"),
                row.get("collection_slug"),
                row.get("family_slug"),
                row.get("role_slug"),
                row.get("qa_status"),
                json_text(row.get("quality_flags") or []),
                bool_to_int(row.get("file_exists")) or 0,
                bool_to_int(row.get("readable_image")) or 0,
                row.get("sha_duplicate_count"),
                row.get("path_reference_count"),
            ),
        )
        image_row_id = int(cursor.lastrowid) if cursor.lastrowid else find_image_row_id(conn, source_image_url)
        image_ids[source_image_url] = image_row_id
    return image_ids


def insert_gear_piece_images(
    conn: sqlite3.Connection,
    gear_pieces: list[dict[str, Any]],
    images: list[dict[str, Any]],
    piece_ids: dict[tuple[str, str, str | None, str], int],
    image_ids: dict[str, int],
) -> int:
    exact_by_gearset_slot: dict[tuple[str, str], list[int]] = {}
    occupied_by_gearset_slot: dict[tuple[str, str], list[int]] = {}

    for piece in gear_pieces:
        gearset_slug = canonical_gearset_slug(piece.get("gearset_slug"))
        slot = piece.get("slot")
        name = piece.get("name") or piece.get("name_en")
        if not gearset_slug or not slot or not name:
            continue
        piece_id = piece_ids.get((gearset_slug, slot, piece.get("source_item_url"), name))
        if not piece_id:
            continue
        occupied = (
            piece.get("official_occupied_slots") or []
            if piece.get("official_equip_slot_source")
            else piece.get("occupied_slots") or []
        )
        if not piece.get("official_equip_slot_source") or slot in occupied:
            exact_by_gearset_slot.setdefault((gearset_slug, slot), []).append(piece_id)
        for occupied_slot in occupied:
            occupied_by_gearset_slot.setdefault((gearset_slug, occupied_slot), []).append(piece_id)

    link_rows: list[tuple[int, int, str]] = []
    for image in images:
        gearset_slug = canonical_gearset_slug(image.get("gearset_slug"))
        slot = image.get("slot")
        source_image_url = image.get("source_image_url")
        if not gearset_slug or not slot or not source_image_url:
            continue
        image_row_id = image_ids.get(source_image_url)
        if not image_row_id:
            continue
        exact_piece_ids = exact_by_gearset_slot.get((gearset_slug, slot), [])
        if exact_piece_ids:
            link_rows.extend((piece_id, image_row_id, "exact_slot") for piece_id in exact_piece_ids)
            continue
        occupied_piece_ids = occupied_by_gearset_slot.get((gearset_slug, slot), [])
        link_rows.extend((piece_id, image_row_id, "occupied_slot") for piece_id in occupied_piece_ids)

    conn.executemany(
        "INSERT OR IGNORE INTO gear_piece_images (piece_id, image_row_id, link_type) VALUES (?, ?, ?)",
        link_rows,
    )
    return conn.execute("SELECT COUNT(*) FROM gear_piece_images").fetchone()[0]


def create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX idx_gear_pieces_name_en ON gear_pieces(name_en);
        CREATE INDEX idx_gear_pieces_name_zh ON gear_pieces(name_zh);
        CREATE INDEX idx_gear_pieces_slot ON gear_pieces(slot);
        CREATE INDEX idx_images_gearset_slot ON images(gearset_slug, slot);
        CREATE INDEX idx_images_sha256 ON images(sha256);
        CREATE INDEX idx_images_qa_status ON images(qa_status);
        CREATE INDEX idx_gear_piece_images_image ON gear_piece_images(image_row_id);

        CREATE VIEW v_piece_images AS
        SELECT
            gp.piece_id,
            gp.gearset_slug,
            gp.slot AS piece_slot,
            gp.name_en,
            gp.name_zh,
            i.image_row_id,
            i.source_image_url,
            i.local_path,
            i.slot AS image_slot,
            i.image_part,
            i.model_gender,
            i.race,
            i.dye_state,
            i.width,
            i.height,
            gpi.link_type
        FROM gear_piece_images gpi
        JOIN gear_pieces gp ON gp.piece_id = gpi.piece_id
        JOIN images i ON i.image_row_id = gpi.image_row_id;
        """
    )


def summarize_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
    scalar = lambda sql: conn.execute(sql).fetchone()[0]
    unlinked_slot_sql = """
        SELECT COUNT(*)
        FROM images i
        WHERE i.slot IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM gear_piece_images gpi
            WHERE gpi.image_row_id = i.image_row_id
        )
    """
    return {
        "gearsets": scalar("SELECT COUNT(*) FROM gearsets"),
        "gear_pieces": scalar("SELECT COUNT(*) FROM gear_pieces"),
        "images": scalar("SELECT COUNT(*) FROM images"),
        "ready_images": scalar("SELECT COUNT(*) FROM images WHERE qa_status = 'ready'"),
        "missing_or_invalid_images": scalar(
            "SELECT COUNT(*) FROM images WHERE qa_status != 'ready' OR qa_status IS NULL"
        ),
        "slot_images": scalar("SELECT COUNT(*) FROM images WHERE slot IS NOT NULL"),
        "full_preview_images": scalar("SELECT COUNT(*) FROM images WHERE slot IS NULL"),
        "linked_slot_images": scalar("SELECT COUNT(DISTINCT image_row_id) FROM gear_piece_images"),
        "unlinked_slot_images": scalar(unlinked_slot_sql),
        "slot_images_without_gearset": scalar("SELECT COUNT(*) FROM images WHERE slot IS NOT NULL AND gearset_id IS NULL"),
        "gear_piece_image_links": scalar("SELECT COUNT(*) FROM gear_piece_images"),
    }


def dedupe_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for row in rows:
        value = row.get(key)
        if value in seen:
            continue
        seen.add(value)
        deduped.append(row)
    return deduped


def find_piece_id(conn: sqlite3.Connection, gearset_slug: str, slot: str, source_item_url: str | None, name: str) -> int:
    row = conn.execute(
        """
        SELECT piece_id FROM gear_pieces
        WHERE gearset_slug = ? AND slot = ? AND source_item_url IS ? AND name = ?
        """,
        (gearset_slug, slot, source_item_url, name),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"piece was not inserted: {gearset_slug} {slot} {name}")
    return int(row[0])


def find_image_row_id(conn: sqlite3.Connection, source_image_url: str) -> int:
    row = conn.execute("SELECT image_row_id FROM images WHERE source_image_url = ?", (source_image_url,)).fetchone()
    if row is None:
        raise RuntimeError(f"image was not inserted: {source_image_url}")
    return int(row[0])


def bool_to_int(value: object) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def summary_to_text(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
