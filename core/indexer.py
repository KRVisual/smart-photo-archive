"""
Smart Photo Archive - Photo Indexer

MVP responsibilities:
- Discover supported images
- Calculate SHA-256 hashes
- Skip photos already indexed
- Extract metadata / EXIF
- Store photo records in SQLite
- Maintain the schema required by the AI embedding/search pipeline

Moved/missing file detection is intentionally NOT implemented yet.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import config
from core.hashing import calculate_file_hash
from core.metadata import extract_metadata


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
}


@dataclass
class IndexResult:
    discovered: int = 0
    indexed: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0


def _get_config_value(*names: str, default=None):
    """
    Try several possible config constant names.

    This keeps the indexer compatible with the current project while
    avoiding unnecessary changes to core/config.py.
    """
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)

    return default


def get_photo_directory() -> Path:
    configured = _get_config_value(
        "PHOTO_DIR",
        "PHOTOS_DIR",
        "PHOTO_DIRECTORY",
        "PHOTOS_DIRECTORY",
        default="photos",
    )

    return Path(configured).expanduser().resolve()


def get_database_path() -> Path:
    configured = _get_config_value(
        "DATABASE_PATH",
        "DB_PATH",
        "DATABASE_FILE",
        "DB_FILE",
        default="photos.db",
    )

    return Path(configured).expanduser().resolve()


def connect_database(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    """
    Create the MVP photos table.

    Important fields for ai/clip_embed.py and ai/search.py include:
        id
        file_path
        file_name
        status
        width
        height
        thumbnail_path
    """

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            file_path TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,

            sha256 TEXT NOT NULL,

            status TEXT NOT NULL DEFAULT 'indexed',

            width INTEGER,
            height INTEGER,

            thumbnail_path TEXT,

            camera_make TEXT,
            camera_model TEXT,
            lens_model TEXT,

            iso REAL,
            aperture REAL,
            shutter_speed TEXT,
            focal_length REAL,

            date_taken TEXT,
            orientation INTEGER,

            gps_latitude REAL,
            gps_longitude REAL,
            gps_altitude REAL,

            image_format TEXT,
            color_mode TEXT,

            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_photos_sha256
        ON photos(sha256)
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_photos_status
        ON photos(status)
        """
    )

    connection.commit()


def discover_photos(photo_directory: Path) -> list[Path]:
    if not photo_directory.exists():
        raise FileNotFoundError(
            f"Photo directory does not exist: {photo_directory}"
        )

    if not photo_directory.is_dir():
        raise NotADirectoryError(
            f"Photo path is not a directory: {photo_directory}"
        )

    photos: list[Path] = []

    for path in photo_directory.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            photos.append(path.resolve())

    photos.sort(key=lambda p: str(p).lower())

    return photos


def find_photo_by_hash(
    connection: sqlite3.Connection,
    file_hash: str,
):
    return connection.execute(
        """
        SELECT *
        FROM photos
        WHERE sha256 = ?
        LIMIT 1
        """,
        (file_hash,),
    ).fetchone()


def find_photo_by_path(
    connection: sqlite3.Connection,
    file_path: Path,
):
    return connection.execute(
        """
        SELECT *
        FROM photos
        WHERE file_path = ?
        LIMIT 1
        """,
        (str(file_path),),
    ).fetchone()


def metadata_value(metadata: Any, key: str, default=None):
    """
    Support metadata returned either as:
    - dict
    - dataclass/object
    """

    if metadata is None:
        return default

    if isinstance(metadata, dict):
        return metadata.get(key, default)

    return getattr(metadata, key, default)


def insert_photo(
    connection: sqlite3.Connection,
    photo_path: Path,
    file_hash: str,
    metadata: Any,
) -> None:

    values = {
        "file_path": str(photo_path),
        "file_name": photo_path.name,
        "sha256": file_hash,
        "status": "indexed",

        "width": metadata_value(metadata, "width"),
        "height": metadata_value(metadata, "height"),

        "thumbnail_path": metadata_value(
            metadata,
            "thumbnail_path",
        ),

        "camera_make": metadata_value(
            metadata,
            "camera_make",
        ),

        "camera_model": metadata_value(
            metadata,
            "camera_model",
        ),

        "lens_model": metadata_value(
            metadata,
            "lens_model",
        ),

        "iso": metadata_value(
            metadata,
            "iso",
        ),

        "aperture": metadata_value(
            metadata,
            "aperture",
        ),

        "shutter_speed": metadata_value(
            metadata,
            "shutter_speed",
        ),

        "focal_length": metadata_value(
            metadata,
            "focal_length",
        ),

        "date_taken": metadata_value(
            metadata,
            "date_taken",
        ),

        "orientation": metadata_value(
            metadata,
            "orientation",
        ),

        "gps_latitude": metadata_value(
            metadata,
            "gps_latitude",
        ),

        "gps_longitude": metadata_value(
            metadata,
            "gps_longitude",
        ),

        "gps_altitude": metadata_value(
            metadata,
            "gps_altitude",
        ),

        "image_format": metadata_value(
            metadata,
            "image_format",
        ),

        "color_mode": metadata_value(
            metadata,
            "color_mode",
        ),
    }

    connection.execute(
        """
        INSERT INTO photos (
            file_path,
            file_name,
            sha256,
            status,

            width,
            height,

            thumbnail_path,

            camera_make,
            camera_model,
            lens_model,

            iso,
            aperture,
            shutter_speed,
            focal_length,

            date_taken,
            orientation,

            gps_latitude,
            gps_longitude,
            gps_altitude,

            image_format,
            color_mode
        )
        VALUES (
            :file_path,
            :file_name,
            :sha256,
            :status,

            :width,
            :height,

            :thumbnail_path,

            :camera_make,
            :camera_model,
            :lens_model,

            :iso,
            :aperture,
            :shutter_speed,
            :focal_length,

            :date_taken,
            :orientation,

            :gps_latitude,
            :gps_longitude,
            :gps_altitude,

            :image_format,
            :color_mode
        )
        """,
        values,
    )


def index_photos(
    photo_directory: Path | None = None,
    database_path: Path | None = None,
) -> IndexResult:

    if photo_directory is None:
        photo_directory = get_photo_directory()

    if database_path is None:
        database_path = get_database_path()

    result = IndexResult()

    photos = discover_photos(photo_directory)

    result.discovered = len(photos)

    connection = connect_database(database_path)

    try:
        create_schema(connection)

        for number, photo_path in enumerate(photos, start=1):

            print(
                f"[{number}/{result.discovered}] "
                f"{photo_path.name}"
            )

            try:
                file_hash = calculate_file_hash(photo_path)

                existing_hash = find_photo_by_hash(
                    connection,
                    file_hash,
                )

                if existing_hash is not None:
                    result.skipped += 1

                    print("    SKIPPED - already indexed")
                    continue

                existing_path = find_photo_by_path(
                    connection,
                    photo_path,
                )

                if existing_path is not None:
                    result.skipped += 1

                    print("    SKIPPED - path already indexed")
                    continue

                metadata = extract_metadata(photo_path)

                insert_photo(
                    connection,
                    photo_path,
                    file_hash,
                    metadata,
                )

                connection.commit()

                result.indexed += 1

                print("    INDEXED")

            except Exception as exc:
                connection.rollback()

                result.failed += 1

                print(
                    f"    FAILED - "
                    f"{type(exc).__name__}: {exc}"
                )

    finally:
        connection.close()

    return result


def main() -> None:
    photo_directory = get_photo_directory()
    database_path = get_database_path()

    print()
    print("Smart Photo Archive")
    print("===================")
    print()

    print(f"Photo directory: {photo_directory}")
    print(f"Database:        {database_path}")

    print()
    print("Starting photo index...")
    print()

    try:
        result = index_photos(
            photo_directory=photo_directory,
            database_path=database_path,
        )

    except Exception as exc:
        print()
        print("Indexer failed.")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        return

    print()
    print("===================")
    print("Index complete")
    print("===================")

    print(f"Discovered: {result.discovered}")
    print(f"Indexed:    {result.indexed}")
    print(f"Updated:    {result.updated}")
    print(f"Skipped:    {result.skipped}")
    print(f"Failed:     {result.failed}")

    print()


if __name__ == "__main__":
    main()
