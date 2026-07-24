"""
Photo indexing engine for Smart Photo Archive.

Features:
- Recursive photo discovery
- SHA-256 hashing
- EXIF metadata extraction
- Thumbnail generation
- SQLite persistence
- Incremental indexing
- Safe database schema migration
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image, UnidentifiedImageError

from core.config import (
    DATABASE_PATH,
    SUPPORTED_EXTENSIONS,
    THUMBNAIL_DIR,
    THUMBNAIL_SIZE,
)
from core.hashing import calculate_file_hash
from core.metadata import extract_metadata


METADATA_VERSION = 1

ProgressCallback = Callable[[int, int, Path, str], None]


@dataclass(slots=True)
class IndexResult:
    discovered: int = 0
    indexed: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def normalize_path(path: str | Path) -> str:
    """Return an absolute normalized path."""
    return str(
        Path(path)
        .expanduser()
        .resolve()
    )


def get_database_connection() -> sqlite3.Connection:
    """Open and configure the SQLite database."""

    database_path = Path(DATABASE_PATH)

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        database_path
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA journal_mode = WAL"
    )

    connection.execute(
        "PRAGMA synchronous = NORMAL"
    )

    return connection


# =========================================================
# DATABASE SCHEMA / MIGRATIONS
# =========================================================

def get_existing_columns(
    connection: sqlite3.Connection,
) -> set[str]:
    """Return all column names currently in the photos table."""

    rows = connection.execute(
        "PRAGMA table_info(photos)"
    ).fetchall()

    return {
        row["name"]
        for row in rows
    }


def migrate_database(
    connection: sqlite3.Connection,
) -> None:
    """
    Add newer metadata columns to an existing database safely.

    Existing rows and photo IDs remain untouched.
    """

    columns = get_existing_columns(
        connection
    )

    migrations: dict[str, str] = {
        "camera_make":
            "ALTER TABLE photos ADD COLUMN camera_make TEXT",

        "camera_model":
            "ALTER TABLE photos ADD COLUMN camera_model TEXT",

        "lens_model":
            "ALTER TABLE photos ADD COLUMN lens_model TEXT",

        "iso":
            "ALTER TABLE photos ADD COLUMN iso INTEGER",

        "aperture":
            "ALTER TABLE photos ADD COLUMN aperture REAL",

        "shutter_speed":
            "ALTER TABLE photos ADD COLUMN shutter_speed TEXT",

        "focal_length":
            "ALTER TABLE photos ADD COLUMN focal_length REAL",

        "date_taken":
            "ALTER TABLE photos ADD COLUMN date_taken TEXT",

        "orientation":
            "ALTER TABLE photos ADD COLUMN orientation INTEGER",

        "gps_latitude":
            "ALTER TABLE photos ADD COLUMN gps_latitude REAL",

        "gps_longitude":
            "ALTER TABLE photos ADD COLUMN gps_longitude REAL",

        "gps_altitude":
            "ALTER TABLE photos ADD COLUMN gps_altitude REAL",

        "metadata_version":
            """
            ALTER TABLE photos
            ADD COLUMN metadata_version INTEGER NOT NULL DEFAULT 0
            """,
    }

    for column_name, sql in migrations.items():

        if column_name not in columns:
            connection.execute(sql)

            print(
                f"Database migration: added {column_name}"
            )

    connection.commit()


def initialize_index_database() -> None:
    """
    Create the photos table when necessary and migrate old schemas.
    """

    with get_database_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                file_path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                file_extension TEXT NOT NULL,

                file_size INTEGER NOT NULL,
                modified_time REAL NOT NULL,
                file_hash TEXT,

                width INTEGER,
                height INTEGER,
                image_format TEXT,
                color_mode TEXT,

                thumbnail_path TEXT,

                status TEXT NOT NULL DEFAULT 'indexed',
                error_message TEXT,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                camera_make TEXT,
                camera_model TEXT,
                lens_model TEXT,

                iso INTEGER,
                aperture REAL,
                shutter_speed TEXT,
                focal_length REAL,

                date_taken TEXT,
                orientation INTEGER,

                gps_latitude REAL,
                gps_longitude REAL,
                gps_altitude REAL,

                metadata_version INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        migrate_database(
            connection
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_photos_file_hash
            ON photos(file_hash)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_photos_date_taken
            ON photos(date_taken)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_photos_camera_model
            ON photos(camera_model)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_photos_status
            ON photos(status)
            """
        )

        connection.commit()


# =========================================================
# PHOTO DISCOVERY
# =========================================================

def discover_images(
    folder: str | Path,
) -> Iterator[Path]:
    """Recursively discover supported image files."""

    root = Path(folder).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {root}"
        )

    supported = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in SUPPORTED_EXTENSIONS
    }

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in supported:
            yield path


# =========================================================
# EXISTING PHOTO LOOKUP
# =========================================================

def get_existing_photo(
    connection: sqlite3.Connection,
    file_path: str,
) -> sqlite3.Row | None:
    """Return an existing photo row by path."""

    return connection.execute(
        """
        SELECT *
        FROM photos
        WHERE file_path = ?
        """,
        (file_path,),
    ).fetchone()


def file_is_unchanged(
    existing: sqlite3.Row,
    file_size: int,
    modified_time: float,
) -> bool:
    """
    Return True when the file does not require reprocessing.

    Photos using an older metadata version are refreshed once.
    """

    return (
        existing["file_size"] == file_size
        and existing["modified_time"] == modified_time
        and existing["status"] == "indexed"
        and existing["metadata_version"] >= METADATA_VERSION
    )


# =========================================================
# THUMBNAILS
# =========================================================

def build_thumbnail_path(
    file_hash: str,
) -> Path:
    """Return a deterministic thumbnail path."""

    thumbnail_directory = Path(
        THUMBNAIL_DIR
    )

    thumbnail_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        thumbnail_directory
        / f"{file_hash}.jpg"
    )


def generate_thumbnail(
    image_path: Path,
    file_hash: str,
) -> Path:
    """Create a thumbnail if one does not already exist."""

    thumbnail_path = build_thumbnail_path(
        file_hash
    )

    if thumbnail_path.exists():
        return thumbnail_path

    with Image.open(image_path) as image:

        image.thumbnail(
            THUMBNAIL_SIZE
        )

        if image.mode not in (
            "RGB",
            "L",
        ):
            image = image.convert(
                "RGB"
            )

        image.save(
            thumbnail_path,
            format="JPEG",
            quality=85,
            optimize=True,
        )

    return thumbnail_path


# =========================================================
# METADATA HELPERS
# =========================================================

def get_gps_values(
    metadata: dict,
) -> tuple[
    float | None,
    float | None,
    float | None,
]:
    """Extract latitude, longitude, and altitude."""

    gps = metadata.get("gps")

    if not gps:
        return (
            None,
            None,
            None,
        )

    return (
        gps.get("latitude"),
        gps.get("longitude"),
        gps.get("altitude"),
    )


# =========================================================
# DATABASE WRITES
# =========================================================

def save_photo_record(
    connection: sqlite3.Connection,
    *,
    image_path: Path,
    file_size: int,
    modified_time: float,
    file_hash: str,
    metadata: dict,
    thumbnail_path: Path,
) -> bool:
    """
    Insert or update a photo.

    Returns:
        False if inserted.
        True if updated.
    """

    normalized_path = normalize_path(
        image_path
    )

    existing = get_existing_photo(
        connection,
        normalized_path,
    )

    timestamp = utc_now()

    (
        gps_latitude,
        gps_longitude,
        gps_altitude,
    ) = get_gps_values(
        metadata
    )

    common_values = (
        image_path.name,
        image_path.suffix.lower(),

        file_size,
        modified_time,
        file_hash,

        metadata.get("width"),
        metadata.get("height"),
        metadata.get("image_format"),
        metadata.get("color_mode"),

        normalize_path(
            thumbnail_path
        ),

        metadata.get("camera_make"),
        metadata.get("camera_model"),
        metadata.get("lens_model"),

        metadata.get("iso"),
        metadata.get("aperture"),
        metadata.get("shutter_speed"),
        metadata.get("focal_length"),

        metadata.get("date_taken"),
        metadata.get("orientation"),

        gps_latitude,
        gps_longitude,
        gps_altitude,
    )

    if existing is None:

        connection.execute(
            """
            INSERT INTO photos (
                file_path,
                file_name,
                file_extension,

                file_size,
                modified_time,
                file_hash,

                width,
                height,
                image_format,
                color_mode,

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

                metadata_version,

                status,
                error_message,

                created_at,
                updated_at
            )

            VALUES (
                ?, ?, ?,

                ?, ?, ?,

                ?, ?, ?, ?,

                ?,

                ?, ?, ?,

                ?, ?, ?, ?,

                ?, ?,

                ?, ?, ?,

                ?,

                'indexed',
                NULL,

                ?, ?
            )
            """,
            (
                normalized_path,
                *common_values,
                METADATA_VERSION,
                timestamp,
                timestamp,
            ),
        )

        return False

    connection.execute(
        """
        UPDATE photos
        SET
            file_name = ?,
            file_extension = ?,

            file_size = ?,
            modified_time = ?,
            file_hash = ?,

            width = ?,
            height = ?,
            image_format = ?,
            color_mode = ?,

            thumbnail_path = ?,

            camera_make = ?,
            camera_model = ?,
            lens_model = ?,

            iso = ?,
            aperture = ?,
            shutter_speed = ?,
            focal_length = ?,

            date_taken = ?,
            orientation = ?,

            gps_latitude = ?,
            gps_longitude = ?,
            gps_altitude = ?,

            metadata_version = ?,

            status = 'indexed',
            error_message = NULL,

            updated_at = ?

        WHERE file_path = ?
        """,
        (
            *common_values,
            METADATA_VERSION,
            timestamp,
            normalized_path,
        ),
    )

    return True


# =========================================================
# INDEXING
# =========================================================

def index_photo(
    connection: sqlite3.Connection,
    image_path: Path,
) -> str:
    """Index one photo."""

    normalized_path = normalize_path(
        image_path
    )

    stat = image_path.stat()

    existing = get_existing_photo(
        connection,
        normalized_path,
    )

    if (
        existing is not None
        and file_is_unchanged(
            existing,
            stat.st_size,
            stat.st_mtime,
        )
    ):
        return "skipped"

    file_hash = calculate_file_hash(
        image_path
    )

    metadata = extract_metadata(
        image_path
    )

    thumbnail_path = generate_thumbnail(
        image_path,
        file_hash,
    )

    was_updated = save_photo_record(
        connection,
        image_path=image_path,
        file_size=stat.st_size,
        modified_time=stat.st_mtime,
        file_hash=file_hash,
        metadata=metadata,
        thumbnail_path=thumbnail_path,
    )

    if was_updated:
        return "updated"

    return "indexed"


def index_folder(
    folder: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> IndexResult:
    """Index all supported photos in a directory."""

    initialize_index_database()

    images = list(
        discover_images(folder)
    )

    result = IndexResult(
        discovered=len(images)
    )

    with get_database_connection() as connection:

        for position, image_path in enumerate(
            images,
            start=1,
        ):

            try:
                status = index_photo(
                    connection,
                    image_path,
                )

                if status == "indexed":
                    result.indexed += 1

                elif status == "updated":
                    result.updated += 1

                elif status == "skipped":
                    result.skipped += 1

                connection.commit()

            except (
                FileNotFoundError,
                PermissionError,
                UnidentifiedImageError,
                OSError,
                ValueError,
                sqlite3.Error,
            ) as error:

                connection.rollback()

                result.failed += 1
                status = "failed"

                print(
                    f"ERROR indexing "
                    f"{image_path.name}: {error}"
                )

            if progress_callback is not None:
                progress_callback(
                    position,
                    len(images),
                    image_path,
                    status,
                )

    return result


# =========================================================
# CLI OUTPUT
# =========================================================

def print_progress(
    current: int,
    total: int,
    image_path: Path,
    status: str,
) -> None:
    """Print indexing progress."""

    print(
        f"[{current}/{total}] "
        f"{status.upper():8} "
        f"{image_path.name}"
    )


def main() -> None:
    """Run the interactive photo indexer."""

    print("=" * 60)
    print(
        "Smart Photo Archive - Photo Indexer"
    )
    print("=" * 60)

    folder = input(
        "Enter the folder containing your photos: "
    ).strip().strip('"')

    try:
        result = index_folder(
            folder,
            progress_callback=print_progress,
        )

    except (
        FileNotFoundError,
        NotADirectoryError,
    ) as error:

        print(
            f"Error: {error}"
        )

        return

    print()
    print("Indexing complete")
    print(
        f"Discovered: {result.discovered}"
    )
    print(
        f"New:        {result.indexed}"
    )
    print(
        f"Updated:    {result.updated}"
    )
    print(
        f"Skipped:    {result.skipped}"
    )
    print(
        f"Failed:     {result.failed}"
    )


if __name__ == "__main__":
    main()
