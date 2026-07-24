"""
Central configuration for Smart Photo Archive.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = PROJECT_ROOT / "photos.db"

CACHE_DIR = PROJECT_ROOT / "cache"
THUMBNAIL_DIR = CACHE_DIR / "thumbnails"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

THUMBNAIL_SIZE = (400, 400)
THUMBNAIL_QUALITY = 85

HASH_ALGORITHM = "sha256"
HASH_CHUNK_SIZE = 1024 * 1024

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED_MODEL = "laion2b_s34b_b79k"

DEFAULT_SCAN_DIRECTORY = PROJECT_ROOT / "photos"

LOG_LEVEL = "INFO"


def initialize_directories() -> None:
    """Create required application directories."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_SCAN_DIRECTORY.mkdir(parents=True, exist_ok=True)


def print_configuration() -> None:
    """Print the active configuration."""

    print("=" * 60)
    print("Smart Photo Archive Configuration")
    print("=" * 60)
    print(f"Project root          : {PROJECT_ROOT}")
    print(f"Database path         : {DATABASE_PATH}")
    print(f"Cache directory       : {CACHE_DIR}")
    print(f"Thumbnail directory   : {THUMBNAIL_DIR}")
    print(f"Default photo folder  : {DEFAULT_SCAN_DIRECTORY}")
    print(f"Thumbnail size        : {THUMBNAIL_SIZE}")
    print(f"Hash algorithm        : {HASH_ALGORITHM}")
    print(f"Hash chunk size       : {HASH_CHUNK_SIZE} bytes")
    print(f"CLIP model            : {CLIP_MODEL_NAME}")
    print(f"CLIP pretrained model : {CLIP_PRETRAINED_MODEL}")


if __name__ == "__main__":
    initialize_directories()
    print_configuration()
