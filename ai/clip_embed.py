"""
CLIP embedding system for Smart Photo Archive.

Generates normalized CLIP image embeddings and stores them
inside the Smart Photo Archive SQLite database.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageOps

from core.config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED_MODEL,
    DATABASE_PATH,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_preprocess = None
_tokenizer = None


def utc_now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def get_clip_model():
    """
    Load CLIP once and reuse it.

    Returns:
        Tuple containing model, preprocessing pipeline, and tokenizer.
    """

    global _model
    global _preprocess
    global _tokenizer

    if _model is None:
        print(f"Loading CLIP on {DEVICE}...")
        print(f"Model: {CLIP_MODEL_NAME}")
        print(f"Pretrained: {CLIP_PRETRAINED_MODEL}")

        _model, _, _preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED_MODEL,
            device=DEVICE,
        )

        _tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        _model.eval()

        print("CLIP loaded successfully.")

    return _model, _preprocess, _tokenizer


def get_database_connection() -> sqlite3.Connection:
    """Open Smart Photo Archive's SQLite database."""

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_embedding_table() -> None:
    """Create the CLIP embedding table."""

    with get_database_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS photo_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                photo_id INTEGER NOT NULL UNIQUE,

                model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL,

                embedding BLOB NOT NULL,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                FOREIGN KEY(photo_id)
                    REFERENCES photos(id)
                    ON DELETE CASCADE
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_photo_embeddings_photo_id
            ON photo_embeddings(photo_id)
            """
        )


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding."""

    vector = vector.astype(np.float32)

    magnitude = np.linalg.norm(vector)

    if magnitude == 0:
        raise ValueError("Cannot normalize a zero-length embedding.")

    return vector / magnitude


def generate_image_embedding(
    image_path: str | Path,
) -> np.ndarray:
    """
    Generate a normalized CLIP embedding for an image.

    Args:
        image_path: Path to an image.

    Returns:
        Normalized float32 NumPy vector.
    """

    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {path}"
        )

    model, preprocess, _ = get_clip_model()

    with Image.open(path) as image:
        # Correct images that contain EXIF rotation information.
        image = ImageOps.exif_transpose(image)

        image = image.convert("RGB")

        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)

    with torch.inference_mode():
        features = model.encode_image(image_tensor)

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        )

    vector = (
        features
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return normalize_embedding(vector)


def generate_text_embedding(text: str) -> np.ndarray:
    """
    Convert natural-language text into a CLIP embedding.

    Example:
        "a city street at night"
    """

    query = text.strip()

    if not query:
        raise ValueError("Search text cannot be empty.")

    model, _, tokenizer = get_clip_model()

    tokens = tokenizer([query]).to(DEVICE)

    with torch.inference_mode():
        features = model.encode_text(tokens)

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        )

    vector = (
        features
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return normalize_embedding(vector)


def embedding_to_blob(
    embedding: np.ndarray,
) -> bytes:
    """Convert a NumPy embedding into SQLite bytes."""

    return embedding.astype(np.float32).tobytes()


def blob_to_embedding(
    blob: bytes,
    dimension: int,
) -> np.ndarray:
    """Restore an embedding from SQLite."""

    vector = np.frombuffer(
        blob,
        dtype=np.float32,
    )

    if len(vector) != dimension:
        raise ValueError(
            "Stored embedding dimension does not match "
            f"database value: expected {dimension}, "
            f"got {len(vector)}."
        )

    return vector.copy()


def embedding_exists(
    connection: sqlite3.Connection,
    photo_id: int,
) -> bool:
    """Check whether a photo already has a CLIP embedding."""

    model_key = (
        f"{CLIP_MODEL_NAME}:"
        f"{CLIP_PRETRAINED_MODEL}"
    )

    row = connection.execute(
        """
        SELECT id
        FROM photo_embeddings
        WHERE photo_id = ?
        AND model_name = ?
        """,
        (
            photo_id,
            model_key,
        ),
    ).fetchone()

    return row is not None


def save_embedding(
    connection: sqlite3.Connection,
    photo_id: int,
    embedding: np.ndarray,
) -> None:
    """Insert or update an image embedding."""

    timestamp = utc_now()

    model_key = (
        f"{CLIP_MODEL_NAME}:"
        f"{CLIP_PRETRAINED_MODEL}"
    )

    connection.execute(
        """
        INSERT INTO photo_embeddings (
            photo_id,
            model_name,
            dimension,
            embedding,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)

        ON CONFLICT(photo_id)
        DO UPDATE SET
            model_name = excluded.model_name,
            dimension = excluded.dimension,
            embedding = excluded.embedding,
            updated_at = excluded.updated_at
        """,
        (
            photo_id,
            model_key,
            int(embedding.shape[0]),
            embedding_to_blob(embedding),
            timestamp,
            timestamp,
        ),
    )


def embed_all_photos(
    force: bool = False,
) -> tuple[int, int, int]:
    """
    Generate embeddings for all indexed photos.

    Args:
        force:
            Rebuild embeddings even when they already exist.

    Returns:
        Tuple:
            generated, skipped, failed
    """

    initialize_embedding_table()

    generated = 0
    skipped = 0
    failed = 0

    with get_database_connection() as connection:

        photos = connection.execute(
            """
            SELECT id, file_path, file_name
            FROM photos
            WHERE status = 'indexed'
            ORDER BY id
            """
        ).fetchall()

        total = len(photos)

        print()
        print(f"Found {total} indexed photos.")
        print()

        if total == 0:
            return generated, skipped, failed

        # Load model before starting so any model error
        # occurs before processing individual images.
        get_clip_model()

        for position, photo in enumerate(
            photos,
            start=1,
        ):
            photo_id = photo["id"]
            file_path = photo["file_path"]
            file_name = photo["file_name"]

            if (
                not force
                and embedding_exists(
                    connection,
                    photo_id,
                )
            ):
                skipped += 1

                print(
                    f"[{position}/{total}] "
                    f"SKIPPED  {file_name}"
                )

                continue

            try:
                embedding = generate_image_embedding(
                    file_path
                )

                save_embedding(
                    connection,
                    photo_id,
                    embedding,
                )

                connection.commit()

                generated += 1

                print(
                    f"[{position}/{total}] "
                    f"EMBEDDED {file_name}"
                )

            except Exception as error:
                failed += 1

                print(
                    f"[{position}/{total}] "
                    f"FAILED   {file_name}"
                )

                print(
                    f"           {error}"
                )

    return generated, skipped, failed


def main() -> None:
    """Run CLIP embedding generation."""

    print("=" * 60)
    print("Smart Photo Archive - CLIP Embedding Generator")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    generated, skipped, failed = embed_all_photos()

    print()
    print("=" * 60)
    print("Embedding complete")
    print("=" * 60)

    print(f"Generated: {generated}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")


if __name__ == "__main__":
    main()
