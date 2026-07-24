"""
Semantic search for Smart Photo Archive.

Converts natural-language text into a CLIP embedding,
compares it against stored image embeddings, and returns
the best matching photos.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from ai.clip_embed import (
    blob_to_embedding,
    generate_text_embedding,
)
from core.config import DATABASE_PATH


@dataclass(slots=True)
class SearchResult:
    photo_id: int
    file_name: str
    file_path: str
    thumbnail_path: str | None
    width: int | None
    height: int | None
    score: float


def get_connection() -> sqlite3.Connection:
    """Open the Smart Photo Archive SQLite database."""

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Calculate cosine similarity.

    The stored CLIP vectors are normalized, so dot product
    is equivalent to cosine similarity.
    """

    return float(np.dot(a, b))


def search_photos(
    query: str,
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Search indexed photos with natural language.

    Example:
        search_photos("cherry blossoms")
    """

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_embedding = generate_text_embedding(query)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                p.id AS photo_id,
                p.file_name,
                p.file_path,
                p.thumbnail_path,
                p.width,
                p.height,
                e.dimension,
                e.embedding
            FROM photos AS p
            INNER JOIN photo_embeddings AS e
                ON e.photo_id = p.id
            WHERE p.status = 'indexed'
            """
        ).fetchall()

    results: list[SearchResult] = []

    for row in rows:
        image_embedding = blob_to_embedding(
            row["embedding"],
            row["dimension"],
        )

        score = cosine_similarity(
            query_embedding,
            image_embedding,
        )

        results.append(
            SearchResult(
                photo_id=row["photo_id"],
                file_name=row["file_name"],
                file_path=row["file_path"],
                thumbnail_path=row["thumbnail_path"],
                width=row["width"],
                height=row["height"],
                score=score,
            )
        )

    results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return results[:top_k]


def print_results(
    query: str,
    results: list[SearchResult],
) -> None:
    """Print ranked search results."""

    print()
    print("=" * 65)
    print(f"Top matches for: {query!r}")
    print("=" * 65)

    if not results:
        print("No embedded photos found.")
        return

    for rank, result in enumerate(results, start=1):
        print()
        print(f"{rank}. {result.file_name}")
        print(f"   Similarity: {result.score * 100:.1f}%")

        if result.width is not None and result.height is not None:
            print(
                f"   Resolution: "
                f"{result.width} x {result.height}"
            )

        print(f"   Path: {result.file_path}")


def main() -> None:
    """Run interactive semantic search."""

    print("=" * 65)
    print("Smart Photo Archive - AI Semantic Search")
    print("=" * 65)

    while True:
        print()

        query = input(
            "Describe what you're looking for "
            "(or type 'quit'): "
        ).strip()

        if query.lower() in {
            "quit",
            "exit",
            "q",
        }:
            print("Search closed.")
            break

        if not query:
            continue

        try:
            matches = search_photos(
                query,
                top_k=5,
            )

            print_results(
                query,
                matches,
            )

        except Exception as error:
            print(f"Search failed: {error}")


if __name__ == "__main__":
    main()
