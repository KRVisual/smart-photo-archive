"""
Hashing utilities for Smart Photo Archive.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from core.config import HASH_ALGORITHM, HASH_CHUNK_SIZE


def calculate_file_hash(file_path: str | Path) -> str:
    """
    Calculate a cryptographic hash for a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hexadecimal hash string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path is a directory.
        OSError: If the file cannot be read.
        ValueError: If the configured hash algorithm is invalid.
    """

    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if path.is_dir():
        raise IsADirectoryError(f"Expected a file, got directory: {path}")

    try:
        hasher = hashlib.new(HASH_ALGORITHM)
    except ValueError as error:
        raise ValueError(
            f"Unsupported hash algorithm: {HASH_ALGORITHM}"
        ) from error

    with path.open("rb") as file:
        while True:
            chunk = file.read(HASH_CHUNK_SIZE)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def main() -> None:
    """Interactive hashing test."""

    print("=" * 60)
    print("Smart Photo Archive - Hashing Test")
    print("=" * 60)

    file_input = input("Enter a file path: ").strip().strip('"')

    try:
        result = calculate_file_hash(file_input)

        print()
        print(f"File: {Path(file_input).name}")
        print(f"Hash: {result}")

    except (FileNotFoundError, IsADirectoryError, OSError, ValueError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
