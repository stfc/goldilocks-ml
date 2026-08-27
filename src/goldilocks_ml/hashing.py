"""Shared file integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

_BLOCK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
