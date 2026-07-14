"""Paths in Anki's flat media namespace."""

from pathlib import Path


def parse_media_filename(value: str) -> str:
    """Parse one filename stored directly inside a media directory."""
    if (
        not value
        or value in {".", ".."}
        or "\0" in value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(
            f"Invalid media reference '{value}': expected one filename in media/."
        )
    return value


def media_path(media_dir: Path, filename: str) -> Path:
    return media_dir / parse_media_filename(filename)
