"""Dependency-free validation for Anki's flat media namespace."""

import os
from pathlib import Path, PureWindowsPath


def validate_media_filename(filename: str) -> str:
    """Return a portable flat media filename or reject an unsafe path."""
    if (
        not filename
        or filename in {".", ".."}
        or "\0" in filename
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
        or PureWindowsPath(filename).drive
    ):
        raise ValueError(
            f"Unsafe media path '{filename}' outside the supported media namespace: "
            "expected one filename inside media/."
        )
    return filename


def validate_local_media_path(path: Path, filename: str) -> Path:
    """Validate one local source-media path without following symlinks."""
    safe_name = validate_media_filename(filename)
    if (
        path.name != safe_name
        or path.parent.name != "media"
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ValueError(
            f"Unsafe local media path '{path}': expected media/{safe_name}."
        )
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    has_symlink = False
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            has_symlink = True
            break
    if has_symlink:
        raise ValueError(
            f"Unsafe local media path '{path}': symbolic links are not allowed."
        )
    return path
