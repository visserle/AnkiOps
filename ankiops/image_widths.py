"""Fix Markdown image widths in serialized AnkiOps decks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ankiops.collection import NOTE_TYPES_DIR
from ankiops.interchange import deserialize, serialize

logger = logging.getLogger(__name__)

_MARKDOWN_IMAGE_RE = re.compile(
    r"(?P<image>!\[[^\]\n]*\]\((?:<[^>\n]*>|[^\)\n]*)\))"
    r"(?P<width>\{width=(?P<width_value>\d+)\})?"
)


@dataclass
class ImageWidthFixResult:
    """Summary for a fix-image-widths run."""

    decks_checked: int = 0
    notes_checked: int = 0
    images_checked: int = 0
    decks_changed: int = 0
    notes_changed: int = 0
    images_changed: int = 0

    @property
    def changed(self) -> bool:
        return self.images_changed > 0


def _replacement_with_width(match: re.Match[str], width: int) -> str:
    return f"{match.group('image')}{{width={width}}}"


def _force_field_width(field_content: str, width: int) -> tuple[str, int, int]:
    images_checked = 0
    images_changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal images_checked, images_changed
        images_checked += 1
        current = match.group("width_value")
        if current != str(width):
            images_changed += 1
        return _replacement_with_width(match, width)

    return (
        _MARKDOWN_IMAGE_RE.sub(replace, field_content),
        images_checked,
        images_changed,
    )


def _cluster_target(width: int, clusters: list[int], tolerance: int) -> int:
    for cluster_width in clusters:
        if abs(width - cluster_width) <= tolerance:
            return cluster_width
    clusters.append(width)
    return width


def _normalize_field_widths(
    field_content: str,
    *,
    clusters: list[int],
    tolerance: int,
) -> tuple[str, int, int]:
    images_checked = 0
    images_changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal images_checked, images_changed
        images_checked += 1
        width_value = match.group("width_value")
        if width_value is None:
            return match.group(0)

        width = int(width_value)
        target_width = _cluster_target(width, clusters, tolerance)
        if width == target_width:
            return match.group(0)

        images_changed += 1
        return _replacement_with_width(match, target_width)

    return (
        _MARKDOWN_IMAGE_RE.sub(replace, field_content),
        images_checked,
        images_changed,
    )


def fix_image_widths_in_data(
    data: dict[str, Any],
    *,
    tolerance: int,
    width: int | None = None,
) -> ImageWidthFixResult:
    """Fix Markdown image widths in a serialized AnkiOps collection mapping."""
    result = ImageWidthFixResult()
    decks = data.get("decks")
    if not isinstance(decks, list):
        return result

    for deck in decks:
        if not isinstance(deck, dict):
            continue

        result.decks_checked += 1
        deck_changed = False
        notes = deck.get("notes")
        if not isinstance(notes, list):
            continue

        for note in notes:
            if not isinstance(note, dict):
                continue

            result.notes_checked += 1
            note_changed = False
            clusters: list[int] = []
            fields = note.get("fields")
            if not isinstance(fields, dict):
                continue

            for field_name, field_content in list(fields.items()):
                if not isinstance(field_content, str):
                    continue

                if width is None:
                    new_content, checked, changed = _normalize_field_widths(
                        field_content,
                        clusters=clusters,
                        tolerance=tolerance,
                    )
                else:
                    new_content, checked, changed = _force_field_width(
                        field_content,
                        width,
                    )

                result.images_checked += checked
                result.images_changed += changed

                if new_content != field_content:
                    fields[field_name] = new_content
                    note_changed = True

            if note_changed:
                result.notes_changed += 1
                deck_changed = True

        if deck_changed:
            result.decks_changed += 1

    return result


def fix_image_widths_collection(
    collection_dir: Path,
    *,
    deck: str | None = None,
    no_subdecks: bool = False,
    tolerance: int = 5,
    width: int | None = None,
    note_types_dir: Path | None = None,
) -> ImageWidthFixResult:
    """Serialize, fix image widths, and rewrite changed scoped decks."""
    resolved_note_types_dir = note_types_dir or (collection_dir / NOTE_TYPES_DIR)
    data = serialize(
        collection_dir,
        deck=deck,
        no_subdecks=no_subdecks,
        note_types_dir=resolved_note_types_dir,
    )
    result = fix_image_widths_in_data(data, tolerance=tolerance, width=width)

    if result.changed:
        deserialize(
            data,
            collection_dir=collection_dir,
            note_types_dir=resolved_note_types_dir,
            overwrite=True,
            quiet=True,
        )

    return result
