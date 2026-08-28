"""Tag policies for shared (collab) decks."""

from __future__ import annotations

from collections.abc import Iterable

from ankiops.notes import normalize_tags

# The leech tag marks personal scheduling state (cards that repeatedly lapse).
# It must never enter or leave a shared deck through collab sync.
LEECH_TAG = "leech"


def exclude_leech_tags(tags: Iterable[str] | str | None) -> tuple[str, ...]:
    """Return normalized tags without the leech tag."""
    return tuple(tag for tag in normalize_tags(tags) if tag != LEECH_TAG)
