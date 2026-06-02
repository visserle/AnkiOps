"""Helpers for note-level Anki tags."""

from __future__ import annotations

from collections.abc import Iterable

from ankiops.markdown_format import TAGS_COMMENT_RE


def normalize_tags(tags: Iterable[str] | str | None = None) -> tuple[str, ...]:
    """Return canonical Anki tags: trimmed, deduped, and sorted."""
    if tags is None:
        return ()

    raw_tags = tags.split() if isinstance(tags, str) else tags
    normalized = {
        tag.strip() for tag in raw_tags if isinstance(tag, str) and tag.strip()
    }
    return tuple(sorted(normalized))


def parse_tags_comment(line: str) -> tuple[str, ...] | None:
    """Parse a full-line tags comment, returning None when it is not one."""
    match = TAGS_COMMENT_RE.match(line)
    if not match:
        return None
    return normalize_tags(match.group(1))


def format_tags_comment(tags: Iterable[str] | str | None) -> str | None:
    """Format tags as a Markdown metadata comment."""
    normalized = normalize_tags(tags)
    if not normalized:
        return None
    return f"<!-- tags: {' '.join(normalized)} -->"
