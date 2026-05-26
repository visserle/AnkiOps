"""Helpers for stable note-level fingerprints."""

from __future__ import annotations

import json
from collections.abc import Iterable

from blake3 import blake3

from ankiops.tags import normalize_tags


def _stable_payload(
    note_type: str,
    fields: dict[str, str],
    *,
    tags: Iterable[str] | str | None = (),
) -> bytes:
    payload = {
        "note_type": note_type,
        "fields": fields,
        "tags": list(normalize_tags(tags)),
    }
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def note_fingerprint(
    note_type: str,
    fields: dict[str, str],
    *,
    tags: Iterable[str] | str | None = (),
) -> str:
    """Compute a stable note-level fingerprint from type + fields + tags."""
    return blake3(_stable_payload(note_type, fields, tags=tags)).hexdigest(length=8)
