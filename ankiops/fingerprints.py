"""Helpers for stable note-level fingerprints."""

from __future__ import annotations

import json

from blake3 import blake3


def _stable_payload(note_type: str, fields: dict[str, str]) -> bytes:
    payload = {
        "note_type": note_type,
        "fields": fields,
    }
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def note_fingerprint(note_type: str, fields: dict[str, str]) -> str:
    """Compute a stable note-level fingerprint from type + fields."""
    return blake3(_stable_payload(note_type, fields)).hexdigest(length=8)
