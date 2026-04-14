"""Shared runtime models for LLM task execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ankiops.models import NoteTypeConfig

from .task_types import NotePayload


@dataclass(frozen=True)
class EligibleCandidate:
    item_id: int
    deck_name: str
    payload: NotePayload
    note_type_config: NoteTypeConfig
    serialized_note: dict[str, Any]
