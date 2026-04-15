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


@dataclass(frozen=True)
class TaskExecutionProgress:
    job_id: int
    task_name: str
    total: int
    completed: int
    in_flight: int
    queued: int
    updated: int
    unchanged: int
    skipped: int
    errors: int
    canceled: int

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 1.0
        return min(self.completed / self.total, 1.0)

    @property
    def is_finished(self) -> bool:
        return self.completed >= self.total
