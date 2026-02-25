"""Types shared by the AI prompt runtime."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelProfile:
    """A named model runtime profile loaded from models.yaml."""

    name: str
    provider: str
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: int
    max_in_flight: int


@dataclass(frozen=True)
class ModelsConfig:
    """Collection-scoped model profile configuration."""

    default_profile: str
    profiles: dict[str, ModelProfile]
    source_path: Path | None = None


@dataclass(frozen=True)
class RuntimeAIConfig:
    """Fully resolved runtime AI configuration."""

    profile: str
    provider: str
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: int
    max_in_flight: int
    api_key: str | None = None


@dataclass(frozen=True)
class PromptConfig:
    """Prompt definition loaded from a YAML file."""

    name: str
    prompt: str
    target_fields: list[str]
    send_fields: list[str]
    note_types: list[str]
    model_profile: str | None = None
    temperature: float = 0.0
    source_path: Path | None = None

    def matches_note_type(self, note_type: str) -> bool:
        return any(
            fnmatch.fnmatchcase(note_type, pattern) for pattern in self.note_types
        )


@dataclass(frozen=True)
class PromptChange:
    prompt_name: str
    deck_name: str
    note_key: str
    note_type: str
    field_name: str
    original_text: str
    edited_text: str


@dataclass
class PromptRunResult:
    processed_decks: int = 0
    processed_notes: int = 0
    prompted_notes: int = 0
    changed_fields: int = 0
    changes: list[PromptChange] = field(default_factory=list)
    changed_decks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class AsyncInlineBatchEditor(Protocol):
    """Protocol for a batch-capable async inline note editor."""

    async def edit_batch(
        self,
        prompt_config: PromptConfig,
        note_payloads: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Return edited notes keyed by note_key."""
