"""Provider adapter interfaces for runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..domain.contracts import NoteUpdateContract
from ..domain.outcomes import ProviderOutcome
from ..domain.payloads import NotePayload


@dataclass(frozen=True)
class AdapterRequest:
    note_payload: NotePayload
    contract: NoteUpdateContract
    system_prompt: str
    task_prompt: str
    model_id: str
    max_output_tokens: int
    user_message_text: str | None = None
    temperature: float | None = None


class StructuredProviderAdapter(Protocol):
    async def generate(self, request: AdapterRequest) -> ProviderOutcome: ...

    async def close(self) -> None: ...
