from __future__ import annotations

import asyncio

from ankiops.llm_v2.domain.outcomes import (
    ProviderOutcome,
    ProviderOutcomeKind,
    ProviderUsage,
)
from ankiops.llm_v2.domain.payloads import NotePayload, NoteUpdate
from ankiops.llm_v2.providers.adapter_base import AdapterRequest
from ankiops.llm_v2.runtime.engine import StructuredOutputEngine


class _FakeAdapter:
    def __init__(self, outcome: ProviderOutcome):
        self._outcome = outcome
        self.last_request: AdapterRequest | None = None

    async def generate(self, request: AdapterRequest) -> ProviderOutcome:
        self.last_request = request
        return self._outcome

    async def close(self) -> None:
        return None


def test_engine_converts_mismatched_note_key_to_validation_error() -> None:
    adapter = _FakeAdapter(
        ProviderOutcome(
            kind=ProviderOutcomeKind.SUCCESS,
            update=NoteUpdate(note_key="nk-other", edits={"Question": "Fixed"}),
            usage=ProviderUsage(input_tokens=1, output_tokens=1),
        )
    )
    engine = StructuredOutputEngine(adapter)

    outcome = asyncio.run(
        engine.generate_note_update(
            note_payload=NotePayload(
                note_key="nk-1",
                note_type="AnkiOpsQA",
                editable_fields={"Question": "Broken"},
            ),
            system_prompt="System",
            task_prompt="Fix",
            model_id="gpt-5.4",
            max_output_tokens=200,
            temperature=None,
        )
    )

    assert outcome.kind is ProviderOutcomeKind.VALIDATION_ERROR
    assert outcome.error_message == "Model returned mismatched note_key"
