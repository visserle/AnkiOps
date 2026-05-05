"""Contract-driven runtime engine for v2 provider adapters."""

from __future__ import annotations

from ..domain.contracts import build_note_update_contract
from ..domain.outcomes import ProviderOutcome, ProviderOutcomeKind
from ..domain.payloads import NotePayload
from ..providers.adapter_base import AdapterRequest, StructuredProviderAdapter


class StructuredOutputEngine:
    def __init__(self, adapter: StructuredProviderAdapter) -> None:
        self._adapter = adapter

    async def close(self) -> None:
        await self._adapter.close()

    async def generate_note_update(
        self,
        *,
        note_payload: NotePayload,
        system_prompt: str,
        task_prompt: str,
        model_id: str,
        max_output_tokens: int,
        temperature: float | None,
        user_message_text: str | None = None,
    ) -> ProviderOutcome:
        contract = build_note_update_contract(note_payload)
        request = AdapterRequest(
            note_payload=note_payload,
            contract=contract,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            model_id=model_id,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            user_message_text=user_message_text,
        )
        outcome = await self._adapter.generate(request)
        if (
            outcome.kind is ProviderOutcomeKind.SUCCESS
            and outcome.update is not None
            and outcome.update.note_key != note_payload.note_key
        ):
            return ProviderOutcome(
                kind=ProviderOutcomeKind.VALIDATION_ERROR,
                error_message="Model returned mismatched note_key",
                provider_message_id=outcome.provider_message_id,
                response_model_id=outcome.response_model_id,
                request_id=outcome.request_id,
                usage=outcome.usage,
                latency_ms=outcome.latency_ms,
                raw_text=outcome.raw_text,
                raw_json=outcome.raw_json,
            )
        return outcome
