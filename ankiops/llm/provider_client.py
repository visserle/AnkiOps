"""Runtime provider client backed by llm_v2 structured adapters."""

from __future__ import annotations

import json
import os

from ankiops.llm_v2.catalog import capabilities_from_model_spec
from ankiops.llm_v2.domain.capabilities import ModelCapabilities, TransportMode
from ankiops.llm_v2.domain.contracts import build_note_update_contract
from ankiops.llm_v2.domain.errors import CapabilityError
from ankiops.llm_v2.domain.outcomes import ProviderOutcome, ProviderOutcomeKind
from ankiops.llm_v2.domain.payloads import NotePayload as V2NotePayload
from ankiops.llm_v2.runtime import StructuredOutputEngine

from .llm_errors import LlmFatalError, LlmNoteError, LlmNoteErrorCategory
from .prompting import build_system_prompt, build_user_message
from .task_types import (
    NotePayload,
    NoteUpdate,
    PreparedAttemptRequest,
    ProviderAttemptErrorContext,
    ProviderAttemptOutcome,
    TaskConfig,
    TaskRequestOptions,
)


def create_structured_adapter(*args, **kwargs):
    from ankiops.llm_v2.providers import create_structured_adapter as _create_adapter

    return _create_adapter(*args, **kwargs)


class ProviderClient:
    """Adapter-backed provider client for structured note updates."""

    def __init__(self, task: TaskConfig) -> None:
        self._system_prompt = task.system_prompt
        self._model = task.model

        api_key = _resolve_api_key(task.model.api_key)
        try:
            self._capabilities = capabilities_from_model_spec(task.model)
        except CapabilityError as error:
            raise LlmFatalError(str(error)) from error

        try:
            self._adapter = create_structured_adapter(
                model=task.model,
                capabilities=self._capabilities,
                api_key=api_key,
                timeout_seconds=task.timeout_seconds,
            )
        except ModuleNotFoundError as error:
            if error.name == "openai":
                raise LlmFatalError(
                    "openai package is required for OpenAI-compatible LLM providers"
                ) from error
            raise
        self._engine = StructuredOutputEngine(self._adapter)

    async def close(self) -> None:
        await self._engine.close()

    def prepare_attempt_request(
        self,
        *,
        note_payload: NotePayload,
        task_prompt: str,
        request_options: TaskRequestOptions,
        model_id: str,
    ) -> PreparedAttemptRequest:
        v2_payload = _to_v2_payload(note_payload)
        contract = build_note_update_contract(v2_payload)

        system_prompt_text = build_system_prompt(self._system_prompt)
        user_message_text = build_user_message(task_prompt, note_payload)
        request_params = self._build_request_params(
            model_id=model_id,
            max_output_tokens=request_options.max_output_tokens,
            temperature=request_options.temperature,
            schema=contract.json_schema,
        )

        return PreparedAttemptRequest(
            note_payload=note_payload,
            system_prompt_text=system_prompt_text,
            user_message_text=user_message_text,
            request_params=request_params,
            contract_fingerprint=contract.fingerprint,
            transport_mode=self._capabilities.transport_mode.value,
            capability_snapshot=_capability_snapshot(self._capabilities),
        )

    def _build_request_params(
        self,
        *,
        model_id: str,
        max_output_tokens: int,
        temperature: float | None,
        schema: dict[str, object],
    ) -> dict[str, object]:
        if (
            self._capabilities.transport_mode
            is TransportMode.OPENAI_RESPONSES_STRUCTURED
        ):
            params: dict[str, object] = {
                "model": model_id,
                "max_output_tokens": max_output_tokens,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "note_update",
                        "strict": True,
                        "schema": schema,
                    }
                },
            }
        else:
            params = {
                "model": model_id,
                "max_tokens": max_output_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "note_update",
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            if self._model.provider.strip().lower() == "ollama":
                params["extra_body"] = {"think": False}

        if temperature is not None:
            params["temperature"] = temperature

        return params

    async def generate_update(
        self,
        *,
        prepared_request: PreparedAttemptRequest,
    ) -> ProviderAttemptOutcome:
        v2_payload = _to_v2_payload(prepared_request.note_payload)

        request_params = prepared_request.request_params
        model_id = str(request_params.get("model") or "")
        if not model_id:
            raise LlmFatalError("Prepared request is missing provider model")

        max_output_tokens = _max_output_tokens_from_request_params(request_params)

        outcome = await self._engine.generate_note_update(
            note_payload=v2_payload,
            system_prompt=prepared_request.system_prompt_text,
            task_prompt="",
            model_id=model_id,
            max_output_tokens=max_output_tokens,
            temperature=_temperature_from_request_params(request_params),
            user_message_text=prepared_request.user_message_text,
        )

        if outcome.kind is ProviderOutcomeKind.SUCCESS and outcome.update is not None:
            update = NoteUpdate(
                note_key=outcome.update.note_key,
                edits=dict(outcome.update.edits),
            )
            raw_text = outcome.raw_text or json.dumps(
                {
                    "note_key": update.note_key,
                    "edits": update.edits,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return ProviderAttemptOutcome(
                update=update,
                provider_message_id=outcome.provider_message_id,
                response_model_id=outcome.response_model_id,
                stop_reason=None,
                request_id=outcome.request_id,
                rate_limit_headers={},
                input_tokens=outcome.usage.input_tokens,
                output_tokens=outcome.usage.output_tokens,
                latency_ms=outcome.latency_ms,
                retry_count=0,
                response_raw_text=raw_text,
                response_full_json=outcome.raw_json,
            )

        if outcome.kind is ProviderOutcomeKind.REFUSAL:
            raise LlmNoteError(
                outcome.refusal_text or "Provider refused request",
                category=LlmNoteErrorCategory.PROVIDER,
                attempt_context=_to_attempt_context(outcome),
            )

        if outcome.kind is ProviderOutcomeKind.VALIDATION_ERROR:
            raise LlmNoteError(
                outcome.error_message or "Structured output validation failed",
                category=LlmNoteErrorCategory.NOTE,
                attempt_context=_to_attempt_context(outcome),
            )

        if outcome.kind is ProviderOutcomeKind.PROVIDER_ERROR:
            raise LlmNoteError(
                outcome.error_message or "Provider request failed",
                category=LlmNoteErrorCategory.PROVIDER,
                attempt_context=_to_attempt_context(outcome),
            )

        if outcome.kind is ProviderOutcomeKind.FATAL_ERROR:
            raise LlmFatalError(
                outcome.error_message or "Unexpected provider failure",
                attempt_context=_to_attempt_context(outcome),
            )

        raise LlmFatalError("Unexpected provider failure")


def _to_attempt_context(outcome: ProviderOutcome) -> ProviderAttemptErrorContext:
    return ProviderAttemptErrorContext(
        outcome_kind=outcome.kind.value,
        refusal_reason=outcome.refusal_text,
        provider_message_id=outcome.provider_message_id,
        response_model_id=outcome.response_model_id,
        stop_reason=None,
        request_id=outcome.request_id,
        rate_limit_headers={},
        input_tokens=outcome.usage.input_tokens,
        output_tokens=outcome.usage.output_tokens,
        latency_ms=outcome.latency_ms,
        retry_count=0,
        response_raw_text=outcome.raw_text,
        response_full_json=outcome.raw_json,
    )


def _capability_snapshot(capabilities: ModelCapabilities) -> dict[str, object]:
    return {
        "provider": capabilities.provider,
        "model_id": capabilities.model_id,
        "transport_mode": capabilities.transport_mode.value,
        "supports_strict_json": capabilities.supports_strict_json,
    }


def _to_v2_payload(payload: NotePayload) -> V2NotePayload:
    return V2NotePayload(
        note_key=payload.note_key,
        note_type=payload.note_type,
        editable_fields=dict(payload.editable_fields),
        read_only_fields=dict(payload.read_only_fields),
    )


def _resolve_api_key(configured_value: str) -> str:
    if configured_value.startswith("$"):
        env_name = configured_value[1:].strip()
        if not env_name:
            raise LlmFatalError("Model api_key env reference must include a variable")
        resolved = os.environ.get(env_name)
        if not resolved:
            raise LlmFatalError(
                f"Required environment variable '{env_name}' is not set"
            )
        return resolved
    return configured_value


def _temperature_from_request_params(request_params: dict[str, object]) -> float | None:
    value = request_params.get("temperature")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _max_output_tokens_from_request_params(request_params: dict[str, object]) -> int:
    raw_responses = request_params.get("max_output_tokens")
    if isinstance(raw_responses, int):
        return raw_responses

    raw_chat = request_params.get("max_tokens")
    if isinstance(raw_chat, int):
        return raw_chat

    raise LlmFatalError("Prepared request is missing max output tokens")
