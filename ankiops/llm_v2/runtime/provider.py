"""Provider runtime for strict structured LLM requests."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ankiops.llm.model_registry import ModelSpec
from ankiops.llm.task_types import (
    NotePayload,
    TaskConfig,
    TaskRequestOptions,
)

from ..catalog import capabilities_from_model_spec
from ..domain.capabilities import ModelCapabilities, TransportMode
from ..domain.contracts import build_note_update_contract
from ..domain.errors import CapabilityError, RuntimeFatalError
from ..domain.outcomes import ProviderOutcome
from ..domain.payloads import NotePayload as V2NotePayload
from ..providers import AdapterRequest, StructuredProviderAdapter
from ..providers.adapter_factory import create_structured_adapter
from .engine import StructuredOutputEngine


@dataclass(frozen=True)
class PreparedProviderRequest:
    adapter_request: AdapterRequest | None
    system_prompt_text: str
    user_message_text: str
    request_params: dict[str, object]
    contract_fingerprint: str
    transport_mode: str
    capability_snapshot: dict[str, object]


class ProviderRuntime:
    """Prepare and execute contract-first requests against one provider adapter."""

    def __init__(self, task: TaskConfig) -> None:
        self._task = task
        self._model = task.model
        self._capabilities = _resolve_capabilities(task.model)
        self._adapter = _create_adapter(
            model=task.model,
            capabilities=self._capabilities,
            timeout_seconds=task.timeout_seconds,
        )
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
    ) -> PreparedProviderRequest:
        v2_payload = _to_v2_payload(note_payload)
        contract = build_note_update_contract(v2_payload)
        system_prompt_text = self._task.system_prompt.strip()
        user_message_text = _build_user_message(task_prompt, note_payload)
        request_params = _build_request_params(
            model=self._model,
            capabilities=self._capabilities,
            model_id=model_id,
            request_options=request_options,
            schema_name=contract.schema_name,
            schema=contract.json_schema,
        )
        adapter_request = AdapterRequest(
            note_payload=v2_payload,
            contract=contract,
            system_prompt=system_prompt_text,
            task_prompt=task_prompt,
            model_id=model_id,
            max_output_tokens=request_options.max_output_tokens,
            temperature=request_options.temperature,
            user_message_text=user_message_text,
        )
        return PreparedProviderRequest(
            adapter_request=adapter_request,
            system_prompt_text=system_prompt_text,
            user_message_text=user_message_text,
            request_params=request_params,
            contract_fingerprint=contract.fingerprint,
            transport_mode=self._capabilities.transport_mode.value,
            capability_snapshot=_capability_snapshot(self._capabilities),
        )

    async def generate_update(
        self,
        *,
        prepared_request: PreparedProviderRequest,
    ) -> ProviderOutcome:
        if prepared_request.adapter_request is None:
            raise RuntimeFatalError("Prepared request is missing adapter request")
        return await self._engine.generate(prepared_request.adapter_request)


def _resolve_capabilities(model: ModelSpec) -> ModelCapabilities:
    try:
        return capabilities_from_model_spec(model)
    except CapabilityError as error:
        raise RuntimeFatalError(str(error)) from error


def _create_adapter(
    *,
    model: ModelSpec,
    capabilities: ModelCapabilities,
    timeout_seconds: int,
) -> StructuredProviderAdapter:
    api_key = _resolve_api_key(model.api_key)
    try:
        return create_structured_adapter(
            model=model,
            capabilities=capabilities,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    except ModuleNotFoundError as error:
        if error.name == "openai":
            raise RuntimeFatalError(
                "openai package is required for OpenAI-compatible LLM providers"
            ) from error
        raise


def _build_request_params(
    *,
    model: ModelSpec,
    capabilities: ModelCapabilities,
    model_id: str,
    request_options: TaskRequestOptions,
    schema_name: str,
    schema: dict[str, object],
) -> dict[str, object]:
    transport_mode = capabilities.transport_mode
    if transport_mode is TransportMode.OPENAI_RESPONSES_STRUCTURED:
        params: dict[str, object] = {
            "model": model_id,
            "max_output_tokens": request_options.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
    elif transport_mode is TransportMode.OPENAI_COMPAT_JSON_SCHEMA_STRICT:
        params = {
            "model": model_id,
            "max_tokens": request_options.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if model.provider.strip().lower() == "ollama":
            params["extra_body"] = {"think": False}
    elif transport_mode is TransportMode.ANTHROPIC_TOOL_USE_STRICT:
        params = {
            "model": model_id,
            "max_tokens": request_options.max_output_tokens,
            "tools": [
                {
                    "name": schema_name,
                    "description": "Return the structured note update payload",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": schema_name},
        }
    elif transport_mode is TransportMode.GEMINI_NATIVE_STRUCTURED:
        params = {
            "model": model_id,
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "maxOutputTokens": request_options.max_output_tokens,
            },
        }
    else:
        raise RuntimeFatalError(f"Unsupported transport mode: {transport_mode}")

    if request_options.temperature is not None:
        if transport_mode is TransportMode.GEMINI_NATIVE_STRUCTURED:
            generation_config = params["generationConfig"]
            if isinstance(generation_config, dict):
                generation_config["temperature"] = request_options.temperature
        else:
            params["temperature"] = request_options.temperature

    return params


def _capability_snapshot(capabilities: ModelCapabilities) -> dict[str, object]:
    return {
        "provider": capabilities.provider,
        "model_id": capabilities.model_id,
        "transport_mode": capabilities.transport_mode.value,
        "supports_strict_schema": capabilities.supports_strict_schema,
        "supports_streaming_structured": capabilities.supports_streaming_structured,
        "schema_limits_profile": capabilities.schema_limits_profile.value,
    }


def _build_user_message(task_prompt: str, note_payload: NotePayload) -> str:
    payload: dict[str, object] = {
        "note_key": note_payload.note_key,
        "note_type": note_payload.note_type,
        "editable_fields": note_payload.editable_fields,
    }
    if note_payload.read_only_fields:
        payload["read_only_fields"] = note_payload.read_only_fields
    note_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"<task>\n{task_prompt.strip()}\n</task>\n\n<note>\n{note_json}\n</note>"


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
            raise RuntimeFatalError(
                "Model api_key env reference must include a variable"
            )
        resolved = os.environ.get(env_name)
        if not resolved:
            raise RuntimeFatalError(
                f"Required environment variable '{env_name}' is not set"
            )
        return resolved
    return configured_value
