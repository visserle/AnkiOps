"""OpenAI Responses API adapter for strict structured output."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AuthenticationError

from ..domain.contracts import ContractValidationError
from ..domain.outcomes import ProviderOutcome, ProviderOutcomeKind, ProviderUsage
from .adapter_base import AdapterRequest


class OpenAIResponsesStructuredAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: int = 60,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout_seconds),
            max_retries=0,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def generate(self, request: AdapterRequest) -> ProviderOutcome:
        started_at = time.monotonic()
        request_kwargs: dict[str, object] = {
            "model": request.model_id,
            "input": [
                {"role": "system", "content": request.system_prompt.strip()},
                {
                    "role": "user",
                    "content": request.user_message_text
                    if request.user_message_text is not None
                    else _build_user_message(request),
                },
            ],
            "max_output_tokens": request.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.contract.schema_name,
                    "schema": request.contract.json_schema,
                    "strict": True,
                }
            },
        }
        if request.temperature is not None:
            request_kwargs["temperature"] = request.temperature

        try:
            response = await self._client.responses.create(**request_kwargs)
        except AuthenticationError as error:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.FATAL_ERROR,
                error_message=f"Provider authentication failed: {error}",
            )
        except APIConnectionError as error:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.PROVIDER_ERROR,
                error_message=f"Provider connection error: {error}",
            )
        except APIStatusError as error:
            status_code = getattr(error, "status_code", "unknown")
            return ProviderOutcome(
                kind=ProviderOutcomeKind.PROVIDER_ERROR,
                error_message=f"Provider returned HTTP {status_code}: {error}",
            )
        except Exception as error:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.FATAL_ERROR,
                error_message=f"Unexpected provider error: {error}",
            )

        latency_ms = round((time.monotonic() - started_at) * 1000)
        usage = ProviderUsage(
            input_tokens=_usage_value(getattr(response, "usage", None), "input_tokens"),
            output_tokens=_usage_value(
                getattr(response, "usage", None),
                "output_tokens",
            ),
        )
        provider_message_id = _as_non_empty_str(getattr(response, "id", None))
        response_model_id = (
            _as_non_empty_str(getattr(response, "model", None)) or request.model_id
        )
        request_id = _as_non_empty_str(getattr(response, "_request_id", None))
        raw_json = _response_to_json(response)

        refusal_text = _extract_refusal_text(response)
        if refusal_text is not None:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.REFUSAL,
                refusal_text=refusal_text,
                provider_message_id=provider_message_id,
                response_model_id=response_model_id,
                request_id=request_id,
                usage=usage,
                latency_ms=latency_ms,
                raw_json=raw_json,
            )

        output_parsed = getattr(response, "output_parsed", None)
        if output_parsed is not None:
            try:
                update = request.contract.parse_data(output_parsed)
            except ContractValidationError as error:
                return ProviderOutcome(
                    kind=ProviderOutcomeKind.VALIDATION_ERROR,
                    error_message=str(error),
                    provider_message_id=provider_message_id,
                    response_model_id=response_model_id,
                    request_id=request_id,
                    usage=usage,
                    latency_ms=latency_ms,
                    raw_json=raw_json,
                )
            return ProviderOutcome(
                kind=ProviderOutcomeKind.SUCCESS,
                update=update,
                provider_message_id=provider_message_id,
                response_model_id=response_model_id,
                request_id=request_id,
                usage=usage,
                latency_ms=latency_ms,
                raw_json=raw_json,
            )

        raw_text = _extract_output_text(response)
        if not raw_text:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.PROVIDER_ERROR,
                error_message="Provider response contained no JSON text output",
                provider_message_id=provider_message_id,
                response_model_id=response_model_id,
                request_id=request_id,
                usage=usage,
                latency_ms=latency_ms,
                raw_json=raw_json,
            )

        try:
            update = request.contract.parse_raw_json(raw_text)
        except ContractValidationError as error:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.VALIDATION_ERROR,
                error_message=str(error),
                provider_message_id=provider_message_id,
                response_model_id=response_model_id,
                request_id=request_id,
                usage=usage,
                latency_ms=latency_ms,
                raw_text=raw_text,
                raw_json=raw_json,
            )

        return ProviderOutcome(
            kind=ProviderOutcomeKind.SUCCESS,
            update=update,
            provider_message_id=provider_message_id,
            response_model_id=response_model_id,
            request_id=request_id,
            usage=usage,
            latency_ms=latency_ms,
            raw_text=raw_text,
            raw_json=raw_json,
        )


def _build_user_message(request: AdapterRequest) -> str:
    payload: dict[str, object] = {
        "note_key": request.note_payload.note_key,
        "note_type": request.note_payload.note_type,
        "editable_fields": request.note_payload.editable_fields,
    }
    if request.note_payload.read_only_fields:
        payload["read_only_fields"] = request.note_payload.read_only_fields
    note_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        f"<task>\n{request.task_prompt.strip()}\n</task>\n\n"
        f"<note>\n{note_json}\n</note>"
    )


def _usage_value(usage: object, name: str) -> int:
    value = getattr(usage, name, 0)
    return value if isinstance(value, int) else 0


def _as_non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _response_to_json(response: object) -> str | None:
    model_dump_json = getattr(response, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            value = model_dump_json()
            return value if isinstance(value, str) else None
        except Exception:
            return None

    to_json = getattr(response, "to_json", None)
    if callable(to_json):
        try:
            value = to_json()
            return value if isinstance(value, str) else None
        except Exception:
            return None

    return None


def _extract_refusal_text(response: object) -> str | None:
    refusal_parts: list[str] = []
    for part in _iter_response_parts(response):
        part_type = _field(part, "type")
        if part_type != "refusal":
            continue
        refusal = _field(part, "refusal")
        if isinstance(refusal, str) and refusal:
            refusal_parts.append(refusal)
    if not refusal_parts:
        return None
    return "\n".join(refusal_parts)


def _extract_output_text(response: object) -> str | None:
    top_level_text = _field(response, "output_text")
    if isinstance(top_level_text, str) and top_level_text:
        return top_level_text

    text_parts: list[str] = []
    for part in _iter_response_parts(response):
        part_type = _field(part, "type")
        if part_type not in {"output_text", "text"}:
            continue
        text = _field(part, "text")
        if isinstance(text, str) and text:
            text_parts.append(text)
    if not text_parts:
        return None
    return "".join(text_parts)


def _iter_response_parts(response: object) -> Iterable[object]:
    output = _field(response, "output")
    if not isinstance(output, list):
        return []

    parts: list[object] = []
    for item in output:
        content = _field(item, "content")
        if isinstance(content, list):
            parts.extend(content)
    return parts


def _field(obj: object, name: str) -> object:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
