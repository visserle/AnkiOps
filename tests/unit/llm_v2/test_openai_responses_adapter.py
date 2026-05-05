from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ankiops.llm_v2.domain.contracts import build_note_update_contract
from ankiops.llm_v2.domain.outcomes import ProviderOutcomeKind
from ankiops.llm_v2.domain.payloads import NotePayload
from ankiops.llm_v2.providers.adapter_base import AdapterRequest
from ankiops.llm_v2.providers.openai_responses_adapter import (
    OpenAIResponsesStructuredAdapter,
)


class _FakeResponsesClient:
    def __init__(
        self,
        *,
        response: object | None = None,
        error: Exception | None = None,
    ):
        self._response = response
        self._error = error
        self.last_kwargs: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        self.last_kwargs = dict(kwargs)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _FakeClient:
    def __init__(
        self,
        *,
        response: object | None = None,
        error: Exception | None = None,
    ):
        self.responses = _FakeResponsesClient(response=response, error=error)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _build_request() -> AdapterRequest:
    payload = NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={"Question": "Broken"},
        read_only_fields={"Source": "Book"},
    )
    contract = build_note_update_contract(payload)
    return AdapterRequest(
        note_payload=payload,
        contract=contract,
        system_prompt="System prompt",
        task_prompt="Fix grammar",
        model_id="gpt-5.4",
        max_output_tokens=200,
        temperature=0.2,
    )


def test_generate_uses_output_parsed_when_available() -> None:
    response = SimpleNamespace(
        id="resp_123",
        model="gpt-5.4",
        output_parsed={"note_key": "nk-1", "edits": {"Question": "Fixed"}},
        usage=SimpleNamespace(input_tokens=11, output_tokens=5),
        model_dump_json=lambda: "{\"id\":\"resp_123\"}",
    )
    client = _FakeClient(response=response)
    adapter = OpenAIResponsesStructuredAdapter(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        client=client,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.SUCCESS
    assert outcome.update is not None
    assert outcome.update.edits == {"Question": "Fixed"}
    assert outcome.usage.input_tokens == 11
    assert outcome.usage.output_tokens == 5
    assert outcome.raw_json == '{"id":"resp_123"}'

    request_kwargs = client.responses.last_kwargs
    assert request_kwargs is not None
    text = request_kwargs["text"]
    assert isinstance(text, dict)
    text_format = text["format"]
    assert isinstance(text_format, dict)
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True


def test_generate_maps_refusal_to_refusal_outcome() -> None:
    response = SimpleNamespace(
        id="resp_123",
        model="gpt-5.4",
        output=[
            {
                "type": "message",
                "content": [{"type": "refusal", "refusal": "I cannot do that."}],
            }
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2),
        model_dump_json=lambda: "{}",
    )
    client = _FakeClient(response=response)
    adapter = OpenAIResponsesStructuredAdapter(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        client=client,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.REFUSAL
    assert outcome.refusal_text == "I cannot do that."


def test_generate_returns_validation_error_for_invalid_structured_output() -> None:
    response = SimpleNamespace(
        id="resp_123",
        model="gpt-5.4",
        output_parsed={"note_key": "nk-1", "edits": {"Source": "Book"}},
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        model_dump_json=lambda: "{}",
    )
    client = _FakeClient(response=response)
    adapter = OpenAIResponsesStructuredAdapter(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        client=client,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.VALIDATION_ERROR
    assert outcome.error_message is not None
    assert "not editable" in outcome.error_message
