from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import patch

import pytest

from ankiops.llm.llm_errors import LlmFatalError, LlmNoteError, LlmNoteErrorCategory
from ankiops.llm.model_registry import ModelSpec
from ankiops.llm.provider_client import ProviderClient
from ankiops.llm.task_types import (
    NotePayload,
    TaskConfig,
    TaskRequestOptions,
)
from ankiops.llm_v2.domain.capabilities import ModelCapabilities, TransportMode
from ankiops.llm_v2.domain.outcomes import (
    ProviderOutcome,
    ProviderOutcomeKind,
    ProviderUsage,
)
from ankiops.llm_v2.domain.payloads import NoteUpdate

TEST_MODEL = ModelSpec(
    model="gpt-5.4",
    model_id="gpt-5.4",
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
)


class _FakeAdapter:
    def __init__(self, outcome: ProviderOutcome):
        self._outcome = outcome
        self.last_request = None

    async def generate(self, request):
        self.last_request = request
        return self._outcome

    async def close(self):
        return None


@pytest.fixture
def note_payload() -> NotePayload:
    return NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={"Question": "Broken"},
        read_only_fields={"Source": "Book"},
    )


def test_missing_api_key_raises_fatal(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LlmFatalError, match="OPENAI_API_KEY"):
        ProviderClient(
            TaskConfig(
                name="grammar",
                model=TEST_MODEL,
                system_prompt="System prompt for tests",
                prompt="Fix grammar",
            )
        )


def test_prepare_attempt_request_uses_responses_json_schema(monkeypatch, note_payload):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    capabilities = ModelCapabilities(
        provider="openai",
        model_id="gpt-5.4",
        transport_mode=TransportMode.OPENAI_RESPONSES_STRUCTURED,
        supports_strict_json=True,
    )
    adapter = _FakeAdapter(
        ProviderOutcome(
            kind=ProviderOutcomeKind.SUCCESS,
            update=NoteUpdate(note_key="nk-1", edits={"Question": "Fixed"}),
            usage=ProviderUsage(input_tokens=1, output_tokens=1),
        )
    )

    with (
        patch(
            "ankiops.llm.provider_client.capabilities_from_model_spec",
            return_value=capabilities,
        ),
        patch(
            "ankiops.llm.provider_client.create_structured_adapter",
            return_value=adapter,
        ),
    ):
        client = ProviderClient(
            TaskConfig(
                name="grammar",
                model=TEST_MODEL,
                system_prompt="System prompt for tests",
                prompt="Fix grammar",
            )
        )
        prepared = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(max_output_tokens=200),
            model_id="gpt-5.4",
        )

    assert prepared.request_params["max_output_tokens"] == 200
    text = prepared.request_params["text"]
    assert isinstance(text, dict)
    text_format = text["format"]
    assert isinstance(text_format, dict)
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True


def test_prepare_attempt_request_uses_openai_compat_params_for_ollama(
    monkeypatch,
    note_payload,
):
    ollama_model = replace(TEST_MODEL, provider="ollama", model_id="gemma4:e2b")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    capabilities = ModelCapabilities(
        provider="ollama",
        model_id="gemma4:e2b",
        transport_mode=TransportMode.OPENAI_COMPAT_JSON_SCHEMA_STRICT,
        supports_strict_json=True,
    )
    adapter = _FakeAdapter(
        ProviderOutcome(
            kind=ProviderOutcomeKind.SUCCESS,
            update=NoteUpdate(note_key="nk-1", edits={"Question": "Fixed"}),
            usage=ProviderUsage(input_tokens=1, output_tokens=1),
        )
    )

    with (
        patch(
            "ankiops.llm.provider_client.capabilities_from_model_spec",
            return_value=capabilities,
        ),
        patch(
            "ankiops.llm.provider_client.create_structured_adapter",
            return_value=adapter,
        ),
    ):
        client = ProviderClient(
            TaskConfig(
                name="grammar",
                model=ollama_model,
                system_prompt="System prompt for tests",
                prompt="Fix grammar",
            )
        )
        prepared = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(max_output_tokens=128),
            model_id="gemma4:e2b",
        )

    assert prepared.request_params["max_tokens"] == 128
    assert prepared.request_params["extra_body"] == {"think": False}


def test_generate_update_maps_success(monkeypatch, note_payload):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = _FakeAdapter(
        ProviderOutcome(
            kind=ProviderOutcomeKind.SUCCESS,
            update=NoteUpdate(note_key="nk-1", edits={"Question": "Fixed"}),
            provider_message_id="msg_123",
            response_model_id="gpt-5.4",
            usage=ProviderUsage(input_tokens=11, output_tokens=3),
            latency_ms=50,
            raw_text='{"note_key":"nk-1","edits":{"Question":"Fixed"}}',
            raw_json='{"id":"msg_123"}',
        )
    )

    with (
        patch(
            "ankiops.llm.provider_client.capabilities_from_model_spec",
            return_value=ModelCapabilities(
                provider="openai",
                model_id="gpt-5.4",
                transport_mode=TransportMode.OPENAI_RESPONSES_STRUCTURED,
                supports_strict_json=True,
            ),
        ),
        patch(
            "ankiops.llm.provider_client.create_structured_adapter",
            return_value=adapter,
        ),
    ):
        client = ProviderClient(
            TaskConfig(
                name="grammar",
                model=TEST_MODEL,
                system_prompt="System prompt for tests",
                prompt="Fix grammar",
            )
        )
        prepared = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(max_output_tokens=200),
            model_id="gpt-5.4",
        )
        outcome = asyncio.run(client.generate_update(prepared_request=prepared))

    assert outcome.update.note_key == "nk-1"
    assert outcome.update.edits == {"Question": "Fixed"}
    assert outcome.provider_message_id == "msg_123"
    assert outcome.input_tokens == 11
    assert outcome.output_tokens == 3


def test_generate_update_maps_refusal_to_provider_note_error(monkeypatch, note_payload):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = _FakeAdapter(
        ProviderOutcome(
            kind=ProviderOutcomeKind.REFUSAL,
            refusal_text="I cannot assist with that.",
            usage=ProviderUsage(input_tokens=1, output_tokens=1),
        )
    )

    with (
        patch(
            "ankiops.llm.provider_client.capabilities_from_model_spec",
            return_value=ModelCapabilities(
                provider="openai",
                model_id="gpt-5.4",
                transport_mode=TransportMode.OPENAI_RESPONSES_STRUCTURED,
                supports_strict_json=True,
            ),
        ),
        patch(
            "ankiops.llm.provider_client.create_structured_adapter",
            return_value=adapter,
        ),
    ):
        client = ProviderClient(
            TaskConfig(
                name="grammar",
                model=TEST_MODEL,
                system_prompt="System prompt for tests",
                prompt="Fix grammar",
            )
        )
        prepared = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(max_output_tokens=200),
            model_id="gpt-5.4",
        )
        with pytest.raises(LlmNoteError) as raised:
            asyncio.run(client.generate_update(prepared_request=prepared))

    assert raised.value.category is LlmNoteErrorCategory.PROVIDER
