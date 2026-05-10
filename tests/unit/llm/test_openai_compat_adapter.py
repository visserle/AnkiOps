from __future__ import annotations

import asyncio
from types import SimpleNamespace

from httpx import Request
from openai import APIConnectionError

from ankiops.llm.domain.contracts import build_note_update_contract
from ankiops.llm.domain.outcomes import ProviderOutcomeKind
from ankiops.llm.domain.payloads import NotePayload
from ankiops.llm.providers.adapter_base import AdapterRequest
from ankiops.llm.providers.openai_compat_adapter import OpenAICompatStructuredAdapter


class _FakeCompletions:
    def __init__(self, response: object):
        self._response = response
        self.last_kwargs: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        self.last_kwargs = dict(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response: object):
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _RaisingCompletions:
    def __init__(self, error: Exception):
        self._error = error

    async def create(self, **kwargs: object) -> object:
        raise self._error


class _RaisingClient:
    def __init__(self, error: Exception):
        self.chat = SimpleNamespace(completions=_RaisingCompletions(error))
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
        system_prompt="System",
        task_prompt="Fix grammar",
        model_id="claude-sonnet-4-6",
        max_output_tokens=128,
    )


def _response(content: str | None, finish_reason: str = "stop") -> object:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id="chatcmpl_123",
        model="claude-sonnet-4-6",
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
        model_dump_json=lambda: "{}",
    )


def test_generate_success_for_openai_compat_adapter() -> None:
    fake_client = _FakeClient(
        _response('{"note_key":"nk-1","edits":{"Question":"Fixed"}}')
    )
    adapter = OpenAICompatStructuredAdapter(
        api_key="key",
        base_url="https://api.anthropic.com/v1",
        provider="anthropic",
        client=fake_client,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.SUCCESS
    assert outcome.update is not None
    assert outcome.update.edits == {"Question": "Fixed"}

    kwargs = fake_client.chat.completions.last_kwargs
    assert kwargs is not None
    response_format = kwargs["response_format"]
    assert isinstance(response_format, dict)
    json_schema = response_format["json_schema"]
    assert isinstance(json_schema, dict)
    assert json_schema["strict"] is True


def test_generate_refusal_for_content_filter() -> None:
    fake_client = _FakeClient(_response(None, finish_reason="content_filter"))
    adapter = OpenAICompatStructuredAdapter(
        api_key="key",
        base_url="https://api.anthropic.com/v1",
        provider="anthropic",
        client=fake_client,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.REFUSAL


def test_generate_validation_error_for_bad_json() -> None:
    fake_client = _FakeClient(_response("{not valid json}"))
    adapter = OpenAICompatStructuredAdapter(
        api_key="key",
        base_url="https://api.anthropic.com/v1",
        provider="anthropic",
        client=fake_client,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.VALIDATION_ERROR
    assert outcome.error_message is not None


def test_connection_error_exhaustion_is_provider_error() -> None:
    fake_client = _RaisingClient(
        APIConnectionError(
            message="Connection error.",
            request=Request("POST", "https://api.example.com/v1/chat/completions"),
        )
    )
    adapter = OpenAICompatStructuredAdapter(
        api_key="key",
        base_url="https://api.example.com/v1",
        provider="groq",
        retries=0,
        retry_backoff_jitter=False,
        client=fake_client,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.PROVIDER_ERROR
    assert outcome.error_message is not None
