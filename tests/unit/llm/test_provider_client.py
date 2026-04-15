from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from ankiops.llm.llm_errors import LlmFatalError, LlmNoteError
from ankiops.llm.model_registry import ModelSpec
from ankiops.llm.prompting import build_system_prompt
from ankiops.llm.provider_client import (
    ProviderClient,
    _retry_after_seconds,
    _throttle_delay_from_headers,
)
from ankiops.llm.task_types import NotePayload, TaskConfig, TaskRequestOptions

TEST_MODEL = ModelSpec(
    model="claude-sonnet-4-6",
    model_id="claude-sonnet-4-6",
    provider="anthropic",
    base_url="https://api.anthropic.com/v1/",
    api_key="$ANTHROPIC_API_KEY",
)

GROQ_TEST_MODEL = ModelSpec(
    model="groq-oss-120b",
    model_id="openai/gpt-oss-120b",
    provider="groq",
    base_url="https://api.groq.com/openai/v1",
    api_key="$GROQ_API_KEY",
)

OLLAMA_TEST_MODEL = ModelSpec(
    model="gemma4-e2b",
    model_id="gemma4:e2b",
    provider="ollama",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def _extract_note_json(message: str) -> dict[str, object]:
    start = message.index("<note>\n") + len("<note>\n")
    end = message.index("\n</note>")
    return json.loads(message[start:end])


def _response(
    *,
    content: str | None,
    finish_reason: str = "stop",
    message_id: str = "chatcmpl_123",
    model: str = "claude-sonnet-4-6",
    prompt_tokens: int = 311,
    completion_tokens: int = 37,
):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id=message_id,
        model=model,
        choices=[choice],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        model_dump_json=lambda: "{}",
    )


@pytest.fixture
def task_config() -> TaskConfig:
    return TaskConfig(
        name="grammar",
        model=TEST_MODEL,
        system_prompt="System prompt for tests",
        prompt="Fix grammar",
    )


@pytest.fixture
def client(monkeypatch, task_config: TaskConfig) -> ProviderClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return ProviderClient(task_config)


@pytest.fixture
def note_payload() -> NotePayload:
    return NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={"Question": "Broken"},
        read_only_fields={"Source": "Book"},
    )


def test_generate_update_builds_request_and_returns_update(
    task_config: TaskConfig,
    client: ProviderClient,
    note_payload: NotePayload,
    caplog,
):
    response = _response(content='{"note_key":"nk-1","edits":{"Question":"Fixed"}}')
    with (
        patch.object(
            client,
            "_create_message_with_headers",
            new=AsyncMock(return_value=(response, None, {})),
        ) as create,
        caplog.at_level(logging.DEBUG, logger="ankiops.llm.provider_client"),
    ):
        prepared_request = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(temperature=0, max_output_tokens=200),
            model_id="claude-sonnet-4-6",
        )
        update_result = asyncio.run(
            client.generate_update(
                prepared_request=prepared_request,
            )
        )

    assert update_result.update.note_key == "nk-1"
    assert update_result.update.edits == {"Question": "Fixed"}
    assert update_result.provider_message_id == "chatcmpl_123"
    assert update_result.response_model_id == "claude-sonnet-4-6"
    assert update_result.input_tokens == 311
    assert update_result.output_tokens == 37
    assert update_result.latency_ms >= 0

    call_kwargs = create.call_args.args[0]
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][0]["content"] == build_system_prompt(
        task_config.system_prompt
    )
    assert call_kwargs["messages"][1]["role"] == "user"
    assert "<task>\nFix grammar\n</task>" in call_kwargs["messages"][1]["content"]
    assert _extract_note_json(call_kwargs["messages"][1]["content"]) == {
        "note_key": "nk-1",
        "note_type": "AnkiOpsQA",
        "editable_fields": {"Question": "Broken"},
        "read_only_fields": {"Source": "Book"},
    }
    assert call_kwargs["temperature"] == 0
    assert call_kwargs["max_tokens"] == 200
    assert call_kwargs["response_format"]["type"] == "json_schema"
    assert call_kwargs["response_format"]["json_schema"]["name"] == "note_update"
    assert call_kwargs["response_format"]["json_schema"]["strict"] is True

    assert (
        "Requesting update for nk-1 (AnkiOpsQA, editable=1, read_only=1)" in caplog.text
    )
    assert "Response for nk-1: message_id=chatcmpl_123" in caplog.text


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        (
            _response(content=None, finish_reason="content_filter"),
            "Provider refused request",
        ),
        (
            _response(content=None),
            "no JSON text output",
        ),
    ],
    ids=["content-filter", "missing-json-text"],
)
def test_generate_update_rejects_note_level_failures(
    client: ProviderClient,
    note_payload: NotePayload,
    response,
    expected_error: str,
):
    with patch.object(
        client,
        "_create_message_with_headers",
        new=AsyncMock(return_value=(response, None, {})),
    ):
        prepared_request = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(),
            model_id="claude-sonnet-4-6",
        )
        with pytest.raises(LlmNoteError, match=expected_error):
            asyncio.run(
                client.generate_update(
                    prepared_request=prepared_request,
                )
            )


def test_generate_update_retries_connection_error_once_then_succeeds(
    monkeypatch,
    note_payload: NotePayload,
    caplog,
):
    retrying_model = replace(
        TEST_MODEL,
        retries=1,
        retry_backoff_seconds=0.2,
        retry_backoff_jitter=False,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = ProviderClient(
        TaskConfig(
            name="grammar",
            model=retrying_model,
            system_prompt="System prompt for tests",
            prompt="Fix grammar",
        )
    )

    response = _response(content='{"note_key":"nk-1","edits":{"Question":"Fixed"}}')
    connection_error = APIConnectionError(
        message="Connection error.",
        request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
    )

    with (
        patch.object(
            client,
            "_create_message_with_headers",
            new=AsyncMock(side_effect=[connection_error, (response, None, {})]),
        ) as create,
        patch("ankiops.llm.provider_client.asyncio.sleep") as sleep,
        caplog.at_level(logging.WARNING, logger="ankiops.llm.provider_client"),
    ):
        prepared_request = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(),
            model_id="claude-sonnet-4-6",
        )
        update_result = asyncio.run(
            client.generate_update(
                prepared_request=prepared_request,
            )
        )

    assert update_result.retry_count == 1
    assert update_result.update.edits == {"Question": "Fixed"}
    assert create.call_count == 2
    sleep.assert_called_once_with(0.2)
    assert "Retrying nk-1 after connection error (1/1) in 0.20s" in caplog.text


@pytest.mark.parametrize("status_code", [408, 409, 503])
def test_generate_update_fails_after_exhausting_retries(
    monkeypatch,
    note_payload: NotePayload,
    status_code: int,
):
    retrying_model = replace(
        TEST_MODEL,
        retries=1,
        retry_backoff_seconds=0.25,
        retry_backoff_jitter=False,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = ProviderClient(
        TaskConfig(
            name="grammar",
            model=retrying_model,
            system_prompt="System prompt for tests",
            prompt="Fix grammar",
        )
    )

    error = APIStatusError(
        message="upstream unavailable",
        response=httpx.Response(
            status_code,
            request=httpx.Request(
                "POST", "https://api.example.com/v1/chat/completions"
            ),
        ),
        body=None,
    )

    with (
        patch.object(
            client,
            "_create_message_with_headers",
            new=AsyncMock(side_effect=[error, error]),
        ) as create,
        patch("ankiops.llm.provider_client.asyncio.sleep") as sleep,
    ):
        prepared_request = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(),
            model_id="claude-sonnet-4-6",
        )
        with pytest.raises(
            LlmFatalError,
            match=f"Provider returned HTTP {status_code}",
        ):
            asyncio.run(
                client.generate_update(
                    prepared_request=prepared_request,
                )
            )

    assert create.call_count == 2
    sleep.assert_called_once_with(0.25)


def test_generate_update_handles_non_retryable_quota_429_without_sleep(
    monkeypatch,
    note_payload: NotePayload,
):
    retrying_model = replace(
        TEST_MODEL,
        retries=3,
        retry_backoff_seconds=0.25,
        retry_backoff_jitter=False,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = ProviderClient(
        TaskConfig(
            name="grammar",
            model=retrying_model,
            system_prompt="System prompt for tests",
            prompt="Fix grammar",
        )
    )

    error = APIStatusError(
        message="insufficient_quota",
        response=httpx.Response(
            429,
            request=httpx.Request(
                "POST", "https://api.example.com/v1/chat/completions"
            ),
        ),
        body={"error": {"type": "insufficient_quota"}},
    )

    with (
        patch.object(
            client,
            "_create_message_with_headers",
            new=AsyncMock(side_effect=[error]),
        ) as create,
        patch("ankiops.llm.provider_client.asyncio.sleep") as sleep,
    ):
        prepared_request = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(),
            model_id="claude-sonnet-4-6",
        )
        with pytest.raises(
            LlmFatalError,
            match="Provider quota or billing limit reached",
        ):
            asyncio.run(
                client.generate_update(
                    prepared_request=prepared_request,
                )
            )

    assert create.call_count == 1
    sleep.assert_not_called()


def test_generate_update_respects_retry_after_seconds_hint(
    monkeypatch,
    note_payload: NotePayload,
):
    retrying_model = replace(
        TEST_MODEL,
        retries=1,
        retry_backoff_seconds=0.25,
        retry_backoff_jitter=False,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = ProviderClient(
        TaskConfig(
            name="grammar",
            model=retrying_model,
            system_prompt="System prompt for tests",
            prompt="Fix grammar",
        )
    )

    response = _response(content='{"note_key":"nk-1","edits":{"Question":"Fixed"}}')
    error = APIStatusError(
        message="rate limited",
        response=httpx.Response(
            429,
            headers={"Retry-After": "5"},
            request=httpx.Request(
                "POST", "https://api.example.com/v1/chat/completions"
            ),
        ),
        body={"error": {"type": "rate_limit_exceeded"}},
    )

    with (
        patch.object(
            client,
            "_create_message_with_headers",
            new=AsyncMock(side_effect=[error, (response, None, {})]),
        ) as create,
        patch("ankiops.llm.provider_client.asyncio.sleep") as sleep,
    ):
        prepared_request = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(),
            model_id="claude-sonnet-4-6",
        )
        asyncio.run(
            client.generate_update(
                prepared_request=prepared_request,
            )
        )

    assert create.call_count == 2
    sleep.assert_called_once_with(5.0)


def test_generate_update_prefers_retry_after_ms_hint(
    monkeypatch,
    note_payload: NotePayload,
):
    retrying_model = replace(
        TEST_MODEL,
        retries=1,
        retry_backoff_seconds=0.25,
        retry_backoff_jitter=False,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = ProviderClient(
        TaskConfig(
            name="grammar",
            model=retrying_model,
            system_prompt="System prompt for tests",
            prompt="Fix grammar",
        )
    )

    response = _response(content='{"note_key":"nk-1","edits":{"Question":"Fixed"}}')
    error = APIStatusError(
        message="rate limited",
        response=httpx.Response(
            429,
            headers={"Retry-After": "5", "Retry-After-Ms": "1500"},
            request=httpx.Request(
                "POST", "https://api.example.com/v1/chat/completions"
            ),
        ),
        body={"error": {"type": "rate_limit_exceeded"}},
    )

    with (
        patch.object(
            client,
            "_create_message_with_headers",
            new=AsyncMock(side_effect=[error, (response, None, {})]),
        ) as create,
        patch("ankiops.llm.provider_client.asyncio.sleep") as sleep,
    ):
        prepared_request = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(),
            model_id="claude-sonnet-4-6",
        )
        asyncio.run(
            client.generate_update(
                prepared_request=prepared_request,
            )
        )

    assert create.call_count == 2
    sleep.assert_called_once_with(1.5)


def test_retry_after_seconds_parses_http_date_header():
    retry_after_date = format_datetime(
        datetime.now(timezone.utc) + timedelta(seconds=60),
        usegmt=True,
    )
    error = APIStatusError(
        message="rate limited",
        response=httpx.Response(
            429,
            headers={"Retry-After": retry_after_date},
            request=httpx.Request(
                "POST", "https://api.example.com/v1/chat/completions"
            ),
        ),
        body={"error": {"type": "rate_limit_exceeded"}},
    )

    parsed = _retry_after_seconds(error)
    assert parsed is not None
    assert parsed >= 45
    assert parsed <= 61


def test_throttle_delay_from_headers_parses_compact_reset_duration():
    delay = _throttle_delay_from_headers(
        {
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "2m30.5s",
        }
    )
    assert delay == pytest.approx(150.5)


def test_throttle_delay_from_headers_prefers_retry_after_ms():
    delay = _throttle_delay_from_headers(
        {
            "retry-after": "5",
            "retry-after-ms": "1250",
        }
    )
    assert delay == pytest.approx(1.25)


def test_throttle_delay_from_headers_supports_iso_reset_timestamp():
    reset_at = (
        (datetime.now(timezone.utc) + timedelta(seconds=30))
        .isoformat()
        .replace("+00:00", "Z")
    )
    delay = _throttle_delay_from_headers(
        {
            "anthropic-ratelimit-requests-remaining": "0",
            "anthropic-ratelimit-requests-reset": reset_at,
        }
    )
    assert delay is not None
    assert delay >= 20
    assert delay <= 31


def test_missing_api_key_raises_fatal(task_config: TaskConfig, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LlmFatalError, match="ANTHROPIC_API_KEY"):
        ProviderClient(task_config)


def test_literal_api_key_is_used_directly(monkeypatch):
    monkeypatch.delenv("sk-ant-literal-123", raising=False)
    literal_model = ModelSpec(
        model="claude-sonnet-4-6",
        model_id="claude-sonnet-4-6",
        provider="anthropic",
        base_url="https://api.anthropic.com/v1/",
        api_key="sk-ant-literal-123",
    )
    task = TaskConfig(
        name="grammar",
        model=literal_model,
        system_prompt="System prompt for tests",
        prompt="Fix grammar",
    )

    with patch("ankiops.llm.provider_client.AsyncOpenAI") as client_class:
        ProviderClient(task)

    assert client_class.call_args.kwargs["api_key"] == "sk-ant-literal-123"


def test_prepare_attempt_request_disables_strict_for_groq(monkeypatch, note_payload):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    task = TaskConfig(
        name="grammar",
        model=GROQ_TEST_MODEL,
        system_prompt="System prompt for tests",
        prompt="Fix grammar",
    )
    client = ProviderClient(task)

    prepared_request = client.prepare_attempt_request(
        note_payload=note_payload,
        task_prompt="Fix grammar",
        request_options=TaskRequestOptions(),
        model_id="openai/gpt-oss-120b",
    )

    assert (
        prepared_request.request_params["response_format"]["json_schema"]["strict"]
        is False
    )


def test_prepare_attempt_request_includes_ollama_think_toggle(note_payload):
    task = TaskConfig(
        name="grammar",
        model=OLLAMA_TEST_MODEL,
        system_prompt="System prompt for tests",
        prompt="Fix grammar",
    )
    client = ProviderClient(task)

    prepared_request = client.prepare_attempt_request(
        note_payload=note_payload,
        task_prompt="Fix grammar",
        request_options=TaskRequestOptions(),
        model_id="gemma4:e2b",
    )

    assert prepared_request.request_params["extra_body"] == {"think": False}


def test_generate_update_succeeds_for_ollama_with_thinking_disabled(note_payload):
    task = TaskConfig(
        name="grammar",
        model=OLLAMA_TEST_MODEL,
        system_prompt="System prompt for tests",
        prompt="Fix grammar",
    )
    client = ProviderClient(task)
    response = _response(
        content='{"note_key":"nk-1","edits":{"Question":"Fixed"}}',
        model="gemma4:e2b",
    )

    with patch.object(
        client,
        "_create_message_with_headers",
        new=AsyncMock(return_value=(response, None, {})),
    ) as create:
        prepared_request = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(),
            model_id="gemma4:e2b",
        )
        update_result = asyncio.run(
            client.generate_update(
                prepared_request=prepared_request,
            )
        )

    call_kwargs = create.call_args.args[0]
    assert call_kwargs["extra_body"] == {"think": False}
    assert update_result.update.note_key == "nk-1"
    assert update_result.update.edits == {"Question": "Fixed"}


def test_prepare_attempt_request_omits_think_for_non_ollama(client, note_payload):
    prepared_request = client.prepare_attempt_request(
        note_payload=note_payload,
        task_prompt="Fix grammar",
        request_options=TaskRequestOptions(),
        model_id="claude-sonnet-4-6",
    )

    assert "extra_body" not in prepared_request.request_params
