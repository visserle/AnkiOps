from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from ankiops.llm.llm_errors import LlmFatalError, LlmNoteError
from ankiops.llm.llm_models import NotePayload, TaskConfig, TaskRequestOptions
from ankiops.llm.model_registry import CLAUDE_SONNET_4_6
from ankiops.llm.prompting import build_system_prompt
from ankiops.llm.provider_client import ProviderClient


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
        model=CLAUDE_SONNET_4_6,
        system_prompt="System prompt for tests",
        prompt="Fix grammar",
        api_key_env="ANTHROPIC_API_KEY",
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
            api_model="claude-sonnet-4-6",
        )
        update_result = asyncio.run(
            client.generate_update(
                prepared_request=prepared_request,
                request_options=TaskRequestOptions(
                    temperature=0,
                    max_output_tokens=200,
                ),
            )
        )

    assert update_result.update.note_key == "nk-1"
    assert update_result.update.edits == {"Question": "Fixed"}
    assert update_result.provider_message_id == "chatcmpl_123"
    assert update_result.provider_model == "claude-sonnet-4-6"
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
            api_model="claude-sonnet-4-6",
        )
        with pytest.raises(LlmNoteError, match=expected_error):
            asyncio.run(
                client.generate_update(
                    prepared_request=prepared_request,
                    request_options=TaskRequestOptions(),
                )
            )


def test_generate_update_retries_connection_error_once_then_succeeds(
    client: ProviderClient,
    note_payload: NotePayload,
    caplog,
):
    response = _response(content='{"note_key":"nk-1","edits":{"Question":"Fixed"}}')
    connection_error = APIConnectionError(
        message="Connection error.",
        request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
    )
    request_options = TaskRequestOptions(
        retries=1,
        retry_backoff_seconds=0.2,
        retry_backoff_jitter=False,
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
            request_options=request_options,
            api_model="claude-sonnet-4-6",
        )
        update_result = asyncio.run(
            client.generate_update(
                prepared_request=prepared_request,
                request_options=request_options,
            )
        )

    assert update_result.retry_count == 1
    assert update_result.update.edits == {"Question": "Fixed"}
    assert create.call_count == 2
    sleep.assert_called_once_with(0.2)
    assert "Retrying nk-1 after connection error (1/1) in 0.20s" in caplog.text


def test_generate_update_fails_after_exhausting_retries(
    client: ProviderClient,
    note_payload: NotePayload,
):
    error = APIStatusError(
        message="upstream unavailable",
        response=httpx.Response(
            503,
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
            request_options=TaskRequestOptions(
                retries=1,
                retry_backoff_seconds=0.25,
                retry_backoff_jitter=False,
            ),
            api_model="claude-sonnet-4-6",
        )
        with pytest.raises(LlmFatalError, match="Provider returned HTTP 503"):
            asyncio.run(
                client.generate_update(
                    prepared_request=prepared_request,
                    request_options=TaskRequestOptions(
                        retries=1,
                        retry_backoff_seconds=0.25,
                        retry_backoff_jitter=False,
                    ),
                )
            )

    assert create.call_count == 2
    sleep.assert_called_once_with(0.25)


def test_missing_api_key_raises_fatal(task_config: TaskConfig, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LlmFatalError, match="ANTHROPIC_API_KEY"):
        ProviderClient(task_config)
