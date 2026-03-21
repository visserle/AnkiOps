from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ankiops.llm.anthropic_models import SONNET
from ankiops.llm.claude import ClaudeClient
from ankiops.llm.errors import LlmFatalError, LlmNoteError
from ankiops.llm.models import (
    LlmAttemptResultType,
    NotePayload,
    TaskConfig,
    TaskRequestOptions,
)
from ankiops.llm.prompting import build_system_prompt


def _extract_note_json(message: str) -> dict[str, object]:
    start = message.index("<note>\n") + len("<note>\n")
    end = message.index("\n</note>")
    return json.loads(message[start:end])


def _response(
    *,
    blocks: list[object],
    stop_reason: str = "end_turn",
    message_id: str = "msg_123",
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 311,
    output_tokens: int = 37,
) -> MagicMock:
    response = MagicMock()
    response.id = message_id
    response.model = model
    response.content = blocks
    response.stop_reason = stop_reason
    response.usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return response


@pytest.fixture
def task_config() -> TaskConfig:
    return TaskConfig(
        name="grammar",
        model=SONNET,
        system_prompt="System prompt for tests",
        prompt="Fix grammar",
        api_key_env="ANTHROPIC_API_KEY",
    )


@pytest.fixture
def client(monkeypatch, task_config: TaskConfig) -> ClaudeClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return ClaudeClient(task_config)


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
    client: ClaudeClient,
    note_payload: NotePayload,
    caplog,
):
    response = _response(
        blocks=[
            SimpleNamespace(
                type="text",
                text='{"note_key":"nk-1","edits":{"Question":"Fixed"}}',
            )
        ]
    )
    with (
        patch.object(
            client,
            "_create_message_with_headers",
            new=AsyncMock(return_value=(response, None, {})),
        ) as create,
        caplog.at_level(logging.DEBUG, logger="ankiops.llm.claude"),
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
                    temperature=0, max_output_tokens=200
                ),
            )
        )

    assert update_result.update.note_key == "nk-1"
    assert update_result.update.edits == {"Question": "Fixed"}
    assert update_result.provider_message_id == "msg_123"
    assert update_result.provider_model == "claude-sonnet-4-6"
    assert update_result.input_tokens == 311
    assert update_result.output_tokens == 37
    assert update_result.latency_ms >= 0

    call_kwargs = create.call_args.args[0]
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["system"] == build_system_prompt(task_config.system_prompt)
    assert "<task>\nFix grammar\n</task>" in call_kwargs["messages"][0]["content"]
    assert _extract_note_json(call_kwargs["messages"][0]["content"]) == {
        "note_key": "nk-1",
        "note_type": "AnkiOpsQA",
        "editable_fields": {"Question": "Broken"},
        "read_only_fields": {"Source": "Book"},
    }
    assert call_kwargs["temperature"] == 0
    assert call_kwargs["max_tokens"] == 200
    assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
    schema = call_kwargs["output_config"]["format"]["schema"]
    assert "Question" in schema["properties"]["edits"]["properties"]
    assert "required" not in schema["properties"]["edits"]
    assert (
        "Requesting update for nk-1 (AnkiOpsQA, editable=1, read_only=1)"
    ) in caplog.text
    assert "Response for nk-1: message_id=msg_123, stop_reason=end_turn" in caplog.text
    assert "input_tokens=311, output_tokens=37" in caplog.text
    assert "retries=0" in caplog.text
    assert "Broken" not in caplog.text
    assert "<task>" not in caplog.text
    assert '{"note_key"' not in caplog.text


@pytest.mark.parametrize(
    ("blocks", "stop_reason", "expected_error"),
    [
        (
            [SimpleNamespace(type="text", text="Cannot comply")],
            "refusal",
            "Provider refused request",
        ),
        (
            [SimpleNamespace(type="thinking", thinking="...")],
            "end_turn",
            "no JSON text output",
        ),
    ],
    ids=["refusal", "missing-json-text"],
)
def test_generate_update_rejects_note_level_failures(
    client: ClaudeClient,
    note_payload: NotePayload,
    blocks: list[object],
    stop_reason: str,
    expected_error: str,
    caplog,
):
    response = _response(blocks=blocks, stop_reason=stop_reason)

    with (
        patch.object(
            client,
            "_create_message_with_headers",
            new=AsyncMock(return_value=(response, None, {})),
        ),
        caplog.at_level(logging.DEBUG, logger="ankiops.llm.claude"),
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

    assert (
        "Requesting update for nk-1 (AnkiOpsQA, editable=1, read_only=1)"
    ) in caplog.text
    assert "Response for nk-1: message_id=msg_123" in caplog.text
    assert "Broken" not in caplog.text
    assert "<task>" not in caplog.text


def test_generate_update_surfaces_unsupported_model_as_fatal(
    client: ClaudeClient,
    note_payload: NotePayload,
):
    from anthropic import APIStatusError

    error = APIStatusError(
        message=(
            "Error code: 400 - {'type': 'error', 'error': "
            "{'type': 'invalid_request_error', 'message': "
            "\"'claude-sonnet-4-6' does not support output format.\"}}"
        ),
        response=MagicMock(status_code=400),
        body=None,
    )

    with patch.object(
        client,
        "_create_message_with_headers",
        new=AsyncMock(side_effect=error),
    ):
        prepared_request = client.prepare_attempt_request(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(),
            api_model="claude-sonnet-4-6",
        )
        with pytest.raises(
            LlmFatalError,
            match="Configured Anthropic model does not support structured outputs",
        ):
            asyncio.run(
                client.generate_update(
                    prepared_request=prepared_request,
                    request_options=TaskRequestOptions(),
                )
            )


def test_generate_update_retries_connection_error_once_then_succeeds(
    client: ClaudeClient,
    note_payload: NotePayload,
    caplog,
):
    from anthropic import APIConnectionError

    response = _response(
        blocks=[
            SimpleNamespace(
                type="text",
                text='{"note_key":"nk-1","edits":{"Question":"Fixed"}}',
            )
        ]
    )
    connection_error = APIConnectionError(
        message="Connection error.",
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
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
        patch("ankiops.llm.claude.asyncio.sleep") as sleep,
        caplog.at_level(logging.WARNING, logger="ankiops.llm.claude"),
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
    client: ClaudeClient,
    note_payload: NotePayload,
):
    from anthropic import APIStatusError

    error = APIStatusError(
        message="upstream unavailable",
        response=MagicMock(status_code=503),
        body=None,
    )

    with (
        patch.object(
            client,
            "_create_message_with_headers",
            new=AsyncMock(side_effect=[error, error]),
        ) as create,
        patch("ankiops.llm.claude.asyncio.sleep") as sleep,
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
        ClaudeClient(task_config)


class _AsyncBatchResultStream:
    def __init__(self, entries: list[object]) -> None:
        self._entries = entries
        self._index = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._entries):
            raise StopAsyncIteration
        entry = self._entries[self._index]
        self._index += 1
        return entry

    async def close(self) -> None:
        self.closed = True


class _BrokenAsyncBatchResultStream:
    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise RuntimeError("decoder blew up")

    async def close(self) -> None:
        self.closed = True


def test_get_batch_results_awaits_results_and_closes_stream(
    client: ClaudeClient,
    note_payload: NotePayload,
):
    prepared_request = client.prepare_attempt_request(
        note_payload=note_payload,
        task_prompt="Fix grammar",
        request_options=TaskRequestOptions(),
        api_model="claude-sonnet-4-6",
    )
    message = _response(
        blocks=[
            SimpleNamespace(
                type="text",
                text='{"note_key":"nk-1","edits":{"Question":"Fixed"}}',
            )
        ]
    )
    stream = _AsyncBatchResultStream(
        [
            SimpleNamespace(
                custom_id="item-1",
                result=SimpleNamespace(type="succeeded", message=message),
            )
        ]
    )

    with patch.object(
        client._client.messages.batches,
        "results",
        new=AsyncMock(return_value=stream),
    ) as results:
        out = asyncio.run(
            client.get_batch_results(
                provider_batch_id="msgbatch_1",
                prepared_by_custom_id={"item-1": prepared_request},
            )
        )

    results.assert_awaited_once_with("msgbatch_1")
    assert stream.closed is True
    assert len(out) == 1
    assert out[0].result_type is LlmAttemptResultType.SUCCEEDED
    assert out[0].outcome is not None
    assert out[0].outcome.update.edits == {"Question": "Fixed"}


def test_get_batch_results_wraps_decoder_errors_as_fatal_and_closes_stream(
    client: ClaudeClient,
):
    stream = _BrokenAsyncBatchResultStream()

    with patch.object(
        client._client.messages.batches,
        "results",
        new=AsyncMock(return_value=stream),
    ):
        with pytest.raises(
            LlmFatalError,
            match="Provider batch results failed: decoder blew up",
        ):
            asyncio.run(
                client.get_batch_results(
                    provider_batch_id="msgbatch_1",
                    prepared_by_custom_id={},
                )
            )

    assert stream.closed is True
