from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from ankiops.llm.models import (
    NotePayload,
    ProviderConfig,
    SdkType,
    TaskRequestOptions,
)
from ankiops.llm.providers.errors import ProviderFatalError, ProviderNoteError
from ankiops.llm.providers.ollama import OllamaProvider
from ankiops.llm.providers.openai import OpenAIProvider


# ---------------------------------------------------------------------------
# OpenAI provider tests
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    @pytest.fixture()
    def provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        return OpenAIProvider(
            ProviderConfig(
                name="openai-default",
                sdk=SdkType.OPENAI,
                model="gpt-5",
                api_key_env="OPENAI_API_KEY",
            )
        )

    @pytest.fixture()
    def note_payload(self):
        return NotePayload(
            note_key="nk-1",
            note_type="AnkiOpsQA",
            editable_fields={"Question": "Broken"},
            read_only_fields={"Source": "Book"},
        )

    def test_generate_patch_success(self, provider, note_payload):
        mock_response = MagicMock()
        mock_response.output_text = '{"note_key":"nk-1","edits":{"Question":"Fixed"}}'

        with patch.object(
            provider._client.responses, "create", return_value=mock_response
        ) as mock_create:
            patch_result = provider.generate_patch(
                note_payload=note_payload,
                instructions="Fix grammar",
                request_options=TaskRequestOptions(
                    temperature=0, max_output_tokens=200
                ),
                model="gpt-5",
            )

        assert patch_result.note_key == "nk-1"
        assert patch_result.edits == {"Question": "Fixed"}

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5"
        assert call_kwargs["instructions"] == "Fix grammar"
        assert json.loads(call_kwargs["input"]) == {
            "note_key": "nk-1",
            "note_type": "AnkiOpsQA",
            "editable_fields": {"Question": "Broken"},
            "read_only_fields": {"Source": "Book"},
        }
        schema = call_kwargs["text"]["format"]["schema"]
        assert "Question" in schema["properties"]["edits"]["properties"]
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_output_tokens"] == 200

    def test_generate_patch_drops_null_edits(self, provider, note_payload):
        mock_response = MagicMock()
        mock_response.output_text = (
            '{"note_key":"nk-1","edits":{"Question":"Fixed","Source":null}}'
        )
        with patch.object(
            provider._client.responses, "create", return_value=mock_response
        ):
            patch_result = provider.generate_patch(
                note_payload=note_payload,
                instructions="Fix grammar",
                request_options=TaskRequestOptions(),
                model="gpt-5",
            )

        assert patch_result.edits == {"Question": "Fixed"}

    def test_generate_patch_empty_output_raises(self, provider, note_payload):
        mock_response = MagicMock()
        mock_response.output_text = ""
        with patch.object(
            provider._client.responses, "create", return_value=mock_response
        ):
            with pytest.raises(ProviderNoteError, match="no structured output"):
                provider.generate_patch(
                    note_payload=note_payload,
                    instructions="Fix grammar",
                    request_options=TaskRequestOptions(),
                    model="gpt-5",
                )

    def test_generate_patch_omits_empty_read_only_fields(self, provider):
        payload = NotePayload(
            note_key="nk-1",
            note_type="AnkiOpsQA",
            editable_fields={"Question": "Broken"},
        )
        mock_response = MagicMock()
        mock_response.output_text = '{"note_key":"nk-1","edits":{}}'
        with patch.object(
            provider._client.responses, "create", return_value=mock_response
        ) as mock_create:
            provider.generate_patch(
                note_payload=payload,
                instructions="Fix grammar",
                request_options=TaskRequestOptions(),
                model="gpt-5",
            )

        call_kwargs = mock_create.call_args.kwargs
        parsed_input = json.loads(call_kwargs["input"])
        assert "read_only_fields" not in parsed_input

    def test_auth_error_raises_fatal(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "bad-key")
        from openai import AuthenticationError

        provider = OpenAIProvider(
            ProviderConfig(
                name="openai-default",
                sdk=SdkType.OPENAI,
                model="gpt-5",
                api_key_env="OPENAI_API_KEY",
            )
        )
        with patch.object(
            provider._client.responses,
            "create",
            side_effect=AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            ),
        ):
            with pytest.raises(ProviderFatalError, match="authentication"):
                provider.generate_patch(
                    note_payload=NotePayload(
                        note_key="nk-1",
                        note_type="AnkiOpsQA",
                        editable_fields={"Question": "Broken"},
                    ),
                    instructions="Fix grammar",
                    request_options=TaskRequestOptions(),
                    model="gpt-5",
                )

    def test_missing_api_key_raises_fatal(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderFatalError, match="OPENAI_API_KEY"):
            OpenAIProvider(
                ProviderConfig(
                    name="openai-default",
                    sdk=SdkType.OPENAI,
                    model="gpt-5",
                    api_key_env="OPENAI_API_KEY",
                )
            )


# ---------------------------------------------------------------------------
# Ollama provider tests
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    @pytest.fixture()
    def provider(self):
        return OllamaProvider(
            ProviderConfig(
                name="ollama-local",
                sdk=SdkType.OLLAMA,
                model="gpt-oss",
                timeout_seconds=120,
            )
        )

    @pytest.fixture()
    def note_payload(self):
        return NotePayload(
            note_key="nk-1",
            note_type="AnkiOpsQA",
            editable_fields={"Question": "Broken"},
            read_only_fields={"Source": "Book"},
        )

    def test_generate_patch_success(self, provider, note_payload):
        mock_response = MagicMock()
        mock_response.message.content = (
            '{"note_key":"nk-1","edits":{"Question":"Fixed"}}'
        )
        with patch.object(
            provider._client, "chat", return_value=mock_response
        ) as mock_chat:
            patch_result = provider.generate_patch(
                note_payload=note_payload,
                instructions="Fix grammar",
                request_options=TaskRequestOptions(
                    temperature=0, max_output_tokens=128
                ),
                model="gpt-oss",
            )

        assert patch_result.note_key == "nk-1"
        assert patch_result.edits == {"Question": "Fixed"}

        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "gpt-oss"
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        schema = call_kwargs["format"]
        assert "Question" in schema["properties"]["edits"]["properties"]
        assert call_kwargs["options"]["temperature"] == 0
        assert call_kwargs["options"]["num_predict"] == 128

    def test_generate_patch_drops_null_edits(self, provider, note_payload):
        mock_response = MagicMock()
        mock_response.message.content = (
            '{"note_key":"nk-1","edits":{"Question":"Fixed","Source":null}}'
        )
        with patch.object(provider._client, "chat", return_value=mock_response):
            patch_result = provider.generate_patch(
                note_payload=note_payload,
                instructions="Fix grammar",
                request_options=TaskRequestOptions(),
                model="gpt-oss",
            )

        assert patch_result.edits == {"Question": "Fixed"}

    def test_generate_patch_empty_content_raises(self, provider, note_payload):
        mock_response = MagicMock()
        mock_response.message.content = ""
        with patch.object(provider._client, "chat", return_value=mock_response):
            with pytest.raises(ProviderNoteError, match="missing message content"):
                provider.generate_patch(
                    note_payload=note_payload,
                    instructions="Fix grammar",
                    request_options=TaskRequestOptions(),
                    model="gpt-oss",
                )

    def test_connection_error_raises_fatal(self, provider, note_payload):
        with patch.object(
            provider._client,
            "chat",
            side_effect=Exception("Failed to connect to Ollama"),
        ):
            with pytest.raises(ProviderFatalError, match="connect"):
                provider.generate_patch(
                    note_payload=note_payload,
                    instructions="Fix grammar",
                    request_options=TaskRequestOptions(),
                    model="gpt-oss",
                )
