"""Unit tests for AI config, parser, HTTP client, and prompt execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
import pytest
import respx

from ankiops.ai import (
    AIConfigError,
    AIRequestError,
    AIResponseError,
    InlineEditedNote,
    InlineNotePayload,
    OpenAICompatibleAsyncEditor,
    PromptConfig,
    PromptConfigError,
    PromptExecutionError,
    PromptRunner,
    PromptRunOptions,
    RuntimeAIConfig,
    load_model_profiles,
    load_prompt,
    resolve_runtime_config,
)
from ankiops.ai.validators import normalize_batch_response, parse_json_object


@dataclass
class _InlineBatchEditor:
    """Deterministic inline JSON batch editor test double."""

    async def edit_notes(
        self,
        prompt: PromptConfig,
        notes: list[InlineNotePayload],
    ) -> dict[str, InlineEditedNote]:
        _ = prompt
        edited: dict[str, InlineEditedNote] = {}
        for note in notes:
            fields = dict(note.fields)
            if "Question" in fields and isinstance(fields["Question"], str):
                fields["Question"] = fields["Question"].replace("I has", "I have")
            edited[note.note_key] = InlineEditedNote.from_parts(
                note_key=note.note_key,
                note_type=note.note_type,
                fields=fields,
            )
        return edited


@dataclass
class _WrongKeyBatchEditor:
    """Editor that returns a mismatched note_key to verify rejection logic."""

    async def edit_notes(
        self,
        prompt: PromptConfig,
        notes: list[InlineNotePayload],
    ) -> dict[str, InlineEditedNote]:
        _ = prompt
        _ = notes
        return {
            "n1": InlineEditedNote.from_parts(
                note_key="wrong-key",
                note_type="AnkiOpsQA",
                fields={"Question": "x"},
            )
        }


def _runtime() -> RuntimeAIConfig:
    return RuntimeAIConfig(
        profile="remote-fast",
        provider="remote",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="ANKIOPS_AI_API_KEY",
        timeout_seconds=10,
        max_in_flight=3,
        api_key="test-key",
    )


def _prompt() -> PromptConfig:
    return PromptConfig(
        name="grammar",
        prompt="Return inline JSON.",
        target_fields=["Question"],
        send_fields=["Question"],
        note_types=["AnkiOps*"],
    )


def _notes_payload() -> list[InlineNotePayload]:
    return [
        InlineNotePayload(
            note_key="n1",
            note_type="AnkiOpsQA",
            fields={"Question": "I has two lungs."},
        )
    ]


def test_load_prompt_from_yaml(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "grammar.yaml").write_text(
        (
            "name: grammar\n"
            "model_profile: remote-fast\n"
            "prompt: |\n"
            "  Return inline JSON.\n"
            "fields_to_edit:\n"
            "  - Question\n"
            "fields_to_send:\n"
            "  - Question\n"
            "  - Answer\n"
            "note_types:\n"
            "  - AnkiOps*\n"
            "temperature: 0.1\n"
        ),
        encoding="utf-8",
    )

    prompt = load_prompt(prompts_dir, "grammar")

    assert prompt.name == "grammar"
    assert prompt.model_profile == "remote-fast"
    assert prompt.target_fields == ["Question"]
    assert prompt.send_fields == ["Question", "Answer"]
    assert prompt.temperature == pytest.approx(0.1)
    assert prompt.matches_note_type("AnkiOpsQA")


def test_load_prompt_rejects_out_of_range_temperature(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "grammar.yaml").write_text(
        "prompt: Return inline JSON.\ntemperature: 3\n",
        encoding="utf-8",
    )

    with pytest.raises(PromptConfigError, match="temperature"):
        load_prompt(prompts_dir, "grammar")


def test_load_models_config_and_resolve_runtime(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "models.yaml").write_text(
        (
            "default_profile: local-fast\n"
            "profiles:\n"
            "  local-fast:\n"
            "    provider: local\n"
            "    model: llama3.1:8b\n"
            "    base_url: http://localhost:11434/v1\n"
            "    api_key_env: TEST_KEY\n"
            "    timeout_seconds: 60\n"
            "    max_in_flight: 3\n"
            "  remote-fast:\n"
            "    provider: remote\n"
            "    model: gpt-4o-mini\n"
            "    base_url: https://api.openai.com/v1\n"
            "    api_key_env: TEST_KEY\n"
            "    timeout_seconds: 45\n"
            "    max_in_flight: 6\n"
        ),
        encoding="utf-8",
    )

    config = load_model_profiles(prompts_dir)
    assert config.default_profile == "local-fast"
    assert config.profiles["remote-fast"].max_in_flight == 6

    monkeypatch.setenv("TEST_KEY", "secret")
    runtime = resolve_runtime_config(config, profile="remote-fast")
    assert runtime.provider == "remote"
    assert runtime.timeout_seconds == 45
    assert runtime.max_in_flight == 6
    assert runtime.api_key == "secret"


def test_resolve_runtime_rejects_non_positive_overrides(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "models.yaml").write_text(
        (
            "profiles:\n"
            "  local-fast:\n"
            "    provider: local\n"
            "    model: llama3.1:8b\n"
            "    base_url: http://localhost:11434/v1\n"
        ),
        encoding="utf-8",
    )
    config = load_model_profiles(prompts_dir)

    with pytest.raises(AIConfigError, match="max_in_flight"):
        resolve_runtime_config(config, max_in_flight=0)


def test_inline_prompt_updates_only_target_fields_batch_mode():
    prompt = PromptConfig(
        name="grammar",
        prompt="Return inline JSON.",
        target_fields=["Question"],
        send_fields=["Question", "Answer"],
        note_types=["AnkiOps*"],
    )
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": "n1",
                        "note_type": "AnkiOpsQA",
                        "fields": {
                            "Question": "I has two lungs.",
                            "Answer": "Humans have two lungs.",
                        },
                    },
                    {
                        "note_key": "n2",
                        "note_type": "AnkiOpsQA",
                        "fields": {
                            "Question": "I has one heart.",
                            "Answer": "Humans have one heart.",
                        },
                    },
                ],
            }
        ]
    }

    options = PromptRunOptions(include_decks=["Biology"], batch_size=2, max_in_flight=2)
    result = PromptRunner(_InlineBatchEditor()).run(data, prompt, options=options)

    assert result.prompted_notes == 2
    assert result.changed_fields == 2
    fields = result.changed_decks[0]["notes"][0]["fields"]
    assert fields["Question"] == "I have two lungs."
    assert fields["Answer"] == "Humans have two lungs."


def test_inline_prompt_recursive_deck_selection():
    prompt = PromptConfig(
        name="grammar",
        prompt="Return inline JSON.",
        target_fields=["Question"],
        send_fields=["Question"],
        note_types=["AnkiOps*"],
    )
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": "n1",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has two lungs."},
                    }
                ],
            },
            {
                "name": "Biology::Cells",
                "notes": [
                    {
                        "note_key": "n2",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has one nucleus."},
                    }
                ],
            },
            {
                "name": "History",
                "notes": [
                    {
                        "note_key": "n3",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has a timeline."},
                    }
                ],
            },
        ]
    }
    result = PromptRunner(_InlineBatchEditor()).run(
        data,
        prompt,
        options=PromptRunOptions(include_decks=["Biology"]),
    )

    assert result.processed_decks == 2
    assert result.prompted_notes == 2
    assert len(result.changed_decks) == 2


def test_inline_prompt_rejects_mismatched_note_key():
    prompt = PromptConfig(
        name="grammar",
        prompt="Return inline JSON.",
        target_fields=["Question"],
        send_fields=["Question"],
        note_types=["AnkiOps*"],
    )
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": "n1",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has two lungs."},
                    }
                ],
            }
        ]
    }

    result = PromptRunner(_WrongKeyBatchEditor()).run(data, prompt)

    assert result.changed_fields == 0
    assert result.warnings
    assert "note_key mismatch" in result.warnings[0]


def test_inline_prompt_skips_notes_without_note_key():
    prompt = PromptConfig(
        name="grammar",
        prompt="Return inline JSON.",
        target_fields=["Question"],
        send_fields=["Question"],
        note_types=["AnkiOps*"],
    )
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": None,
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has two lungs."},
                    }
                ],
            }
        ]
    }

    result = PromptRunner(_InlineBatchEditor()).run(data, prompt)

    assert result.prompted_notes == 0
    assert result.changed_fields == 0
    assert "skipped note without note_key" in result.warnings[0]


def test_inline_prompt_invalid_batch_size_raises():
    prompt = _prompt()
    with pytest.raises(PromptExecutionError, match="batch_size"):
        PromptRunner(_InlineBatchEditor()).run(
            {"decks": []},
            prompt,
            options=PromptRunOptions(batch_size=0),
        )


def test_normalize_batch_response_accepts_markdown_fenced_json():
    content = (
        "```json\n"
        '{"notes":{"n1":{"note_key":"n1","fields":{"Question":"I have two lungs."}}}}\n'
        "```"
    )

    normalized = normalize_batch_response(content)
    assert normalized["n1"].fields["Question"] == "I have two lungs."


def test_parse_json_object_extracts_prefix_and_suffix_noise():
    content = (
        "Sure, here you go:\n"
        '{"notes":[{"note_key":"n1","fields":{"Question":"I have two lungs."}}]}\n'
        "Thanks!"
    )

    parsed = parse_json_object(content)
    assert parsed["notes"][0]["note_key"] == "n1"


@respx.mock
def test_client_retries_transient_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("ankiops.ai.client.DEFAULT_RETRY_MIN_SECONDS", 0.0)
    monkeypatch.setattr("ankiops.ai.client.DEFAULT_RETRY_MAX_SECONDS", 0.0)

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"notes":{"n1":{"note_key":"n1",'
                                    '"note_type":"AnkiOpsQA",'
                                    '"fields":{"Question":"I have two lungs."}}}}'
                                )
                            }
                        }
                    ]
                },
            ),
        ]
    )

    result = asyncio.run(
        OpenAICompatibleAsyncEditor(_runtime()).edit_notes(_prompt(), _notes_payload())
    )
    assert route.call_count == 2
    assert result["n1"].fields["Question"] == "I have two lungs."


@respx.mock
def test_client_fail_fast_non_retryable_400():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "bad request"}},
        )
    )

    with pytest.raises(AIRequestError, match="400"):
        asyncio.run(
            OpenAICompatibleAsyncEditor(_runtime()).edit_notes(
                _prompt(),
                _notes_payload(),
            )
        )
    assert route.call_count == 1


@respx.mock
def test_client_raises_response_error_on_non_json_response():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text="not-json",
        )
    )

    with pytest.raises(AIResponseError, match="valid JSON"):
        asyncio.run(
            OpenAICompatibleAsyncEditor(_runtime()).edit_notes(
                _prompt(),
                _notes_payload(),
            )
        )
    assert route.call_count == 1


@respx.mock
def test_client_raises_response_error_on_missing_assistant_text():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": []}}]},
        )
    )

    with pytest.raises(AIResponseError, match="assistant text content"):
        asyncio.run(
            OpenAICompatibleAsyncEditor(_runtime()).edit_notes(
                _prompt(),
                _notes_payload(),
            )
        )
    assert route.call_count == 1
