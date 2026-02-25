"""Unit tests for AI infrastructure and prompt-driven inline editing."""

from __future__ import annotations

from dataclasses import dataclass

from ankiops.ai import (
    PromptConfig,
    load_models_config,
    load_prompt_config,
    resolve_runtime_ai_config,
    run_inline_prompt_on_serialized_collection,
    select_decks_with_subdecks,
)


@dataclass
class _InlineBatchEditor:
    """Deterministic inline JSON batch editor test double."""

    async def edit_batch(
        self,
        prompt_config: PromptConfig,
        note_payloads: list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        _ = prompt_config
        edited: dict[str, dict[str, object]] = {}
        for note in note_payloads:
            fields = dict(note["fields"])
            if "Question" in fields and isinstance(fields["Question"], str):
                fields["Question"] = fields["Question"].replace("I has", "I have")
            note_key = str(note["note_key"])
            edited[note_key] = {
                "note_key": note_key,
                "note_type": note["note_type"],
                "fields": fields,
            }
        return edited


@dataclass
class _WrongKeyBatchEditor:
    """Editor that returns a mismatched note_key to verify rejection logic."""

    async def edit_batch(
        self,
        prompt_config: PromptConfig,
        note_payloads: list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        _ = prompt_config
        _ = note_payloads
        return {
            "n1": {
                "note_key": "wrong-key",
                "note_type": "AnkiOpsQA",
                "fields": {"Question": "x"},
            }
        }


def test_select_decks_with_subdecks_recursive_include():
    decks = [
        {"name": "Biology", "notes": []},
        {"name": "Biology::Cells", "notes": []},
        {"name": "History", "notes": []},
    ]

    selected = select_decks_with_subdecks(decks, ["Biology"])

    assert [deck["name"] for deck in selected] == ["Biology", "Biology::Cells"]


def test_load_prompt_config_from_yaml(tmp_path):
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
        ),
        encoding="utf-8",
    )

    prompt = load_prompt_config(prompts_dir, "grammar")

    assert prompt.name == "grammar"
    assert prompt.model_profile == "remote-fast"
    assert prompt.target_fields == ["Question"]
    assert prompt.send_fields == ["Question", "Answer"]
    assert prompt.matches_note_type("AnkiOpsQA")


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

    config = load_models_config(prompts_dir)
    assert config.default_profile == "local-fast"
    assert config.profiles["remote-fast"].max_in_flight == 6

    monkeypatch.setenv("TEST_KEY", "secret")
    runtime = resolve_runtime_ai_config(config, profile="remote-fast")
    assert runtime.provider == "remote"
    assert runtime.timeout_seconds == 45
    assert runtime.max_in_flight == 6
    assert runtime.api_key == "secret"


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

    result = run_inline_prompt_on_serialized_collection(
        serialized_data=data,
        include_decks=["Biology"],
        prompt_config=prompt,
        editor=_InlineBatchEditor(),
        batch_size=2,
        max_in_flight=2,
    )

    assert result.prompted_notes == 2
    assert result.changed_fields == 2
    fields = result.changed_decks[0]["notes"][0]["fields"]
    assert fields["Question"] == "I have two lungs."
    assert fields["Answer"] == "Humans have two lungs."


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

    result = run_inline_prompt_on_serialized_collection(
        serialized_data=data,
        include_decks=["Biology"],
        prompt_config=prompt,
        editor=_WrongKeyBatchEditor(),
    )

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

    result = run_inline_prompt_on_serialized_collection(
        serialized_data=data,
        include_decks=["Biology"],
        prompt_config=prompt,
        editor=_InlineBatchEditor(),
    )

    assert result.prompted_notes == 0
    assert result.changed_fields == 0
    assert "skipped note without note_key" in result.warnings[0]
