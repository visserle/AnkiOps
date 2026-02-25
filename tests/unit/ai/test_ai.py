"""Unit tests for AI infrastructure and prompt-driven inline editing."""

from __future__ import annotations

from dataclasses import dataclass

from ankiops.ai import (
    PromptConfig,
    load_ai_config,
    load_prompt_config,
    resolve_runtime_ai_config,
    run_inline_prompt_on_serialized_collection,
    save_ai_config,
    select_decks_with_subdecks,
)
from ankiops.db import SQLiteDbAdapter


@dataclass
class _InlineEditor:
    """Deterministic inline JSON editor test double."""

    def edit_note(
        self,
        prompt_config: PromptConfig,
        note_payload: dict[str, object],
    ) -> dict[str, object]:
        _ = prompt_config
        fields = dict(note_payload["fields"])
        if "Question" in fields and isinstance(fields["Question"], str):
            fields["Question"] = fields["Question"].replace("I has", "I have")
        return {
            "note_key": note_payload["note_key"],
            "note_type": note_payload["note_type"],
            "fields": fields,
        }


@dataclass
class _WrongKeyEditor:
    """Editor that returns a mismatched note_key to verify rejection logic."""

    def edit_note(
        self,
        prompt_config: PromptConfig,
        note_payload: dict[str, object],
    ) -> dict[str, object]:
        _ = prompt_config
        return {
            "note_key": "wrong-key",
            "note_type": note_payload["note_type"],
            "fields": dict(note_payload["fields"]),
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
            "model: tiny-model\n"
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
    assert prompt.model == "tiny-model"
    assert prompt.target_fields == ["Question"]
    assert prompt.send_fields == ["Question", "Answer"]
    assert prompt.matches_note_type("AnkiOpsQA")


def test_inline_prompt_updates_only_target_fields():
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
                    }
                ],
            }
        ]
    }

    result = run_inline_prompt_on_serialized_collection(
        serialized_data=data,
        include_decks=["Biology"],
        prompt_config=prompt,
        editor=_InlineEditor(),
    )

    assert result.prompted_notes == 1
    assert result.changed_fields == 1
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
        editor=_WrongKeyEditor(),
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
        editor=_InlineEditor(),
    )

    assert result.prompted_notes == 0
    assert result.changed_fields == 0
    assert "skipped note without note_key" in result.warnings[0]


def test_ai_config_persistence_and_env_resolution(tmp_path, monkeypatch):
    db = SQLiteDbAdapter.load(tmp_path)
    try:
        defaults = load_ai_config(db)
        assert defaults.provider == "local"

        persisted = save_ai_config(
            db,
            provider="remote",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key_env="TEST_KEY",
            timeout_seconds=45,
        )
        assert persisted.provider == "remote"
    finally:
        db.close()

    monkeypatch.setenv("TEST_KEY", "secret")
    runtime = resolve_runtime_ai_config(persisted)
    assert runtime.api_key == "secret"
    assert runtime.timeout_seconds == 45
