from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from ankiops.fs import FileSystemAdapter
from ankiops.llm.config_loader import load_llm_config_set


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _note_type_configs(tmp_path: Path):
    fs = FileSystemAdapter()
    note_types_dir = tmp_path / "note_types"
    fs.eject_builtin_note_types(note_types_dir)
    return fs.load_note_type_configs(note_types_dir)


def test_load_llm_config_set_accepts_valid_exceptions(tmp_path):
    note_type_configs = _note_type_configs(tmp_path)

    _write(
        tmp_path / "llm/providers/ollama-local.yaml",
        """
        name: ollama-local
        sdk: ollama
        model: gpt-oss
        """,
    )
    _write(
        tmp_path / "llm/tasks/grammar.yaml",
        """
        name: grammar
        prompt: fix grammar
        fields:
          exceptions:
            - read_only: ["Source"]
            - note_types: ["AnkiOpsChoice"]
              read_only: ["Answer"]
            - hidden: ["AI Notes"]
        """,
    )

    config_set = load_llm_config_set(tmp_path, note_type_configs=note_type_configs)

    assert not config_set.provider_errors
    assert not config_set.task_errors
    assert "grammar" in config_set.tasks_by_name
    assert "ollama-local" in config_set.providers_by_name


def test_load_llm_config_set_rejects_invalid_exception_field(tmp_path):
    note_type_configs = _note_type_configs(tmp_path)
    _write(
        tmp_path / "llm/providers/ollama-local.yaml",
        """
        name: ollama-local
        sdk: ollama
        model: gpt-oss
        """,
    )
    _write(
        tmp_path / "llm/tasks/grammar.yaml",
        """
        name: grammar
        prompt: fix grammar
        fields:
          exceptions:
            - note_types: ["AnkiOpsQA"]
              read_only: ["DoesNotExist"]
        """,
    )

    config_set = load_llm_config_set(tmp_path, note_type_configs=note_type_configs)

    assert not config_set.tasks_by_name
    error = next(iter(config_set.task_errors.values()))
    assert "DoesNotExist" in error


def test_load_llm_config_set_parses_decks_include_subdecks(tmp_path):
    note_type_configs = _note_type_configs(tmp_path)
    _write(
        tmp_path / "llm/providers/ollama-local.yaml",
        """
        name: ollama-local
        sdk: ollama
        model: gpt-oss
        """,
    )
    _write(
        tmp_path / "llm/tasks/grammar.yaml",
        (
            "name: grammar\n"
            "prompt: fix grammar\n"
            "decks:\n"
            '  include: ["Parent"]\n'
            "  include_subdecks: false\n"
        ),
    )

    config_set = load_llm_config_set(tmp_path, note_type_configs=note_type_configs)

    assert not config_set.task_errors
    task = config_set.tasks_by_name["grammar"]
    assert task.decks.include == ["Parent"]
    assert task.decks.include_subdecks is False


def test_load_llm_config_set_rejects_non_boolean_include_subdecks(tmp_path):
    note_type_configs = _note_type_configs(tmp_path)
    _write(
        tmp_path / "llm/providers/ollama-local.yaml",
        """
        name: ollama-local
        sdk: ollama
        model: gpt-oss
        """,
    )
    _write(
        tmp_path / "llm/tasks/grammar.yaml",
        (
            "name: grammar\n"
            "prompt: fix grammar\n"
            "decks:\n"
            '  include: ["Parent"]\n'
            '  include_subdecks: "yes"\n'
        ),
    )

    config_set = load_llm_config_set(tmp_path, note_type_configs=note_type_configs)

    assert not config_set.tasks_by_name
    error = next(iter(config_set.task_errors.values()))
    assert "decks.include_subdecks" in error
