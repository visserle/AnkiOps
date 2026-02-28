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


def test_load_llm_config_set_accepts_valid_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    note_type_configs = _note_type_configs(tmp_path)

    _write(
        tmp_path / "llm/providers/ollama-local.yaml",
        """
        version: 1
        name: ollama-local
        type: ollama
        base_url: http://127.0.0.1:11434
        model: gpt-oss
        """,
    )
    _write(
        tmp_path / "llm/providers/openai-default.yaml",
        """
        version: 1
        name: openai-default
        type: openai
        base_url: https://api.openai.com/v1
        api_key_env: OPENAI_API_KEY
        model: gpt-5
        """,
    )
    _write(
        tmp_path / "llm/tasks/grammar.yaml",
        """
        version: 1
        name: grammar
        provider: ollama-local
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
    assert "openai-default" in config_set.providers_by_name


def test_load_llm_config_set_rejects_invalid_exception_field(tmp_path):
    note_type_configs = _note_type_configs(tmp_path)
    _write(
        tmp_path / "llm/providers/ollama-local.yaml",
        """
        version: 1
        name: ollama-local
        type: ollama
        base_url: http://127.0.0.1:11434
        model: gpt-oss
        """,
    )
    _write(
        tmp_path / "llm/tasks/grammar.yaml",
        """
        version: 1
        name: grammar
        provider: ollama-local
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


def test_load_llm_config_set_rejects_missing_openai_env(tmp_path):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    note_type_configs = _note_type_configs(tmp_path)
    _write(
        tmp_path / "llm/providers/openai-default.yaml",
        """
        version: 1
        name: openai-default
        type: openai
        base_url: https://api.openai.com/v1
        api_key_env: OPENAI_API_KEY
        model: gpt-5
        """,
    )

    config_set = load_llm_config_set(tmp_path, note_type_configs=note_type_configs)

    assert not config_set.providers_by_name
    error = next(iter(config_set.provider_errors.values()))
    assert "OPENAI_API_KEY" in error
    monkeypatch.undo()
