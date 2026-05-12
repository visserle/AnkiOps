"""Small fixtures for LLM unit tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from ankiops.models import ANKIOPS_KEY_FIELD, Field, NoteTypeConfig


@pytest.fixture
def write_file():
    def _write_file(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(content).strip() + "\n", encoding="utf-8")

    return _write_file


@pytest.fixture
def llm_collection(tmp_path: Path, write_file):
    write_file(
        tmp_path / "llm/_models.yaml",
        """
        - model: test
          model_id: gpt-test
          api_url: https://api.openai.com/v1/responses
          api_key: $OPENAI_API_KEY
          concurrency: 2
          input_usd_per_mtok: 1
          output_usd_per_mtok: 10
        """,
    )
    return tmp_path


@pytest.fixture
def llm_qa_config() -> NoteTypeConfig:
    return NoteTypeConfig(
        name="AnkiOpsQA",
        fields=[
            Field("Question", "Q:", identifying=True),
            Field("Answer", "A:", identifying=True),
            Field("Source", "S:", identifying=False),
            Field("AI Notes", "AI:", identifying=False),
            ANKIOPS_KEY_FIELD,
        ],
    )


@pytest.fixture
def llm_choice_config() -> NoteTypeConfig:
    return NoteTypeConfig(
        name="AnkiOpsChoice",
        fields=[
            Field("Question", "Q:", identifying=True),
            Field("Choice 1", "C1:", identifying=True),
            Field("Choice 2", "C2:", identifying=True),
            Field("Answer", "A:", identifying=True),
            Field("AI Notes", "AI:", identifying=False),
            ANKIOPS_KEY_FIELD,
        ],
        is_choice=True,
    )
