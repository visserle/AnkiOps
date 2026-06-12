"""Small fixtures for LLM unit tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from ankiops.note_types import ANKIOPS_KEY_FIELD, NoteField, NoteType


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
          base_url: https://api.openai.com/v1
          api_key: $OPENAI_API_KEY
          concurrency: 2
          input_usd_per_mtok: 1
          output_usd_per_mtok: 10
        """,
    )
    return tmp_path


@pytest.fixture
def llm_qa_config() -> NoteType:
    return NoteType(
        name="AnkiOpsQA",
        fields=[
            NoteField("Question", "Q:", identifying=True),
            NoteField("Answer", "A:", identifying=True),
            NoteField("Source", "S:", identifying=False),
            NoteField("AI Notes", "AI:", identifying=False),
            ANKIOPS_KEY_FIELD,
        ],
    )


@pytest.fixture
def llm_choice_config() -> NoteType:
    return NoteType(
        name="AnkiOpsChoice",
        fields=[
            NoteField("Question", "Q:", identifying=True),
            NoteField("Choice 1", "C1:", identifying=True),
            NoteField("Choice 2", "C2:", identifying=True),
            NoteField("Answer", "A:", identifying=True),
            NoteField("AI Notes", "AI:", identifying=False),
            ANKIOPS_KEY_FIELD,
        ],
        is_choice=True,
    )
