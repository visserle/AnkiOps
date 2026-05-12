"""Tests for LLM task config parsing."""

from __future__ import annotations

import pytest

from ankiops.llm.config_loader import load_llm_task_catalog
from ankiops.llm.types import FieldAccess, TaskRequestOptions


def test_load_llm_task_catalog_loads_files_fields_and_request(
    llm_collection,
    write_file,
    llm_qa_config,
    llm_choice_config,
):
    write_file(llm_collection / "llm/prompts/system.md", "System rules")
    write_file(llm_collection / "llm/prompts/user.md", "Fix grammar")
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: !file prompts/system.md
        user_prompt: !file prompts/user.md
        request:
          temperature: 0.25
          max_output_tokens: 512
        fields:
          default_access: read_only
          editable:
            "*": ["AI Notes"]
          hidden:
            "AnkiOpsChoice": ["Answer"]
        """,
    )

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config, llm_choice_config],
    )

    assert not catalog.errors
    task = catalog.tasks_by_name["grammar"]
    assert task.model.model == "test"
    assert task.system_prompt == "System rules"
    assert task.user_prompt == "Fix grammar"
    assert (
        task.system_prompt_path == (llm_collection / "llm/prompts/system.md").resolve()
    )
    assert task.user_prompt_path == (llm_collection / "llm/prompts/user.md").resolve()
    assert task.request == TaskRequestOptions(temperature=0.25, max_output_tokens=512)
    assert task.field_access("AnkiOpsQA", "Question") is FieldAccess.READ_ONLY
    assert task.field_access("AnkiOpsQA", "AI Notes") is FieldAccess.EDITABLE
    assert task.field_access("AnkiOpsChoice", "Answer") is FieldAccess.HIDDEN


@pytest.mark.parametrize(
    ("task_yaml", "expected_error"),
    [
        (
            """
            model: test
            system_prompt: system
            user_prompt: user
            unexpected: true
            """,
            "unknown task key",
        ),
        (
            """
            model: test
            system_prompt: system
            user_prompt: user
            request:
              max_output_tokens: 0
            """,
            "max_output_tokens' must be >= 1",
        ),
        (
            """
            model: test
            system_prompt: !file ../outside.md
            user_prompt: user
            """,
            "!file path must stay within",
        ),
    ],
)
def test_load_llm_task_catalog_reports_invalid_task_config(
    llm_collection,
    write_file,
    llm_qa_config,
    task_yaml,
    expected_error,
):
    write_file(llm_collection / "outside.md", "outside")
    write_file(llm_collection / "llm/grammar.yaml", task_yaml)

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )

    assert not catalog.tasks_by_name
    assert expected_error in catalog.errors[str(llm_collection / "llm/grammar.yaml")]
