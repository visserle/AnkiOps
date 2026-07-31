"""Tests for LLM task config parsing."""

from __future__ import annotations

import pytest

from ankiops.llm import tasks as tasks_module
from ankiops.llm.tasks import FieldAccess, TaskRequestOptions, load_llm_task_catalog


def test_load_llm_task_catalog_loads_prompt_and_input_files_fields_and_request(
    llm_collection,
    write_file,
    llm_qa_config,
    llm_choice_config,
):
    write_file(llm_collection / "llm/prompts/system.md", "System rules")
    write_file(llm_collection / "llm/prompts/user.md", "Fix grammar")
    write_file(llm_collection / "references/rubric.md", "Reference rubric")
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt:
          file: prompts/system.md
        user_prompt:
          file: prompts/user.md
        input_files:
          - ../references/rubric.md
        request:
          max_notes_per_request: 3
          temperature: 0.25
          reasoning: low
        fields:
          default_access: read_only
          editable:
            "*": ["AI Notes"]
          hidden:
            "AnkiOpsChoice": ["Answer"]
        tags: editable
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
    assert task.input_files == ((llm_collection / "references/rubric.md").resolve(),)
    assert task.request == TaskRequestOptions(
        max_notes_per_request=3,
        temperature=0.25,
        reasoning="low",
    )
    assert task.field_access("AnkiOpsQA", "Question") is FieldAccess.READ_ONLY
    assert task.field_access("AnkiOpsQA", "AI Notes") is FieldAccess.EDITABLE
    assert task.field_access("AnkiOpsChoice", "Answer") is FieldAccess.HIDDEN
    assert task.tag_access is FieldAccess.EDITABLE


def test_load_llm_task_catalog_defaults_tags_hidden(
    llm_collection,
    write_file,
    llm_qa_config,
):
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        request:
          max_notes_per_request: 1
        """,
    )

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )

    assert not catalog.errors
    assert catalog.tasks_by_name["grammar"].tag_access is FieldAccess.HIDDEN


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
              max_output_tokens: 512
            """,
            "unknown request key(s): max_output_tokens",
        ),
        (
            """
            model: test
            system_prompt: system
            user_prompt: user
            request:
              max_notes_per_request: 0
            """,
            "request.max_notes_per_request' must be >= 1",
        ),
        (
            """
            model: test
            system_prompt: system
            user_prompt: user
            request:
              max_notes_per_request: false
            """,
            "request.max_notes_per_request' must be an integer",
        ),
        (
            """
            model: test
            system_prompt: system
            user_prompt: user
            input_files: reference.md
            """,
            "'input_files' must be a list",
        ),
        (
            """
            model: test
            system_prompt: system
            user_prompt: user
            request:
              max_notes_per_request: 1
              reasoning: extreme
            """,
            "request.reasoning' must be one of",
        ),
        (
            """
            model: test
            system_prompt: system
            user_prompt: user
            tags: read-only
            request:
              max_notes_per_request: 1
            """,
            "'tags' must be one of",
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
    write_file(llm_collection / "llm/grammar.yaml", task_yaml)

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )

    assert not catalog.tasks_by_name
    assert expected_error in catalog.errors[str(llm_collection / "llm/grammar.yaml")]


def test_load_llm_task_catalog_requires_max_notes_per_request(
    llm_collection,
    write_file,
    llm_qa_config,
):
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        """,
    )

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )

    assert not catalog.tasks_by_name
    assert (
        "request.max_notes_per_request' is required"
        in catalog.errors[str(llm_collection / "llm/grammar.yaml")]
    )


def test_load_llm_task_catalog_rejects_legacy_file_tag(
    llm_collection,
    write_file,
    llm_qa_config,
):
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: !file prompt.md
        user_prompt: user
        request:
          max_notes_per_request: 1
        """,
    )

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )

    assert not catalog.tasks_by_name
    assert "invalid YAML" in catalog.errors[str(llm_collection / "llm/grammar.yaml")]


@pytest.mark.parametrize(
    ("prompt_value", "expected_error"),
    [
        ("{}", "'system_prompt.file' is required"),
        ("{file: prompt.md, extra: true}", "unknown 'system_prompt' key"),
        ("{file: ''}", "'system_prompt.file' must be a non-empty relative path"),
    ],
)
def test_load_llm_task_catalog_rejects_invalid_prompt_file_mapping(
    llm_collection,
    write_file,
    llm_qa_config,
    prompt_value,
    expected_error,
):
    write_file(
        llm_collection / "llm/grammar.yaml",
        f"""
        model: test
        system_prompt: {prompt_value}
        user_prompt: user
        request:
          max_notes_per_request: 1
        """,
    )

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )

    assert expected_error in catalog.errors[str(llm_collection / "llm/grammar.yaml")]


def test_load_llm_task_catalog_rejects_paths_outside_collection(
    llm_collection,
    write_file,
    llm_qa_config,
):
    write_file(llm_collection.parent / "outside.md", "outside")
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt:
          file: ../../outside.md
        user_prompt: user
        request:
          max_notes_per_request: 1
        """,
    )

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )

    assert (
        "must stay within" in catalog.errors[str(llm_collection / "llm/grammar.yaml")]
    )


def test_load_llm_task_catalog_rejects_absolute_input_file_path(
    llm_collection,
    write_file,
    llm_qa_config,
):
    input_path = llm_collection / "reference.md"
    write_file(input_path, "reference")
    write_file(
        llm_collection / "llm/grammar.yaml",
        f"""
        model: test
        system_prompt: system
        user_prompt: user
        input_files:
          - {input_path}
        request:
          max_notes_per_request: 1
        """,
    )

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )

    assert (
        "must be a relative path"
        in catalog.errors[str(llm_collection / "llm/grammar.yaml")]
    )


def test_load_llm_task_catalog_rejects_input_symlink_outside_collection(
    llm_collection,
    write_file,
    llm_qa_config,
):
    outside = llm_collection.parent / "outside.md"
    write_file(outside, "outside")
    link = llm_collection / "reference.md"
    link.symlink_to(outside)
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        input_files:
          - ../reference.md
        request:
          max_notes_per_request: 1
        """,
    )

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )

    assert (
        "must stay within" in catalog.errors[str(llm_collection / "llm/grammar.yaml")]
    )


def test_load_llm_task_catalog_rejects_missing_empty_and_duplicate_input_files(
    llm_collection,
    write_file,
    llm_qa_config,
):
    empty = llm_collection / "empty.txt"
    empty.touch()
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        input_files:
          - ../empty.txt
        request:
          max_notes_per_request: 1
        """,
    )
    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )
    assert (
        "file must be non-empty"
        in catalog.errors[str(llm_collection / "llm/grammar.yaml")]
    )

    write_file(llm_collection / "reference.txt", "reference")
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        input_files:
          - ../reference.txt
          - .././reference.txt
        request:
          max_notes_per_request: 1
        """,
    )
    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )
    assert "duplicate path" in catalog.errors[str(llm_collection / "llm/grammar.yaml")]

    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        input_files:
          - ../missing.txt
        request:
          max_notes_per_request: 1
        """,
    )
    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )
    assert "file not found" in catalog.errors[str(llm_collection / "llm/grammar.yaml")]


def test_load_llm_task_catalog_enforces_combined_input_file_size(
    llm_collection,
    write_file,
    llm_qa_config,
    monkeypatch,
):
    monkeypatch.setattr(tasks_module, "_MAX_INPUT_FILE_BYTES", 10)
    write_file(llm_collection / "first.txt", "12345")
    write_file(llm_collection / "second.txt", "1234")
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        input_files:
          - ../first.txt
          - ../second.txt
        request:
          max_notes_per_request: 1
        """,
    )

    catalog = load_llm_task_catalog(
        llm_collection,
        note_type_configs=[llm_qa_config],
    )

    assert (
        "combined 'input_files' size"
        in catalog.errors[str(llm_collection / "llm/grammar.yaml")]
    )
