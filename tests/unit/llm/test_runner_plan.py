"""Tests for LLM task planning."""

from __future__ import annotations

import pytest

from ankiops.config import LLM_DB_FILENAME
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.llm.runner import plan_task
from ankiops.llm.types import FieldAccess


def _init_collection_db(collection_dir):
    db = SQLiteDbAdapter.open(collection_dir)
    try:
        db.set_profile_name("test")
    finally:
        db.close()


def test_plan_task_summarizes_scope_surface_and_does_not_persist(
    llm_collection,
    write_file,
    monkeypatch,
):
    FileSystemAdapter().eject_builtin_note_types(llm_collection / "note_types")
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: |
          You are a strict editor.
        user_prompt: |
          Fix grammar.
        request:
          max_notes_per_request: 4
        fields:
          default_access: editable
          read_only:
            "AnkiOpsChoice": ["Answer"]
          hidden:
            "*": ["AI Notes"]
        """,
    )
    captured_scope = {}

    def fake_serialize(collection_dir, *, deck=None, no_subdecks=False, note_types_dir):
        captured_scope.update(
            {
                "collection_dir": collection_dir,
                "deck": deck,
                "no_subdecks": no_subdecks,
                "note_types_dir": note_types_dir,
            }
        )
        return {
            "decks": [
                {
                    "source": "local",
                    "name": "Deck",
                    "notes": [
                        {
                            "note_key": "nk-1",
                            "note_type": "AnkiOpsQA",
                            "fields": {
                                "Question": "broken question",
                                "Answer": "answer",
                                "Source": "book",
                                "AI Notes": "private",
                            },
                        },
                        {
                            "note_key": "nk-2",
                            "note_type": "AnkiOpsChoice",
                            "fields": {
                                "Question": "pick one",
                                "Choice 1": "yes",
                                "Choice 2": "no",
                                "Answer": "1",
                                "AI Notes": "private",
                            },
                        },
                        {
                            "note_key": "nk-3",
                            "note_type": "AnkiOpsQA",
                            "fields": {
                                "Question": "another broken question",
                                "Answer": "another answer",
                                "Source": "article",
                                "AI Notes": "private",
                            },
                        },
                    ],
                }
            ]
        }

    monkeypatch.setattr("ankiops.llm.runner.serialize", fake_serialize)

    plan = plan_task(collection_dir=llm_collection, task_name="grammar")

    assert captured_scope == {
        "collection_dir": llm_collection,
        "deck": None,
        "no_subdecks": False,
        "note_types_dir": llm_collection / "note_types",
    }
    assert plan.task_name == "grammar"
    assert plan.summary.decks_seen == 1
    assert plan.summary.decks_matched == 1
    assert plan.summary.notes_seen == 3
    assert plan.summary.eligible == 3
    assert plan.requests_estimate == 2
    assert plan.input_tokens_estimate > 0
    expected_cost = plan.model.estimate_cost(
        input_tokens=plan.input_tokens_estimate,
        output_tokens=plan.input_tokens_estimate,
    )
    assert expected_cost is not None
    assert plan.format_cost_estimate() == expected_cost.format()
    assert "<system>\nYou are a strict editor.\n</system>" in plan.format_full_prompt()

    surface_by_type = {surface.note_type: surface for surface in plan.field_surface}
    assert "AI Notes" in surface_by_type["AnkiOpsQA"].hidden_fields
    assert "Answer" in surface_by_type["AnkiOpsChoice"].read_only_fields
    assert surface_by_type["AnkiOpsQA"].tag_access is FieldAccess.HIDDEN
    assert not (llm_collection / "llm" / LLM_DB_FILENAME).exists()


def test_plan_task_summarizes_autotagger_tag_surface_and_skips_contextless_notes(
    llm_collection,
    write_file,
    monkeypatch,
):
    FileSystemAdapter().eject_builtin_note_types(llm_collection / "note_types")
    write_file(
        llm_collection / "llm/autotagger.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: Tag the note.
        fields:
          default_access: hidden
          read_only:
            "*": ["Text", "Question", "Answer"]
        tags: editable
        request:
          max_notes_per_request: 4
        """,
    )

    def fake_serialize(collection_dir, *, deck=None, no_subdecks=False, note_types_dir):
        return {
            "decks": [
                {
                    "source": "local",
                    "name": "Deck",
                    "notes": [
                        {
                            "note_key": "qa-1",
                            "note_type": "AnkiOpsQA",
                            "tags": ["old"],
                            "fields": {
                                "Question": "What?",
                                "Answer": "That.",
                                "Source": "book",
                            },
                        },
                        {
                            "note_key": "cloze-1",
                            "note_type": "AnkiOpsCloze",
                            "tags": [],
                            "fields": {
                                "Text": "{{c1::Cloze}} text",
                                "Source": "article",
                            },
                        },
                        {
                            "note_key": "rev-1",
                            "note_type": "AnkiOpsReversed",
                            "tags": ["old"],
                            "fields": {
                                "Front": "front",
                                "Back": "back",
                            },
                        },
                    ],
                }
            ]
        }

    monkeypatch.setattr("ankiops.llm.runner.serialize", fake_serialize)

    plan = plan_task(collection_dir=llm_collection, task_name="autotagger")

    assert plan.summary.notes_seen == 3
    assert plan.summary.eligible == 2
    assert plan.summary.skipped == 1
    assert plan.requests_estimate == 2
    surface_by_type = {surface.note_type: surface for surface in plan.field_surface}
    assert surface_by_type["AnkiOpsQA"].tag_access is FieldAccess.EDITABLE
    assert surface_by_type["AnkiOpsCloze"].tag_access is FieldAccess.EDITABLE
    assert surface_by_type["AnkiOpsReversed"].tag_access is FieldAccess.EDITABLE
    assert surface_by_type["AnkiOpsQA"].read_only_fields == ["Question", "Answer"]
    assert surface_by_type["AnkiOpsCloze"].read_only_fields == ["Text"]
    assert surface_by_type["AnkiOpsReversed"].candidate_notes == 0


def test_plan_task_discovers_collab_notes_with_sources(llm_collection, write_file):
    _init_collection_db(llm_collection)
    fs = FileSystemAdapter()
    fs.eject_builtin_note_types(llm_collection / "note_types")
    collab_root = llm_collection / "collab" / "owner" / "repo"
    fs.eject_builtin_note_types(collab_root / "note_types")
    write_file(
        llm_collection / "Local.md",
        """
        <!-- note_key: local-1 -->
        Q: local question
        A: local answer
        """,
    )
    write_file(
        collab_root / "Shared.md",
        """
        <!-- note_key: collab-1 -->
        Q: collab question
        A: collab answer
        """,
    )
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: Fix grammar.
        request:
          max_notes_per_request: 4
        fields:
          default_access: editable
          read_only:
            "*AnkiOpsQA": ["Answer"]
        """,
    )

    plan = plan_task(collection_dir=llm_collection, task_name="grammar")

    surface = {
        (item.source, item.note_type): item
        for item in plan.field_surface
    }
    assert ("local", "AnkiOpsQA") in surface
    assert ("collab/owner/repo", "collab/owner/repo/AnkiOpsQA") in surface
    assert surface[("local", "AnkiOpsQA")].candidate_notes == 1
    assert surface[
        ("collab/owner/repo", "collab/owner/repo/AnkiOpsQA")
    ].candidate_notes == 1


def test_plan_task_rejects_explicit_note_type_pattern_after_deck_filter(
    llm_collection,
    write_file,
):
    _init_collection_db(llm_collection)
    FileSystemAdapter().eject_builtin_note_types(llm_collection / "note_types")
    write_file(
        llm_collection / "Deck.md",
        """
        <!-- note_key: qa-1 -->
        Q: question
        A: answer
        """,
    )
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: Fix grammar.
        request:
          max_notes_per_request: 4
        fields:
          default_access: editable
          read_only:
            "AnkiOpsChoice": ["Answer"]
        """,
    )

    with pytest.raises(ValueError, match="matched no notes after deck filtering"):
        plan_task(collection_dir=llm_collection, task_name="grammar")
