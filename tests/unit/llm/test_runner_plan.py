"""Tests for LLM task planning."""

from __future__ import annotations

from ankiops.config import LLM_DB_FILENAME
from ankiops.fs import FileSystemAdapter
from ankiops.llm.runner import plan_task


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
          notes_per_request: 4
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
    assert not (llm_collection / "llm" / LLM_DB_FILENAME).exists()
