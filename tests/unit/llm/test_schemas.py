"""Tests for strict LLM structured updates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ankiops.llm.runner import EligibleBatch, _apply_batch_parsed_response
from ankiops.llm.schemas import build_response_model
from ankiops.llm.types import EligibleCandidate, LlmItemStatus, NotePayload


def _parsed_response(*updates):
    return SimpleNamespace(
        updates=[
            SimpleNamespace(note_key=note_key, field=field, value=value)
            for note_key, field, value in updates
        ]
    )


def _candidate(llm_qa_config) -> EligibleCandidate:
    return EligibleCandidate(
        item_id=1,
        deck_name="Deck",
        payload=NotePayload(
            note_key="nk-1",
            note_type="AnkiOpsQA",
            editable_fields={"Question": "Broken"},
            read_only_fields={"Source": "Book"},
        ),
        note_type_config=llm_qa_config,
        serialized_note={
            "note_key": "nk-1",
            "note_type": "AnkiOpsQA",
            "fields": {
                "Question": "Broken",
                "Answer": "Existing answer",
                "Source": "Book",
                "AI Notes": "Private",
            },
        },
    )


def _batch(candidate: EligibleCandidate) -> EligibleBatch:
    return EligibleBatch(
        note_type=candidate.payload.note_type,
        note_type_config=candidate.note_type_config,
        candidates=(candidate,),
    )


def test_response_model_accepts_only_known_note_keys_and_editable_fields():
    response_model = build_response_model(
        note_type="AnkiOpsQA",
        payloads=[
            NotePayload(
                note_key="nk-1",
                note_type="AnkiOpsQA",
                editable_fields={"Question": "Broken"},
            )
        ],
    )

    response_model.model_validate(
        {"updates": [{"note_key": "nk-1", "field": "Question", "value": "Fixed"}]}
    )
    with pytest.raises(ValidationError):
        response_model.model_validate(
            {"updates": [{"note_key": "nk-2", "field": "Question", "value": "Fixed"}]}
        )
    with pytest.raises(ValidationError):
        response_model.model_validate(
            {"updates": [{"note_key": "nk-1", "field": "Source", "value": "Book"}]}
        )


def test_apply_batch_parsed_response_updates_editable_fields(llm_qa_config):
    candidate = _candidate(llm_qa_config)

    results = _apply_batch_parsed_response(
        parsed_response=_parsed_response(("nk-1", "Question", "Fixed question")),
        batch=_batch(candidate),
    )

    assert len(results) == 1
    assert results[0].status is LlmItemStatus.SUCCEEDED_UPDATED
    assert results[0].changed_fields == ["Question"]
    assert results[0].error_message is None
    assert candidate.serialized_note["fields"] == {
        "Question": "Fixed question",
        "Answer": "Existing answer",
        "Source": "Book",
        "AI Notes": "Private",
    }


def test_apply_batch_parsed_response_rejects_unexpected_note_key(llm_qa_config):
    with pytest.raises(ValueError, match="unexpected note_key 'nk-2'"):
        _apply_batch_parsed_response(
            parsed_response=_parsed_response(("nk-2", "Question", "Fixed question")),
            batch=_batch(_candidate(llm_qa_config)),
        )


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        (
            [
                ("nk-1", "Question", "Fixed question"),
                ("nk-1", "Question", "Fixed again"),
            ],
            "duplicate update for 'Question'",
        ),
        (
            [("nk-1", "Source", "New source")],
            "read-only field 'Source'",
        ),
        (
            [("nk-1", "AI Notes", "New private note")],
            "hidden or unknown field 'AI Notes'",
        ),
    ],
)
def test_apply_batch_parsed_response_marks_unsafe_candidate_updates(
    llm_qa_config,
    updates,
    expected_error,
):
    results = _apply_batch_parsed_response(
        parsed_response=_parsed_response(*updates),
        batch=_batch(_candidate(llm_qa_config)),
    )

    assert len(results) == 1
    assert results[0].status is LlmItemStatus.NOTE_ERROR
    assert results[0].changed_fields == []
    assert results[0].error_message is not None
    assert expected_error in results[0].error_message
