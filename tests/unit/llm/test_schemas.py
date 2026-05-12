"""Tests for strict LLM structured updates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ankiops.llm.runner import _apply_parsed_response
from ankiops.llm.schemas import build_response_model
from ankiops.llm.types import EligibleCandidate, NotePayload


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


def test_apply_parsed_response_updates_editable_fields(llm_qa_config):
    candidate = _candidate(llm_qa_config)

    changed_fields = _apply_parsed_response(
        parsed_response=_parsed_response(("nk-1", "Question", "Fixed question")),
        candidate=candidate,
    )

    assert changed_fields == ["Question"]
    assert candidate.serialized_note["fields"] == {
        "Question": "Fixed question",
        "Answer": "Existing answer",
        "Source": "Book",
        "AI Notes": "Private",
    }


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        (
            [("nk-2", "Question", "Fixed question")],
            "unexpected note_key 'nk-2'",
        ),
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
def test_apply_parsed_response_rejects_unsafe_updates(
    llm_qa_config,
    updates,
    expected_error,
):
    with pytest.raises(ValueError, match=expected_error):
        _apply_parsed_response(
            parsed_response=_parsed_response(*updates),
            candidate=_candidate(llm_qa_config),
        )
