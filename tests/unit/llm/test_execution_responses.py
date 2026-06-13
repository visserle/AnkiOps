"""Tests for strict LLM structured updates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ankiops.llm.execution import _apply_batch_parsed_response, build_response_model
from ankiops.llm.jobs import LlmItemStatus
from ankiops.llm.planning import EligibleBatch, EligibleCandidate, NotePayload


def _parsed_response(*updates, tag_updates=()):
    return SimpleNamespace(
        updates=[
            SimpleNamespace(note_key=note_key, field=field, value=value)
            for note_key, field, value in updates
        ],
        tag_updates=[
            SimpleNamespace(note_key=note_key, tags=tags)
            for note_key, tags in tag_updates
        ],
    )


def _candidate(llm_qa_config) -> EligibleCandidate:
    return EligibleCandidate(
        item_id=1,
        source="local",
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


def _tag_candidate(
    llm_qa_config,
    *,
    editable: bool = True,
    read_only: bool = False,
    tags: list[str] | None = None,
) -> EligibleCandidate:
    current_tags = ["old"] if tags is None else tags
    return EligibleCandidate(
        item_id=1,
        source="local",
        deck_name="Deck",
        payload=NotePayload(
            note_key="nk-1",
            note_type="AnkiOpsQA",
            editable_fields={},
            read_only_fields={"Question": "Question", "Answer": "Answer"},
            editable_tags=tuple(current_tags) if editable else None,
            read_only_tags=tuple(current_tags) if read_only else None,
        ),
        note_type_config=llm_qa_config,
        serialized_note={
            "note_key": "nk-1",
            "note_type": "AnkiOpsQA",
            "tags": current_tags,
            "fields": {
                "Question": "Question",
                "Answer": "Answer",
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


def test_response_model_accepts_tag_only_updates():
    response_model = build_response_model(
        note_type="AnkiOpsQA",
        payloads=[
            NotePayload(
                note_key="nk-1",
                note_type="AnkiOpsQA",
                editable_fields={},
                read_only_fields={"Question": "Question"},
                editable_tags=(),
            )
        ],
    )

    response_model.model_validate(
        {"tag_updates": [{"note_key": "nk-1", "tags": ["a", "z"]}]}
    )
    with pytest.raises(ValidationError):
        response_model.model_validate(
            {"updates": [{"note_key": "nk-1", "field": "tags", "value": "a z"}]}
        )
    with pytest.raises(ValidationError):
        response_model.model_validate(
            {"tag_updates": [{"note_key": "nk-2", "tags": ["a"]}]}
        )


def test_response_model_accepts_mixed_field_and_tag_updates():
    response_model = build_response_model(
        note_type="AnkiOpsQA",
        payloads=[
            NotePayload(
                note_key="nk-1",
                note_type="AnkiOpsQA",
                editable_fields={"Question": "Broken"},
                editable_tags=("old",),
            )
        ],
    )

    response_model.model_validate(
        {
            "updates": [{"note_key": "nk-1", "field": "Question", "value": "Fixed"}],
            "tag_updates": [{"note_key": "nk-1", "tags": ["new"]}],
        }
    )
    with pytest.raises(ValidationError):
        response_model.model_validate(
            {"updates": [{"note_key": "nk-1", "field": "Question", "value": "Fixed"}]}
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


def test_apply_batch_parsed_response_updates_editable_tags(llm_qa_config):
    candidate = _tag_candidate(llm_qa_config, tags=["old"])

    results = _apply_batch_parsed_response(
        parsed_response=_parsed_response(tag_updates=[("nk-1", ["z", "a", "z"])]),
        batch=_batch(candidate),
    )

    assert len(results) == 1
    assert results[0].status is LlmItemStatus.SUCCEEDED_UPDATED
    assert results[0].changed_fields == ["tags"]
    assert candidate.serialized_note["tags"] == ["a", "z"]


def test_apply_batch_parsed_response_can_clear_tags(llm_qa_config):
    candidate = _tag_candidate(llm_qa_config, tags=["old"])

    results = _apply_batch_parsed_response(
        parsed_response=_parsed_response(tag_updates=[("nk-1", [])]),
        batch=_batch(candidate),
    )

    assert results[0].status is LlmItemStatus.SUCCEEDED_UPDATED
    assert results[0].changed_fields == ["tags"]
    assert candidate.serialized_note["tags"] == []


def test_apply_batch_parsed_response_normalizes_unchanged_tags(llm_qa_config):
    candidate = _tag_candidate(llm_qa_config, tags=["a", "z"])

    results = _apply_batch_parsed_response(
        parsed_response=_parsed_response(tag_updates=[("nk-1", ["z", "a", "z"])]),
        batch=_batch(candidate),
    )

    assert results[0].status is LlmItemStatus.SUCCEEDED_UNCHANGED
    assert results[0].changed_fields == []
    assert candidate.serialized_note["tags"] == ["a", "z"]


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


@pytest.mark.parametrize(
    ("candidate_kwargs", "expected_error"),
    [
        ({"editable": False, "read_only": True}, "read-only tags"),
        ({"editable": False, "read_only": False}, "hidden tags"),
    ],
)
def test_apply_batch_parsed_response_marks_unsafe_tag_updates(
    llm_qa_config,
    candidate_kwargs,
    expected_error,
):
    results = _apply_batch_parsed_response(
        parsed_response=_parsed_response(tag_updates=[("nk-1", ["new"])]),
        batch=_batch(_tag_candidate(llm_qa_config, **candidate_kwargs)),
    )

    assert len(results) == 1
    assert results[0].status is LlmItemStatus.NOTE_ERROR
    assert results[0].changed_fields == []
    assert results[0].error_message is not None
    assert expected_error in results[0].error_message
