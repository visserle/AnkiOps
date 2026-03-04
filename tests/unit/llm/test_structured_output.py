from __future__ import annotations

from collections.abc import Callable

import pytest

from ankiops.llm.models import NotePatch, NotePayload
from ankiops.llm.structured_output import (
    NotePatchContract,
    StructuredOutputError,
    build_note_patch_contract,
    parse_note_patch_json,
    validate_note_patch_data,
)


def _parse(contract: NotePatchContract, payload: str) -> NotePatch:
    return parse_note_patch_json(payload, contract=contract)


def _validate(contract: NotePatchContract, payload: object) -> NotePatch:
    return validate_note_patch_data(payload, contract=contract)


@pytest.fixture
def note_payload() -> NotePayload:
    return NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={
            "Question Text": "Broken question",
            "AI Notes": "Private note",
        },
        read_only_fields={"Source": "Book"},
    )


@pytest.fixture
def contract(note_payload: NotePayload) -> NotePatchContract:
    return build_note_patch_contract(note_payload)


def test_build_note_patch_contract_uses_exact_editable_fields(
    contract: NotePatchContract,
):
    assert contract.editable_fields == frozenset({"Question Text", "AI Notes"})
    assert contract.schema == {
        "type": "object",
        "properties": {
            "note_key": {"type": "string"},
            "edits": {
                "type": "object",
                "properties": {
                    "Question Text": {"type": "string"},
                    "AI Notes": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "required": ["note_key", "edits"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("loader", "payload", "expected_edits"),
    [
        (
            _parse,
            '{"note_key":"nk-1","edits":{"AI Notes":"Updated"}}',
            {"AI Notes": "Updated"},
        ),
        (
            _validate,
            {"note_key": "nk-1", "edits": {}},
            {},
        ),
        (
            _validate,
            {"note_key": "nk-1", "edits": {"Question Text": ""}},
            {"Question Text": ""},
        ),
    ],
    ids=["sparse-json-edits", "empty-edits", "empty-string-clears"],
)
def test_note_patch_validation_accepts_valid_payloads(
    contract: NotePatchContract,
    loader: Callable[[NotePatchContract, object], NotePatch],
    payload: object,
    expected_edits: dict[str, str],
):
    patch = loader(contract, payload)

    assert patch.note_key == "nk-1"
    assert patch.edits == expected_edits


@pytest.mark.parametrize(
    ("loader", "payload", "expected_error"),
    [
        (
            _parse,
            "not json",
            "Response was not valid JSON",
        ),
        (
            _parse,
            '{"note_key":"nk-1","edits":{"Question Text":null}}',
            "Structured output validation failed: edits.Question Text must be a string",
        ),
        (
            _validate,
            {"note_key": "nk-1", "edits": {"Question Text": 3}},
            "Structured output validation failed: edits.Question Text must be a string",
        ),
        (
            _validate,
            {"note_key": "nk-1", "edits": "Question Text"},
            "Structured output validation failed: edits must be an object",
        ),
        (
            _validate,
            {"edits": {"Question Text": "Fixed"}},
            "Structured output validation failed: note_key must be a string",
        ),
        (
            _validate,
            {"note_key": "nk-1", "edits": {"Source": "Book"}},
            "Structured output validation failed: edits.Source is not editable",
        ),
        (
            _validate,
            {"note_key": "nk-1", "edits": {}, "extra": "nope"},
            "Structured output validation failed: extra is not allowed",
        ),
    ],
    ids=[
        "invalid-json",
        "null-edit-value",
        "non-string-edit-value",
        "non-object-edits",
        "missing-note-key",
        "read-only-field",
        "unknown-top-level-field",
    ],
)
def test_note_patch_validation_rejects_invalid_payloads(
    contract: NotePatchContract,
    loader: Callable[[NotePatchContract, object], NotePatch],
    payload: object,
    expected_error: str,
):
    with pytest.raises(StructuredOutputError, match=expected_error):
        loader(contract, payload)
