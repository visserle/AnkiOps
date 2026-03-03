from __future__ import annotations

import pytest

from ankiops.llm.models import NotePayload
from ankiops.llm.structured_output import (
    StructuredOutputError,
    build_note_patch_contract,
    parse_note_patch_json,
    validate_note_patch_data,
)


@pytest.fixture()
def note_payload():
    return NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={
            "Question Text": "Broken question",
            "AI Notes": "Private note",
        },
        read_only_fields={"Source": "Book"},
    )


def test_build_note_patch_contract_uses_exact_editable_fields(note_payload):
    contract = build_note_patch_contract(note_payload)

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


def test_parse_note_patch_json_accepts_sparse_edits(note_payload):
    contract = build_note_patch_contract(note_payload)

    patch = parse_note_patch_json(
        '{"note_key":"nk-1","edits":{"AI Notes":"Updated"}}',
        contract=contract,
    )

    assert patch.note_key == "nk-1"
    assert patch.edits == {"AI Notes": "Updated"}


def test_validate_note_patch_data_allows_empty_edits(note_payload):
    contract = build_note_patch_contract(note_payload)

    patch = validate_note_patch_data(
        {"note_key": "nk-1", "edits": {}},
        contract=contract,
    )

    assert patch.edits == {}


def test_validate_note_patch_data_allows_empty_string_field_clears(note_payload):
    contract = build_note_patch_contract(note_payload)

    patch = validate_note_patch_data(
        {"note_key": "nk-1", "edits": {"Question Text": ""}},
        contract=contract,
    )

    assert patch.edits == {"Question Text": ""}


def test_parse_note_patch_json_rejects_null_values(note_payload):
    contract = build_note_patch_contract(note_payload)
    error_match = (
        "Structured output validation failed: "
        "edits.Question Text must be a string"
    )

    with pytest.raises(StructuredOutputError, match=error_match):
        parse_note_patch_json(
            '{"note_key":"nk-1","edits":{"Question Text":null}}',
            contract=contract,
        )


def test_validate_note_patch_data_rejects_non_string_edit_values(note_payload):
    contract = build_note_patch_contract(note_payload)
    error_match = (
        "Structured output validation failed: "
        "edits.Question Text must be a string"
    )

    with pytest.raises(StructuredOutputError, match=error_match):
        validate_note_patch_data(
            {"note_key": "nk-1", "edits": {"Question Text": 3}},
            contract=contract,
        )


def test_validate_note_patch_data_rejects_non_object_edits(note_payload):
    contract = build_note_patch_contract(note_payload)

    with pytest.raises(
        StructuredOutputError,
        match="Structured output validation failed: edits must be an object",
    ):
        validate_note_patch_data(
            {"note_key": "nk-1", "edits": "Question Text"},
            contract=contract,
        )


def test_validate_note_patch_data_rejects_missing_note_key(note_payload):
    contract = build_note_patch_contract(note_payload)

    with pytest.raises(
        StructuredOutputError,
        match="Structured output validation failed: note_key must be a string",
    ):
        validate_note_patch_data(
            {"edits": {"Question Text": "Fixed"}},
            contract=contract,
        )


def test_validate_note_patch_data_rejects_read_only_or_unknown_fields(note_payload):
    contract = build_note_patch_contract(note_payload)

    with pytest.raises(
        StructuredOutputError,
        match="Structured output validation failed: edits.Source is not editable",
    ):
        validate_note_patch_data(
            {"note_key": "nk-1", "edits": {"Source": "Book"}},
            contract=contract,
        )


def test_validate_note_patch_data_rejects_top_level_unknown_fields(note_payload):
    contract = build_note_patch_contract(note_payload)

    with pytest.raises(
        StructuredOutputError,
        match="Structured output validation failed: extra is not allowed",
    ):
        validate_note_patch_data(
            {"note_key": "nk-1", "edits": {}, "extra": "nope"},
            contract=contract,
        )
