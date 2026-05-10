from __future__ import annotations

import pytest

from ankiops.llm.domain.contracts import (
    ContractValidationError,
    build_note_update_contract,
)
from ankiops.llm.domain.payloads import NotePayload


def test_build_note_update_contract_uses_exact_editable_fields() -> None:
    payload = NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={"Question": "Broken", "Answer": "Bad"},
        read_only_fields={"Source": "Book"},
    )

    contract = build_note_update_contract(payload)

    assert contract.schema_name == "note_update"
    assert contract.editable_fields == frozenset({"Question", "Answer"})
    assert contract.json_schema == {
        "type": "object",
        "properties": {
            "note_key": {"type": "string"},
            "edits": {
                "type": "object",
                "properties": {
                    "Question": {"type": "string"},
                    "Answer": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "required": ["note_key", "edits"],
        "additionalProperties": False,
    }


def test_contract_fingerprint_stable_across_field_order() -> None:
    payload_a = NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={"Question": "Broken", "Answer": "Bad"},
    )
    payload_b = NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={"Answer": "Bad", "Question": "Broken"},
    )

    contract_a = build_note_update_contract(payload_a)
    contract_b = build_note_update_contract(payload_b)

    assert contract_a.fingerprint == contract_b.fingerprint


def test_parse_raw_json_validates_editable_fields() -> None:
    payload = NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={"Question": "Broken"},
    )
    contract = build_note_update_contract(payload)

    with pytest.raises(
        ContractValidationError,
        match="edits.Source is not editable",
    ):
        contract.parse_raw_json('{"note_key":"nk-1","edits":{"Source":"Book"}}')
