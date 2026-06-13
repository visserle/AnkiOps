"""Safe Anki note type conversion for AnkiOpsConnect."""

from __future__ import annotations

from .actions import ANKIOPS_KEY_FIELD_NAME, AnkiOpsConnectActionError
from .collection import card_snapshot, load_notes, note_field_values, note_type_name
from .note_types import field_names, note_type_by_name, note_type_id, template_names


def convert_notes_to_note_type_action(col, params: dict) -> dict:
    return convert_notes_to_note_type(
        col=col,
        note_ids=params.get("noteIds") or [],
        old_note_type=params.get("oldNoteType") or "",
        new_note_type=params.get("newNoteType") or "",
    )

def convert_notes_to_note_type(
    col,
    note_ids,
    old_note_type: str,
    new_note_type: str,
) -> dict:
    note_ids = [int(note_id) for note_id in note_ids]
    if not note_ids:
        return {"changed": 0}
    if not old_note_type or not new_note_type:
        raise AnkiOpsConnectActionError("oldNoteType and newNoteType are required.")

    old_notetype = note_type_by_name(col.models, old_note_type)
    new_notetype = note_type_by_name(col.models, new_note_type)
    if old_notetype is None:
        raise AnkiOpsConnectActionError(f"Old note type not found: {old_note_type}")
    if new_notetype is None:
        raise AnkiOpsConnectActionError(f"New note type not found: {new_note_type}")

    old_fields = field_names(old_notetype)
    new_fields = field_names(new_notetype)
    if ANKIOPS_KEY_FIELD_NAME not in old_fields:
        raise AnkiOpsConnectActionError(
            f"Old note type lacks {ANKIOPS_KEY_FIELD_NAME}."
        )
    if ANKIOPS_KEY_FIELD_NAME not in new_fields:
        raise AnkiOpsConnectActionError(
            f"New note type lacks {ANKIOPS_KEY_FIELD_NAME}."
        )
    field_map = _exact_name_map(old_fields, new_fields, "field")
    template_map = _exact_name_map(
        template_names(old_notetype),
        template_names(new_notetype),
        "template",
    )

    notes = load_notes(col, note_ids)
    for note_id, note in notes.items():
        actual_note_type = note_type_name(note)
        if actual_note_type != old_note_type:
            raise AnkiOpsConnectActionError(
                f"Note {note_id} uses '{actual_note_type}', "
                f"expected '{old_note_type}'."
            )

    before_notes = {
        note_id: note_field_values(note, old_notetype)
        for note_id, note in notes.items()
    }
    before_cards = card_snapshot(col, note_ids)

    info = col.models.change_notetype_info(
        old_notetype_id=note_type_id(old_notetype),
        new_notetype_id=note_type_id(new_notetype),
    )
    request = info.input
    _replace_repeated(request.note_ids, note_ids)
    _replace_repeated(request.new_fields, field_map)
    _replace_repeated(request.new_templates, template_map)
    col.models.change_notetype_of_notes(request)

    after_notes = load_notes(col, note_ids)
    after_cards = card_snapshot(col, note_ids)
    failures = []
    for note_id, note in after_notes.items():
        actual_note_type = note_type_name(note)
        if actual_note_type != new_note_type:
            failures.append(f"note {note_id} still uses '{actual_note_type}'")
            continue
        after_fields = note_field_values(note, new_notetype)
        if after_fields != before_notes[note_id]:
            failures.append(f"note {note_id} fields changed")

    if set(after_notes) != set(notes):
        failures.append("converted note set changed")
    if after_cards != before_cards:
        failures.append("card scheduling or identity changed")

    if failures:
        raise AnkiOpsConnectActionError(
            "Post-conversion verification failed: " + "; ".join(failures)
        )

    return {"changed": len(note_ids)}


def _exact_name_map(old_names: list[str], new_names: list[str], kind: str) -> list[int]:
    if len(set(old_names)) != len(old_names) or len(set(new_names)) != len(new_names):
        raise AnkiOpsConnectActionError(f"Duplicate {kind} names are not supported.")
    if set(old_names) != set(new_names):
        raise AnkiOpsConnectActionError(
            f"Cannot convert note type: {kind} names differ "
            f"({old_names!r} -> {new_names!r})."
        )
    return [old_names.index(name) for name in new_names]


def _replace_repeated(target, values: list[int]) -> None:
    del target[:]
    target.extend(values)
