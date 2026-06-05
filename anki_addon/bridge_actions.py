"""AnkiOps bridge actions.

This module stays importable without ``aqt`` so bridge behaviour can be tested
outside Anki.
"""

from __future__ import annotations

from collections.abc import Callable

BRIDGE_VERSION = 1
ANKIOPS_KEY_FIELD_NAME = "AnkiOps Key"
CARD_SNAPSHOT_COLUMNS = (
    "id",
    "nid",
    "ord",
    "did",
    "queue",
    "type",
    "due",
    "ivl",
    "factor",
    "reps",
    "lapses",
    "left",
    "odue",
    "odid",
)

BridgeAction = Callable[[object, dict], object]


class BridgeActionError(Exception):
    """Raised when a bridge action cannot be completed safely."""


def dispatch_bridge_action(col, action: str, params: dict):
    try:
        handler = BRIDGE_ACTIONS[action]
    except KeyError as error:
        raise BridgeActionError(f"Unknown AnkiOps bridge action: {action}") from error
    return handler(col, params)


def version_action(_col, _params: dict) -> dict:
    return {"version": BRIDGE_VERSION}


def change_notes_notetype_action(col, params: dict) -> dict:
    return change_notes_notetype(
        col=col,
        note_ids=params.get("noteIds") or [],
        old_model=params.get("oldModel") or "",
        new_model=params.get("newModel") or "",
    )


BRIDGE_ACTIONS: dict[str, BridgeAction] = {
    "version": version_action,
    "changeNotesNotetype": change_notes_notetype_action,
}


def change_notes_notetype(col, note_ids, old_model: str, new_model: str) -> dict:
    note_ids = [int(note_id) for note_id in note_ids]
    if not note_ids:
        return {"changed": 0}
    if not old_model or not new_model:
        raise BridgeActionError("oldModel and newModel are required.")

    old_notetype = _model_by_name(col.models, old_model)
    new_notetype = _model_by_name(col.models, new_model)
    if old_notetype is None:
        raise BridgeActionError(f"Old note type not found: {old_model}")
    if new_notetype is None:
        raise BridgeActionError(f"New note type not found: {new_model}")

    old_fields = _field_names(old_notetype)
    new_fields = _field_names(new_notetype)
    if ANKIOPS_KEY_FIELD_NAME not in old_fields:
        raise BridgeActionError(f"Old note type lacks {ANKIOPS_KEY_FIELD_NAME}.")
    if ANKIOPS_KEY_FIELD_NAME not in new_fields:
        raise BridgeActionError(f"New note type lacks {ANKIOPS_KEY_FIELD_NAME}.")
    field_map = _exact_name_map(old_fields, new_fields, "field")
    template_map = _exact_name_map(
        _template_names(old_notetype),
        _template_names(new_notetype),
        "template",
    )

    notes = _load_notes(col, note_ids)
    for note_id, note in notes.items():
        actual_model = _note_model_name(note)
        if actual_model != old_model:
            raise BridgeActionError(
                f"Note {note_id} uses '{actual_model}', expected '{old_model}'."
            )

    before_notes = {
        note_id: _note_field_values(note, old_notetype)
        for note_id, note in notes.items()
    }
    before_cards = _card_snapshot(col, note_ids)

    info = col.models.change_notetype_info(
        old_notetype_id=_model_id(old_notetype),
        new_notetype_id=_model_id(new_notetype),
    )
    request = info.input
    _replace_repeated(request.note_ids, note_ids)
    _replace_repeated(request.new_fields, field_map)
    _replace_repeated(request.new_templates, template_map)
    col.models.change_notetype_of_notes(request)

    after_notes = _load_notes(col, note_ids)
    after_cards = _card_snapshot(col, note_ids)
    failures = []
    for note_id, note in after_notes.items():
        actual_model = _note_model_name(note)
        if actual_model != new_model:
            failures.append(f"note {note_id} still uses '{actual_model}'")
            continue
        after_fields = _note_field_values(note, new_notetype)
        if after_fields != before_notes[note_id]:
            failures.append(f"note {note_id} fields changed")

    if set(after_notes) != set(notes):
        failures.append("converted note set changed")
    if after_cards != before_cards:
        failures.append("card scheduling or identity changed")

    if failures:
        raise BridgeActionError(
            "Post-conversion verification failed: " + "; ".join(failures)
        )

    return {"changed": len(note_ids)}


def _model_by_name(models, name: str):
    return models.by_name(name)


def _model_id(model) -> int:
    try:
        return int(model["id"])
    except Exception as error:
        raise BridgeActionError("Note type is missing an id.") from error


def _field_names(model) -> list[str]:
    return [_item_name(field) for field in model.get("flds", [])]


def _template_names(model) -> list[str]:
    return [_item_name(template) for template in model.get("tmpls", [])]


def _item_name(item) -> str:
    return item.get("name") or ""


def _exact_name_map(old_names: list[str], new_names: list[str], kind: str) -> list[int]:
    if len(set(old_names)) != len(old_names) or len(set(new_names)) != len(new_names):
        raise BridgeActionError(f"Duplicate {kind} names are not supported.")
    if set(old_names) != set(new_names):
        raise BridgeActionError(
            f"Cannot convert note type: {kind} names differ "
            f"({old_names!r} -> {new_names!r})."
        )
    return [old_names.index(name) for name in new_names]


def _load_notes(col, note_ids: list[int]) -> dict[int, object]:
    notes = {}
    for note_id in note_ids:
        try:
            note = col.get_note(note_id)
        except Exception as error:
            raise BridgeActionError(f"Note not found: {note_id}") from error
        if note is None:
            raise BridgeActionError(f"Note not found: {note_id}")
        notes[note_id] = note
    return notes


def _note_model_name(note) -> str:
    notetype = note.note_type()
    if notetype:
        return notetype.get("name", "")
    return ""


def _note_field_values(note, model) -> dict[str, str]:
    field_names = _field_names(model)
    raw_fields = getattr(note, "fields", None)
    if isinstance(raw_fields, dict):
        return {name: raw_fields.get(name, "") for name in field_names}
    if raw_fields is not None:
        return {
            name: raw_fields[index] if index < len(raw_fields) else ""
            for index, name in enumerate(field_names)
        }
    return {}


def _card_snapshot(col, note_ids: list[int]) -> dict[int, tuple]:
    if hasattr(col, "ankiops_bridge_cards_snapshot"):
        return col.ankiops_bridge_cards_snapshot(note_ids)

    placeholders = ",".join("?" for _ in note_ids)
    columns = ", ".join(CARD_SNAPSHOT_COLUMNS)
    rows = col.db.all(
        f"select {columns} from cards where nid in ({placeholders})",
        *note_ids,
    )
    snapshot = {}
    for row in rows:
        values = tuple(row)
        snapshot[int(values[0])] = values
    return snapshot


def _replace_repeated(target, values: list[int]) -> None:
    del target[:]
    target.extend(values)
