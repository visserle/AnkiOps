"""Anki collection actions for AnkiOpsConnect."""

from __future__ import annotations

from .actions import AnkiOpsConnectActionError
from .note_types import field_names, required_note_type

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


def get_active_profile_action(col, _params: dict) -> str:
    if hasattr(col, "ankiops_connect_active_profile"):
        return str(col.ankiops_connect_active_profile)
    try:
        from aqt import mw
    except Exception:
        return ""
    profile_manager = getattr(mw, "pm", None)
    name = getattr(profile_manager, "name", "")
    if callable(name):
        return str(name())
    return str(name)


def deck_names_and_ids_action(col, _params: dict) -> dict[str, int]:
    decks = col.decks
    if hasattr(decks, "all_names_and_ids"):
        return {deck.name: int(deck.id) for deck in decks.all_names_and_ids()}
    if hasattr(decks, "decks"):
        return {deck["name"]: int(deck_id) for deck_id, deck in decks.decks.items()}
    raise AnkiOpsConnectActionError("Cannot read Anki deck names.")


def find_notes_action(col, params: dict) -> list[int]:
    return [int(note_id) for note_id in col.find_notes(params.get("query") or "")]


def notes_info_action(col, params: dict) -> list[dict]:
    results = []
    for note_id in params.get("notes") or []:
        note = _get_note_or_none(col, int(note_id))
        if note is None:
            continue
        model = note.note_type() or {}
        fields = {
            name: {"value": value, "order": index}
            for index, (name, value) in enumerate(
                note_field_values(note, model).items()
            )
        }
        results.append(
            {
                "noteId": int(note_id),
                "modelName": model.get("name", ""),
                "fields": fields,
                "cards": _note_card_ids(col, int(note_id)),
                "tags": list(getattr(note, "tags", [])),
            }
        )
    return results


def cards_info_action(col, params: dict) -> list[dict]:
    results = []
    for card_id in params.get("cards") or []:
        card = _get_card_or_none(col, int(card_id))
        if card is None:
            continue
        note_id = _card_note_id(card)
        note = _get_note_or_none(col, note_id)
        results.append(
            {
                "cardId": int(card_id),
                "note": note_id,
                "deckName": _deck_name(col, _card_deck_id(card)),
                "modelName": note_type_name(note) if note is not None else "",
            }
        )
    return results


def create_deck_action(col, params: dict) -> int:
    return _deck_id(col, params.get("deck") or "", create=True)


def change_deck_action(col, params: dict) -> None:
    card_ids = [int(card_id) for card_id in params.get("cards") or []]
    deck_id = _deck_id(col, params.get("deck") or "", create=True)
    sched = getattr(col, "sched", None)
    if hasattr(sched, "set_deck"):
        sched.set_deck(card_ids, deck_id)
        return None
    for card_id in card_ids:
        card = _get_card_or_none(col, card_id)
        if card is None:
            continue
        setattr(card, "did", deck_id)
        _save_card(col, card)
    return None


def update_note_action(col, params: dict) -> None:
    payload = params.get("note") or {}
    note = _get_note_or_none(col, int(payload.get("id") or 0))
    if note is None:
        raise AnkiOpsConnectActionError(f"Note not found: {payload.get('id')}")
    for field_name, value in (payload.get("fields") or {}).items():
        set_note_field(note, field_name, value)
    if "tags" in payload:
        note.tags = list(payload["tags"])
    _save_note(col, note)
    return None


def delete_notes_action(col, params: dict) -> None:
    note_ids = [int(note_id) for note_id in params.get("notes") or []]
    if hasattr(col, "remove_notes"):
        col.remove_notes(note_ids)
        return None
    raise AnkiOpsConnectActionError("Cannot delete Anki notes.")


def can_add_notes_with_error_detail_action(col, params: dict) -> list[dict]:
    results = []
    for note_payload in params.get("notes") or []:
        try:
            _validate_note_payload_for_add(col, note_payload)
        except Exception as error:
            results.append({"canAdd": False, "error": str(error)})
        else:
            results.append({"canAdd": True})
    return results


def add_notes_action(col, params: dict) -> list[int | str]:
    results = []
    for note_payload in params.get("notes") or []:
        try:
            results.append(_add_note(col, note_payload))
        except Exception as error:
            results.append(str(error))
    return results


def get_media_dir_path_action(col, _params: dict) -> str:
    media = getattr(col, "media", None)
    directory = getattr(media, "dir", None)
    if callable(directory):
        return str(directory())
    if directory:
        return str(directory)
    raise AnkiOpsConnectActionError("Cannot read Anki media directory.")


def load_notes(col, note_ids: list[int]) -> dict[int, object]:
    notes = {}
    for note_id in note_ids:
        try:
            note = col.get_note(note_id)
        except Exception as error:
            raise AnkiOpsConnectActionError(f"Note not found: {note_id}") from error
        if note is None:
            raise AnkiOpsConnectActionError(f"Note not found: {note_id}")
        notes[note_id] = note
    return notes


def note_type_name(note) -> str:
    if note is None:
        return ""
    notetype = note.note_type()
    if notetype:
        return notetype.get("name", "")
    return ""


def note_field_values(note, model) -> dict[str, str]:
    names = field_names(model)
    raw_fields = getattr(note, "fields", None)
    if isinstance(raw_fields, dict):
        return {name: raw_fields.get(name, "") for name in names}
    if raw_fields is not None:
        return {
            name: raw_fields[index] if index < len(raw_fields) else ""
            for index, name in enumerate(names)
        }
    return {}


def card_snapshot(col, note_ids: list[int]) -> dict[int, tuple]:
    if hasattr(col, "ankiops_connect_cards_snapshot"):
        return col.ankiops_connect_cards_snapshot(note_ids)

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


def _get_note_or_none(col, note_id: int):
    try:
        return col.get_note(note_id)
    except Exception:
        return None


def _get_card_or_none(col, card_id: int):
    try:
        return col.get_card(card_id)
    except Exception:
        return None


def _note_card_ids(col, note_id: int) -> list[int]:
    try:
        return [int(card_id) for card_id in col.find_cards(f"nid:{note_id}")]
    except Exception:
        return []


def _card_note_id(card) -> int:
    return int(getattr(card, "nid", getattr(card, "note", 0)))


def _card_deck_id(card) -> int:
    return int(getattr(card, "did", getattr(card, "deck_id", 0)))


def _deck_name(col, deck_id: int) -> str:
    decks = col.decks
    if hasattr(decks, "name"):
        return decks.name(deck_id) or ""
    if hasattr(decks, "name_if_exists"):
        return decks.name_if_exists(deck_id) or ""
    return ""


def _existing_deck_id(col, deck_name: str) -> int | None:
    decks = col.decks
    if hasattr(decks, "all_names_and_ids"):
        for deck in decks.all_names_and_ids():
            if deck.name == deck_name:
                return int(deck.id)
    if hasattr(decks, "by_name"):
        deck = decks.by_name(deck_name)
        if deck:
            return int(deck["id"])
    return None


def _deck_id(col, deck_name: str, *, create: bool) -> int:
    if not deck_name:
        raise AnkiOpsConnectActionError("deck is required.")
    deck_id = _existing_deck_id(col, deck_name)
    if deck_id is not None:
        return deck_id
    if not create:
        raise AnkiOpsConnectActionError(f"deck was not found: {deck_name}")
    decks = col.decks
    if hasattr(decks, "id"):
        return int(decks.id(deck_name))
    raise AnkiOpsConnectActionError(f"Cannot create deck: {deck_name}")


def _save_card(col, card) -> None:
    if hasattr(col, "update_card"):
        col.update_card(card)
    elif hasattr(card, "flush"):
        card.flush()


def _validate_note_payload_for_add(col, note_payload: dict) -> None:
    required_note_type(col, note_payload.get("modelName") or "")
    _deck_id(col, note_payload.get("deckName") or "", create=False)


def _new_note(col, model):
    if hasattr(col, "new_note"):
        return col.new_note(model)
    try:
        from anki.notes import Note
    except Exception as error:
        raise AnkiOpsConnectActionError("Cannot create Anki note.") from error
    return Note(col, model)


def _add_note(col, note_payload: dict) -> int:
    _validate_note_payload_for_add(col, note_payload)
    model = required_note_type(col, note_payload.get("modelName") or "")
    deck_id = _deck_id(col, note_payload.get("deckName") or "", create=False)
    note = _new_note(col, model)
    for field_name, value in (note_payload.get("fields") or {}).items():
        set_note_field(note, field_name, value)
    note.tags = list(note_payload.get("tags") or [])

    if hasattr(col, "add_note"):
        result = col.add_note(note, deck_id)
        if isinstance(result, int):
            return result
        note_id = getattr(note, "id", 0)
        if note_id:
            return int(note_id)
    raise AnkiOpsConnectActionError("Cannot add Anki note.")


def set_note_field(note, field_name: str, value: str) -> None:
    try:
        note[field_name] = value
        return
    except Exception:
        pass

    raw_fields = getattr(note, "fields", None)
    if isinstance(raw_fields, dict):
        raw_fields[field_name] = value
        return
    if raw_fields is not None:
        names = field_names(note.note_type() or {})
        try:
            raw_fields[names.index(field_name)] = value
            return
        except ValueError as error:
            raise AnkiOpsConnectActionError(f"Field not found: {field_name}") from error
    raise AnkiOpsConnectActionError(f"Cannot update field: {field_name}")


def _save_note(col, note) -> None:
    if hasattr(col, "update_note"):
        col.update_note(note)
    elif hasattr(col, "updateNote"):
        col.updateNote(note)
    elif hasattr(note, "flush"):
        note.flush()
