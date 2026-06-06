"""AnkiOpsConnect actions.

This module stays importable without ``aqt`` so AnkiOpsConnect behavior can be
tested outside Anki.
"""

from __future__ import annotations

from collections.abc import Callable

ANKIOPS_CONNECT_VERSION = 1
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

AnkiOpsConnectAction = Callable[[object, dict], object]


class AnkiOpsConnectActionError(Exception):
    """Raised when an AnkiOpsConnect action cannot be completed safely."""


def dispatch_ankiops_connect_action(col, action: str, params: dict):
    try:
        handler = ANKIOPS_CONNECT_ACTIONS[action]
    except KeyError as error:
        raise AnkiOpsConnectActionError(
            f"Unknown AnkiOpsConnect action: {action}"
        ) from error
    return handler(col, params)


def version_action(_col, _params: dict) -> int:
    return ANKIOPS_CONNECT_VERSION


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
        return {
            deck.name: int(deck.id)
            for deck in decks.all_names_and_ids()
        }
    if hasattr(decks, "decks"):
        return {
            deck["name"]: int(deck_id)
            for deck_id, deck in decks.decks.items()
        }
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
                _note_field_values(note, model).items()
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
                "modelName": _note_model_name(note) if note is not None else "",
            }
        )
    return results


def model_names_action(col, _params: dict) -> list[str]:
    models = col.models
    if hasattr(models, "all_names"):
        return list(models.all_names())
    if hasattr(models, "all"):
        return [model.get("name", "") for model in models.all()]
    raise AnkiOpsConnectActionError("Cannot read Anki note type names.")


def model_field_names_action(col, params: dict) -> list[str]:
    return _field_names(_required_model(col, params.get("modelName") or ""))


def model_styling_action(col, params: dict) -> dict[str, str]:
    model_name = params.get("modelName") or ""
    model = _required_model(col, model_name)
    return {"name": model_name, "css": model.get("css", "")}


def model_templates_action(col, params: dict) -> dict[str, dict[str, str]]:
    model = _required_model(col, params.get("modelName") or "")
    return {
        template.get("name", ""): {
            "Front": template.get("qfmt", ""),
            "Back": template.get("afmt", ""),
        }
        for template in model.get("tmpls", [])
    }


def model_field_descriptions_action(col, params: dict) -> list[str]:
    model = _required_model(col, params.get("modelName") or "")
    return [field.get("description", "") for field in model.get("flds", [])]


def model_field_fonts_action(col, params: dict) -> dict[str, dict[str, int | str]]:
    model = _required_model(col, params.get("modelName") or "")
    return {
        field.get("name", ""): {
            "font": field.get("font", ""),
            "size": field.get("size", 0),
        }
        for field in model.get("flds", [])
    }


def create_model_action(col, params: dict) -> None:
    model_name = params.get("modelName") or ""
    if not model_name:
        raise AnkiOpsConnectActionError("modelName is required.")
    if _model_by_name(col.models, model_name) is not None:
        raise AnkiOpsConnectActionError(f"Note type already exists: {model_name}")

    model = _new_model(col.models, model_name)
    model["css"] = params.get("css") or ""
    model["type"] = 1 if params.get("isCloze") else 0
    for field_name in params.get("inOrderFields") or []:
        _add_field_to_model(col.models, model, str(field_name))
    for template in params.get("cardTemplates") or []:
        _add_template_to_model(col.models, model, template)
    _add_model(col.models, model)
    return None


def model_field_add_action(col, params: dict) -> None:
    model = _required_model(col, params.get("modelName") or "")
    _add_field_to_model(col.models, model, params.get("fieldName") or "")
    _save_model(col.models, model)
    return None


def model_field_remove_action(col, params: dict) -> None:
    model = _required_model(col, params.get("modelName") or "")
    field = _required_field(model, params.get("fieldName") or "")
    _remove_field_from_model(col.models, model, field)
    _save_model(col.models, model)
    return None


def model_field_reposition_action(col, params: dict) -> None:
    model = _required_model(col, params.get("modelName") or "")
    field = _required_field(model, params.get("fieldName") or "")
    _move_field_in_model(col.models, model, field, int(params.get("index") or 0))
    _save_model(col.models, model)
    return None


def model_field_set_description_action(col, params: dict) -> None:
    model = _required_model(col, params.get("modelName") or "")
    field = _required_field(model, params.get("fieldName") or "")
    field["description"] = params.get("description") or ""
    _save_model(col.models, model)
    return None


def model_field_set_font_size_action(col, params: dict) -> None:
    model = _required_model(col, params.get("modelName") or "")
    field = _required_field(model, params.get("fieldName") or "")
    field["size"] = int(params.get("fontSize") or 0)
    _save_model(col.models, model)
    return None


def update_model_styling_action(col, params: dict) -> None:
    payload = params.get("model") or {}
    model = _required_model(col, payload.get("name") or "")
    model["css"] = payload.get("css") or ""
    _save_model(col.models, model)
    return None


def model_template_rename_action(col, params: dict) -> None:
    model = _required_model(col, params.get("modelName") or "")
    template = _required_template(model, params.get("oldTemplateName") or "")
    template["name"] = params.get("newTemplateName") or ""
    _save_model(col.models, model)
    return None


def model_template_add_action(col, params: dict) -> None:
    model = _required_model(col, params.get("modelName") or "")
    _add_template_to_model(col.models, model, params.get("template") or {})
    _save_model(col.models, model)
    return None


def update_model_templates_action(col, params: dict) -> None:
    payload = params.get("model") or {}
    model = _required_model(col, payload.get("name") or "")
    templates = payload.get("templates") or {}
    for template_name, template_payload in templates.items():
        template = _required_template(model, template_name)
        template["qfmt"] = template_payload.get("Front", "")
        template["afmt"] = template_payload.get("Back", "")
    _save_model(col.models, model)
    return None


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
        _set_note_field(note, field_name, value)
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


def change_notes_notetype_action(col, params: dict) -> dict:
    return change_notes_notetype(
        col=col,
        note_ids=params.get("noteIds") or [],
        old_model=params.get("oldModel") or "",
        new_model=params.get("newModel") or "",
    )


def multi_action(col, params: dict) -> list:
    results = []
    for action in params.get("actions") or []:
        try:
            results.append(
                dispatch_ankiops_connect_action(
                    col,
                    action.get("action") or "",
                    action.get("params") or {},
                )
            )
        except Exception as error:
            results.append(str(error))
    return results


ANKIOPS_CONNECT_ACTIONS: dict[str, AnkiOpsConnectAction] = {
    "version": version_action,
    "getActiveProfile": get_active_profile_action,
    "deckNamesAndIds": deck_names_and_ids_action,
    "findNotes": find_notes_action,
    "notesInfo": notes_info_action,
    "cardsInfo": cards_info_action,
    "modelNames": model_names_action,
    "modelFieldNames": model_field_names_action,
    "modelStyling": model_styling_action,
    "modelTemplates": model_templates_action,
    "modelFieldDescriptions": model_field_descriptions_action,
    "modelFieldFonts": model_field_fonts_action,
    "createModel": create_model_action,
    "modelFieldAdd": model_field_add_action,
    "modelFieldRemove": model_field_remove_action,
    "modelFieldReposition": model_field_reposition_action,
    "modelFieldSetDescription": model_field_set_description_action,
    "modelFieldSetFontSize": model_field_set_font_size_action,
    "updateModelStyling": update_model_styling_action,
    "modelTemplateRename": model_template_rename_action,
    "modelTemplateAdd": model_template_add_action,
    "updateModelTemplates": update_model_templates_action,
    "createDeck": create_deck_action,
    "changeDeck": change_deck_action,
    "updateNote": update_note_action,
    "deleteNotes": delete_notes_action,
    "canAddNotesWithErrorDetail": can_add_notes_with_error_detail_action,
    "addNotes": add_notes_action,
    "getMediaDirPath": get_media_dir_path_action,
    "changeNotesNotetype": change_notes_notetype_action,
    "multi": multi_action,
}


def change_notes_notetype(col, note_ids, old_model: str, new_model: str) -> dict:
    note_ids = [int(note_id) for note_id in note_ids]
    if not note_ids:
        return {"changed": 0}
    if not old_model or not new_model:
        raise AnkiOpsConnectActionError("oldModel and newModel are required.")

    old_notetype = _model_by_name(col.models, old_model)
    new_notetype = _model_by_name(col.models, new_model)
    if old_notetype is None:
        raise AnkiOpsConnectActionError(f"Old note type not found: {old_model}")
    if new_notetype is None:
        raise AnkiOpsConnectActionError(f"New note type not found: {new_model}")

    old_fields = _field_names(old_notetype)
    new_fields = _field_names(new_notetype)
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
        _template_names(old_notetype),
        _template_names(new_notetype),
        "template",
    )

    notes = _load_notes(col, note_ids)
    for note_id, note in notes.items():
        actual_model = _note_model_name(note)
        if actual_model != old_model:
            raise AnkiOpsConnectActionError(
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
        raise AnkiOpsConnectActionError(
            "Post-conversion verification failed: " + "; ".join(failures)
        )

    return {"changed": len(note_ids)}


def _model_by_name(models, name: str):
    return models.by_name(name)


def _required_model(col, name: str):
    model = _model_by_name(col.models, name)
    if model is None:
        raise AnkiOpsConnectActionError(f"Note type not found: {name}")
    return model


def _model_id(model) -> int:
    try:
        return int(model["id"])
    except Exception as error:
        raise AnkiOpsConnectActionError("Note type is missing an id.") from error


def _field_names(model) -> list[str]:
    return [_item_name(field) for field in model.get("flds", [])]


def _template_names(model) -> list[str]:
    return [_item_name(template) for template in model.get("tmpls", [])]


def _item_name(item) -> str:
    return item.get("name") or ""


def _required_field(model, field_name: str):
    for field in model.get("flds", []):
        if _item_name(field) == field_name:
            return field
    raise AnkiOpsConnectActionError(f"Field not found: {field_name}")


def _required_template(model, template_name: str):
    for template in model.get("tmpls", []):
        if _item_name(template) == template_name:
            return template
    raise AnkiOpsConnectActionError(f"Template not found: {template_name}")


def _new_model(models, model_name: str):
    if hasattr(models, "new"):
        return models.new(model_name)
    return {"name": model_name, "flds": [], "tmpls": [], "css": ""}


def _new_field(models, field_name: str):
    if hasattr(models, "new_field"):
        return models.new_field(field_name)
    return {"name": field_name, "description": "", "font": "", "size": 0}


def _add_field_to_model(models, model, field_name: str) -> None:
    if not field_name:
        raise AnkiOpsConnectActionError("fieldName is required.")
    field = _new_field(models, field_name)
    if hasattr(models, "add_field"):
        models.add_field(model, field)
    elif hasattr(models, "addField"):
        models.addField(model, field)
    else:
        model.setdefault("flds", []).append(field)


def _remove_field_from_model(models, model, field) -> None:
    if hasattr(models, "remove_field"):
        models.remove_field(model, field)
    else:
        model["flds"] = [item for item in model.get("flds", []) if item is not field]


def _move_field_in_model(models, model, field, index: int) -> None:
    if hasattr(models, "reposition_field"):
        models.reposition_field(model, field, index)
    else:
        fields = model.get("flds", [])
        fields.remove(field)
        fields.insert(index, field)


def _template_payload_name(template: dict) -> str:
    return template.get("Name") or template.get("name") or ""


def _new_template(models, template_name: str):
    if hasattr(models, "new_template"):
        return models.new_template(template_name)
    return {"name": template_name, "qfmt": "", "afmt": ""}


def _add_template_to_model(models, model, template_payload: dict) -> None:
    template_name = _template_payload_name(template_payload)
    if not template_name:
        raise AnkiOpsConnectActionError("template name is required.")
    template = _new_template(models, template_name)
    template["qfmt"] = template_payload.get("Front", "")
    template["afmt"] = template_payload.get("Back", "")
    if hasattr(models, "add_template"):
        models.add_template(model, template)
    else:
        model.setdefault("tmpls", []).append(template)


def _add_model(models, model) -> None:
    if hasattr(models, "add"):
        models.add(model)
    else:
        _save_model(models, model)


def _save_model(models, model) -> None:
    if hasattr(models, "save"):
        models.save(model)


def _exact_name_map(old_names: list[str], new_names: list[str], kind: str) -> list[int]:
    if len(set(old_names)) != len(old_names) or len(set(new_names)) != len(new_names):
        raise AnkiOpsConnectActionError(f"Duplicate {kind} names are not supported.")
    if set(old_names) != set(new_names):
        raise AnkiOpsConnectActionError(
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
            raise AnkiOpsConnectActionError(f"Note not found: {note_id}") from error
        if note is None:
            raise AnkiOpsConnectActionError(f"Note not found: {note_id}")
        notes[note_id] = note
    return notes


def _note_model_name(note) -> str:
    if note is None:
        return ""
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
    _required_model(col, note_payload.get("modelName") or "")
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
    model = _required_model(col, note_payload.get("modelName") or "")
    deck_id = _deck_id(col, note_payload.get("deckName") or "", create=False)
    note = _new_note(col, model)
    for field_name, value in (note_payload.get("fields") or {}).items():
        _set_note_field(note, field_name, value)
    note.tags = list(note_payload.get("tags") or [])

    if hasattr(col, "add_note"):
        result = col.add_note(note, deck_id)
        if isinstance(result, int):
            return result
        note_id = getattr(note, "id", 0)
        if note_id:
            return int(note_id)
    raise AnkiOpsConnectActionError("Cannot add Anki note.")


def _set_note_field(note, field_name: str, value: str) -> None:
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
        field_names = _field_names(note.note_type() or {})
        try:
            raw_fields[field_names.index(field_name)] = value
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


def _replace_repeated(target, values: list[int]) -> None:
    del target[:]
    target.extend(values)
