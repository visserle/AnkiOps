"""AnkiOpsConnect action registry."""

from __future__ import annotations

from collections.abc import Callable

ANKIOPS_CONNECT_VERSION = 1
ANKIOPS_KEY_FIELD_NAME = "AnkiOps Key"

AnkiOpsConnectAction = Callable[[object, dict], object]


def _action_registry() -> dict[str, AnkiOpsConnectAction]:
    from .collection import (
        add_notes_action,
        can_add_notes_with_error_detail_action,
        cards_info_action,
        change_deck_action,
        create_deck_action,
        deck_names_and_ids_action,
        delete_notes_action,
        find_notes_action,
        get_active_profile_action,
        get_media_dir_path_action,
        notes_info_action,
        update_note_action,
    )
    from .note_type_conversion import convert_notes_to_note_type_action
    from .note_types import (
        create_note_type_action,
        note_type_field_add_action,
        note_type_field_descriptions_action,
        note_type_field_fonts_action,
        note_type_field_names_action,
        note_type_field_remove_action,
        note_type_field_reposition_action,
        note_type_field_set_description_action,
        note_type_field_set_font_size_action,
        note_type_names_action,
        note_type_styling_action,
        note_type_template_add_action,
        note_type_template_rename_action,
        note_type_templates_action,
        update_note_type_styling_action,
        update_note_type_templates_action,
    )

    return {
        "version": version_action,
        "getActiveProfile": get_active_profile_action,
        "deckNamesAndIds": deck_names_and_ids_action,
        "findNotes": find_notes_action,
        "notesInfo": notes_info_action,
        "cardsInfo": cards_info_action,
        "modelNames": note_type_names_action,
        "modelFieldNames": note_type_field_names_action,
        "modelStyling": note_type_styling_action,
        "modelTemplates": note_type_templates_action,
        "modelFieldDescriptions": note_type_field_descriptions_action,
        "modelFieldFonts": note_type_field_fonts_action,
        "createModel": create_note_type_action,
        "modelFieldAdd": note_type_field_add_action,
        "modelFieldRemove": note_type_field_remove_action,
        "modelFieldReposition": note_type_field_reposition_action,
        "modelFieldSetDescription": note_type_field_set_description_action,
        "modelFieldSetFontSize": note_type_field_set_font_size_action,
        "updateModelStyling": update_note_type_styling_action,
        "modelTemplateRename": note_type_template_rename_action,
        "modelTemplateAdd": note_type_template_add_action,
        "updateModelTemplates": update_note_type_templates_action,
        "createDeck": create_deck_action,
        "changeDeck": change_deck_action,
        "updateNote": update_note_action,
        "deleteNotes": delete_notes_action,
        "canAddNotesWithErrorDetail": can_add_notes_with_error_detail_action,
        "addNotes": add_notes_action,
        "getMediaDirPath": get_media_dir_path_action,
        "convertNotesToNoteType": convert_notes_to_note_type_action,
        "multi": multi_action,
    }


class AnkiOpsConnectActionError(Exception):
    """Raised when an AnkiOpsConnect action cannot be completed safely."""


def dispatch_action(col, action: str, params: dict):
    try:
        handler = _action_registry()[action]
    except KeyError as error:
        raise AnkiOpsConnectActionError(
            f"Unknown AnkiOpsConnect action: {action}"
        ) from error
    return handler(col, params)


def version_action(_col, _params: dict) -> int:
    return ANKIOPS_CONNECT_VERSION


def multi_action(col, params: dict) -> list:
    results = []
    for action in params.get("actions") or []:
        try:
            results.append(
                dispatch_action(
                    col,
                    action.get("action") or "",
                    action.get("params") or {},
                )
            )
        except Exception as error:
            results.append(str(error))
    return results
