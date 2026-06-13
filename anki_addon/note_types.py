"""Anki note type actions for AnkiOpsConnect."""

from __future__ import annotations

from .actions import AnkiOpsConnectActionError


def note_type_names_action(col, _params: dict) -> list[str]:
    models = col.models
    if hasattr(models, "all_names"):
        return list(models.all_names())
    if hasattr(models, "all"):
        return [model.get("name", "") for model in models.all()]
    raise AnkiOpsConnectActionError("Cannot read Anki note type names.")


def note_type_field_names_action(col, params: dict) -> list[str]:
    return field_names(required_note_type(col, params.get("modelName") or ""))


def note_type_styling_action(col, params: dict) -> dict[str, str]:
    model_name = params.get("modelName") or ""
    model = required_note_type(col, model_name)
    return {"name": model_name, "css": model.get("css", "")}


def note_type_templates_action(col, params: dict) -> dict[str, dict[str, str]]:
    model = required_note_type(col, params.get("modelName") or "")
    return {
        template.get("name", ""): {
            "Front": template.get("qfmt", ""),
            "Back": template.get("afmt", ""),
        }
        for template in model.get("tmpls", [])
    }


def note_type_field_descriptions_action(col, params: dict) -> list[str]:
    model = required_note_type(col, params.get("modelName") or "")
    return [field.get("description", "") for field in model.get("flds", [])]


def note_type_field_fonts_action(col, params: dict) -> dict[str, dict[str, int | str]]:
    model = required_note_type(col, params.get("modelName") or "")
    return {
        field.get("name", ""): {
            "font": field.get("font", ""),
            "size": field.get("size", 0),
        }
        for field in model.get("flds", [])
    }


def create_note_type_action(col, params: dict) -> None:
    model_name = params.get("modelName") or ""
    if not model_name:
        raise AnkiOpsConnectActionError("modelName is required.")
    if note_type_by_name(col.models, model_name) is not None:
        raise AnkiOpsConnectActionError(f"Note type already exists: {model_name}")

    model = _new_note_type(col.models, model_name)
    model["css"] = params.get("css") or ""
    model["type"] = 1 if params.get("isCloze") else 0
    for field_name in params.get("inOrderFields") or []:
        _add_field_to_note_type(col.models, model, str(field_name))
    for template in params.get("cardTemplates") or []:
        _add_template_to_note_type(col.models, model, template)
    _add_note_type(col.models, model)
    return None


def note_type_field_add_action(col, params: dict) -> None:
    model = required_note_type(col, params.get("modelName") or "")
    _add_field_to_note_type(col.models, model, params.get("fieldName") or "")
    save_note_type(col.models, model)
    return None


def note_type_field_remove_action(col, params: dict) -> None:
    model = required_note_type(col, params.get("modelName") or "")
    field = _required_field(model, params.get("fieldName") or "")
    _remove_field_from_note_type(col.models, model, field)
    save_note_type(col.models, model)
    return None


def note_type_field_reposition_action(col, params: dict) -> None:
    model = required_note_type(col, params.get("modelName") or "")
    field = _required_field(model, params.get("fieldName") or "")
    _move_field_in_note_type(col.models, model, field, int(params.get("index") or 0))
    save_note_type(col.models, model)
    return None


def note_type_field_set_description_action(col, params: dict) -> None:
    model = required_note_type(col, params.get("modelName") or "")
    field = _required_field(model, params.get("fieldName") or "")
    field["description"] = params.get("description") or ""
    save_note_type(col.models, model)
    return None


def note_type_field_set_font_size_action(col, params: dict) -> None:
    model = required_note_type(col, params.get("modelName") or "")
    field = _required_field(model, params.get("fieldName") or "")
    field["size"] = int(params.get("fontSize") or 0)
    save_note_type(col.models, model)
    return None


def update_note_type_styling_action(col, params: dict) -> None:
    payload = params.get("model") or {}
    model = required_note_type(col, payload.get("name") or "")
    model["css"] = payload.get("css") or ""
    save_note_type(col.models, model)
    return None


def note_type_template_rename_action(col, params: dict) -> None:
    model = required_note_type(col, params.get("modelName") or "")
    template = _required_template(model, params.get("oldTemplateName") or "")
    template["name"] = params.get("newTemplateName") or ""
    save_note_type(col.models, model)
    return None


def note_type_template_add_action(col, params: dict) -> None:
    model = required_note_type(col, params.get("modelName") or "")
    _add_template_to_note_type(col.models, model, params.get("template") or {})
    save_note_type(col.models, model)
    return None


def update_note_type_templates_action(col, params: dict) -> None:
    payload = params.get("model") or {}
    model = required_note_type(col, payload.get("name") or "")
    templates = payload.get("templates") or {}
    for template_name, template_payload in templates.items():
        template = _required_template(model, template_name)
        template["qfmt"] = template_payload.get("Front", "")
        template["afmt"] = template_payload.get("Back", "")
    save_note_type(col.models, model)
    return None


def note_type_by_name(models, name: str):
    return models.by_name(name)


def required_note_type(col, name: str):
    model = note_type_by_name(col.models, name)
    if model is None:
        raise AnkiOpsConnectActionError(f"Note type not found: {name}")
    return model


def note_type_id(model) -> int:
    try:
        return int(model["id"])
    except Exception as error:
        raise AnkiOpsConnectActionError("Note type is missing an id.") from error


def field_names(model) -> list[str]:
    return [_item_name(field) for field in model.get("flds", [])]


def template_names(model) -> list[str]:
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


def _new_note_type(models, model_name: str):
    if hasattr(models, "new"):
        return models.new(model_name)
    return {"name": model_name, "flds": [], "tmpls": [], "css": ""}


def _new_field(models, field_name: str):
    if hasattr(models, "new_field"):
        return models.new_field(field_name)
    return {"name": field_name, "description": "", "font": "", "size": 0}


def _add_field_to_note_type(models, model, field_name: str) -> None:
    if not field_name:
        raise AnkiOpsConnectActionError("fieldName is required.")
    field = _new_field(models, field_name)
    if hasattr(models, "add_field"):
        models.add_field(model, field)
    elif hasattr(models, "addField"):
        models.addField(model, field)
    else:
        model.setdefault("flds", []).append(field)


def _remove_field_from_note_type(models, model, field) -> None:
    if hasattr(models, "remove_field"):
        models.remove_field(model, field)
    else:
        model["flds"] = [item for item in model.get("flds", []) if item is not field]


def _move_field_in_note_type(models, model, field, index: int) -> None:
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


def _add_template_to_note_type(models, model, template_payload: dict) -> None:
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


def _add_note_type(models, model) -> None:
    if hasattr(models, "add"):
        models.add(model)
    else:
        save_note_type(models, model)


def save_note_type(models, model) -> None:
    if hasattr(models, "save"):
        models.save(model)
