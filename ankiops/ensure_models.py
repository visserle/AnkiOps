"""Ensure AnkiOpsQA, AnkiOpsCloze, and AnkiOpsChoice note types exist in Anki and are up
to date with our templates."""

import logging
from importlib import resources

from ankiops.anki_client import invoke
from ankiops.config import NOTE_TYPES

logger = logging.getLogger(__name__)


def _load_template(filename: str) -> str:
    """Read a template file from the models directory."""
    return (
        resources.files("ankiops.models").joinpath(filename).read_text(encoding="utf-8")
    )


def _create_model(model_name: str, is_cloze: bool) -> None:
    """Create a note type in Anki from the template files."""
    cfg = NOTE_TYPES[model_name]
    fields = [field_name for field_name, _, _ in cfg["field_mappings"]]

    css = _load_template("Styling.css")
    front = _load_template(f"{model_name}Front.template.anki")
    back = _load_template(f"{model_name}Back.template.anki")

    invoke(
        "createModel",
        modelName=model_name,
        inOrderFields=fields,
        css=css,
        isCloze=is_cloze,
        cardTemplates=[{"Name": "Card", "Front": front, "Back": back}],
    )
    logger.info(f"Created note type '{model_name}' in Anki")


def _is_model_up_to_date(model_name: str) -> bool:
    """Check if a model's templates and styling match our template files."""
    css = _load_template("Styling.css")
    front = _load_template(f"{model_name}Front.template.anki")
    back = _load_template(f"{model_name}Back.template.anki")

    # Get current model info
    current_styling = invoke("modelStyling", modelName=model_name)
    current_templates = invoke("modelTemplates", modelName=model_name)

    # Compare styling
    if current_styling.get("css", "").strip() != css.strip():
        return False

    # Compare templates - AnkiConnect returns templates as a dict with card names as keys
    card_template = next(iter(current_templates.values()), {})
    current_front = card_template.get("Front", "").strip()
    current_back = card_template.get("Back", "").strip()

    if current_front != front.strip() or current_back != back.strip():
        return False

    return True


def _update_model(model_name: str) -> None:
    """Update an existing note type's templates and styling."""
    css = _load_template("Styling.css")
    front = _load_template(f"{model_name}Front.template.anki")
    back = _load_template(f"{model_name}Back.template.anki")

    # Update styling
    invoke("updateModelStyling", model={"name": model_name, "css": css})

    # Update templates - use the actual template name from Anki
    current_templates = invoke("modelTemplates", modelName=model_name)
    template_name = next(iter(current_templates))
    invoke(
        "updateModelTemplates",
        model={
            "name": model_name,
            "templates": {template_name: {"Front": front, "Back": back}},
        },
    )
    logger.info(f"Updated note type '{model_name}' in Anki")


def ensure_models() -> None:
    """Ensure all required note types exist in Anki and are up to date."""
    existing = set(invoke("modelNames"))

    if "AnkiOpsQA" not in existing:
        _create_model("AnkiOpsQA", is_cloze=False)
    elif not _is_model_up_to_date("AnkiOpsQA"):
        _update_model("AnkiOpsQA")

    if "AnkiOpsCloze" not in existing:
        _create_model("AnkiOpsCloze", is_cloze=True)
    elif not _is_model_up_to_date("AnkiOpsCloze"):
        _update_model("AnkiOpsCloze")

    if "AnkiOpsChoice" not in existing:
        _create_model("AnkiOpsChoice", is_cloze=False)
    elif not _is_model_up_to_date("AnkiOpsChoice"):
        _update_model("AnkiOpsChoice")
