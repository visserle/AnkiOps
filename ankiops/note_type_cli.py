"""Helper utilities for note-type related CLI behavior."""

from __future__ import annotations

import logging
import re
import shutil
import textwrap

import yaml

from ankiops.cli_anki import connect_or_exit
from ankiops.config import (
    get_note_types_dir,
    require_collection_dir,
    require_initialized_collection_dir,
)
from ankiops.fs import FileSystemAdapter
from ankiops.log import clickable_path
from ankiops.models import Field, NoteTypeConfig

logger = logging.getLogger(__name__)
_VALID_FIELD_LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _parse_identifying_answer(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"y", "yes"}:
        return True
    if normalized in {"n", "no"}:
        return False
    raise ValueError("Please enter 'y' or 'n'.")


def _validate_label_input(label: str, *, used_labels: set[str]) -> str:
    normalized = label.strip()
    if normalized.endswith(":"):
        normalized = normalized[:-1].strip()
    if not normalized:
        raise ValueError("Label cannot be empty.")
    if not _VALID_FIELD_LABEL_PATTERN.match(normalized):
        raise ValueError(
            "Label must start with a letter and contain only letters, "
            "numbers, '_' or '-'."
        )
    canonical = f"{normalized}:"
    if canonical in used_labels:
        raise ValueError(f"Label '{canonical}' is already used in this note type.")
    return canonical


def _prompt_field_label(field_name: str, *, used_labels: set[str]) -> str:
    while True:
        raw = input(f"Field '{field_name}' label (':' added automatically): ").strip()
        try:
            return _validate_label_input(raw, used_labels=used_labels)
        except ValueError as error:
            logger.error(str(error))


def _prompt_field_identifying(field_name: str) -> bool:
    while True:
        raw = input(f"Is field '{field_name}' identifying? [y/n]: ").strip()
        try:
            return _parse_identifying_answer(raw)
        except ValueError as error:
            logger.error(str(error))


def _is_choice_field(field_name: str) -> bool:
    return "choice" in field_name.lower()


def _render_bool_values(values: set[bool]) -> str:
    if values == {True}:
        return "yes"
    if values == {False}:
        return "no"
    return "mixed"


def _format_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    max_width: int | None = None,
) -> list[str]:
    if not headers:
        return []

    if max_width is None:
        max_width = shutil.get_terminal_size(fallback=(120, 24)).columns

    max_width = max(max_width, 40)
    separator_width = 2 * (len(headers) - 1)
    max_cell_lengths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            max_cell_lengths[index] = max(max_cell_lengths[index], len(cell))

    widths = [len(header) for header in headers]
    available = max(0, max_width - separator_width - sum(widths))
    expansion_needs = [
        max_cell_lengths[index] - widths[index] for index in range(len(headers))
    ]
    # Distribute extra width fairly across columns so one very long column
    # does not starve the others and force awkward wraps in short labels/fields.
    while available > 0:
        grew = False
        for index, need in enumerate(expansion_needs):
            if available == 0:
                break
            if need <= 0:
                continue
            widths[index] += 1
            expansion_needs[index] -= 1
            available -= 1
            grew = True
        if not grew:
            break

    def _wrap_cell(value: str, width: int) -> list[str]:
        wrapped = textwrap.wrap(
            value,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if wrapped:
            return wrapped
        if not value:
            return [""]
        return textwrap.wrap(
            value,
            width=width,
            break_long_words=True,
            break_on_hyphens=True,
        )

    def _render_row(values: list[str]) -> list[str]:
        wrapped_cells = [
            _wrap_cell(values[index], widths[index]) for index in range(len(values))
        ]
        row_height = max(len(lines) for lines in wrapped_cells)
        rendered: list[str] = []
        for line_index in range(row_height):
            rendered.append(
                "  ".join(
                    (
                        wrapped_cells[column_index][line_index]
                        if line_index < len(wrapped_cells[column_index])
                        else ""
                    ).ljust(widths[column_index])
                    for column_index in range(len(values))
                )
            )
        return rendered

    separator = ["-" * width for width in widths]
    rendered_lines = [*(_render_row(headers)), *(_render_row(separator))]
    for row in rows:
        rendered_lines.extend(_render_row(row))
    return rendered_lines


def _build_global_label_constraints(
    note_type_configs: list[NoteTypeConfig],
) -> tuple[dict[str, str], dict[str, bool]]:
    label_to_field_name: dict[str, str] = {}
    label_to_identifying: dict[str, bool] = {}
    for config in note_type_configs:
        for field_config in config.fields:
            if field_config.label is None:
                continue
            label_to_field_name[field_config.label] = field_config.name
            label_to_identifying[field_config.label] = field_config.identifying
    return label_to_field_name, label_to_identifying


def _validate_global_label_reuse(
    *,
    label: str,
    field_name: str,
    identifying: bool,
    label_to_field_name: dict[str, str],
    label_to_identifying: dict[str, bool],
) -> None:
    existing_field_name = label_to_field_name.get(label)
    if existing_field_name is not None and existing_field_name != field_name:
        raise ValueError(
            f"Label '{label}' is already mapped to field '{existing_field_name}'."
        )

    existing_identifying = label_to_identifying.get(label)
    if existing_identifying is not None and existing_identifying != identifying:
        raise ValueError(
            f"Label '{label}' already has IDENTIFYING={existing_identifying} "
            f"in existing note types."
        )


def _log_note_type_label_info(note_type_configs: list[NoteTypeConfig]) -> None:
    sorted_configs = sorted(note_type_configs, key=lambda config_item: config_item.name)

    note_type_rows: list[list[str]] = []
    for config in sorted_configs:
        labels: list[str] = []
        identifying_labels: list[str] = []
        identifying_base_labels: list[str] = []
        identifying_choice_labels: list[str] = []

        for field_config in config.fields:
            if field_config.label is None:
                continue
            labels.append(field_config.label)
            if not field_config.identifying:
                continue

            identifying_labels.append(field_config.label)
            if config.is_choice and _is_choice_field(field_config.name):
                identifying_choice_labels.append(field_config.label)
            else:
                identifying_base_labels.append(field_config.label)

        labels_rendered = ", ".join(labels) or "(none)"
        if config.is_choice:
            base_rendered = ", ".join(identifying_base_labels) or "(none)"
            choice_rendered = ", ".join(identifying_choice_labels) or "(none)"
            identifying_rendered = (
                f"{base_rendered} + any one choice label ({choice_rendered})"
            )
        else:
            identifying_rendered = ", ".join(identifying_labels) or "(none)"

        note_type_rows.append([config.name, labels_rendered, identifying_rendered])

    logger.info("Note Types")
    logger.info("==========")
    logger.info("----------")
    for line in _format_table(
        headers=["Note type", "Labels", "Identifying"],
        rows=note_type_rows,
    ):
        logger.info(line)

    logger.info("")

    label_index: dict[str, list[tuple[str, str, bool]]] = {}
    for config in sorted(note_type_configs, key=lambda config_item: config_item.name):
        for field_config in config.fields:
            if field_config.label is None:
                continue
            entries = label_index.setdefault(field_config.label, [])
            entries.append(
                (
                    config.name,
                    field_config.name,
                    field_config.identifying,
                )
            )

    registry_rows: list[list[str]] = []
    for label in sorted(label_index):
        entries = sorted(label_index[label], key=lambda entry: (entry[0], entry[1]))
        field_names = sorted({field_name for _, field_name, _ in entries})
        note_type_names = sorted({note_type_name for note_type_name, _, _ in entries})
        identifying_values = {identifying for _, _, identifying in entries}

        registry_rows.append(
            [
                label,
                ", ".join(field_names),
                _render_bool_values(identifying_values),
                ", ".join(note_type_names),
            ]
        )

    logger.info("Label Registry")
    logger.info("==============")
    logger.info("--------------")
    for line in _format_table(
        headers=[
            "Label",
            "Field",
            "Identifying",
            "Used by",
        ],
        rows=registry_rows,
    ):
        logger.info(line)


def _build_template_output_files(
    templates: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    yaml_templates: list[dict[str, str]] = []
    file_contents: dict[str, str] = {}

    template_items = list(templates.items())
    for template_index, (template_name, template_data) in enumerate(
        template_items, start=1
    ):
        suffix = "" if template_index == 1 else str(template_index)
        front_filename = f"Front{suffix}.template.anki"
        back_filename = f"Back{suffix}.template.anki"
        yaml_templates.append(
            {
                "name": template_name,
                "front": front_filename,
                "back": back_filename,
            }
        )
        file_contents[front_filename] = str(template_data.get("Front", ""))
        file_contents[back_filename] = str(template_data.get("Back", ""))

    return yaml_templates, file_contents


def _normalize_styling_payload(styling: object, *, note_type_name: str) -> str:
    if isinstance(styling, str):
        return styling
    if isinstance(styling, dict):
        css = styling.get("css")
        if isinstance(css, str):
            return css
    raise ValueError(
        f"Invalid styling payload for note type '{note_type_name}'. "
        "Expected a string or object with string 'css'."
    )


def run(args) -> None:
    """Handle note-types CLI actions."""
    action_value = getattr(args, "action", "list")
    action = "list" if action_value in {None, ""} else str(action_value)
    if action not in {"list", "import"}:
        logger.error(f"Unknown note-types action: {action}")
        raise SystemExit(2)

    note_types_dir = get_note_types_dir()
    fs = FileSystemAdapter()

    if action == "list":
        require_initialized_collection_dir()
        try:
            note_type_configs = fs.load_note_type_configs(note_types_dir)
        except ValueError as error:
            logger.error(str(error))
            raise SystemExit(1) from error
        _log_note_type_label_info(note_type_configs)
        return

    note_type_name = str(getattr(args, "name", "")).strip()
    if not note_type_name:
        logger.error("A note type name is required for 'note-types import'.")
        raise SystemExit(2)

    anki = connect_or_exit()
    active_profile = anki.get_active_profile()
    collection_dir = require_collection_dir(active_profile)
    logger.debug(f"Collection directory: {collection_dir}")

    destination_dir = note_types_dir / note_type_name
    if destination_dir.exists():
        logger.error(
            f"Target note type directory already exists: {destination_dir}. "
            "Use a different name or remove the existing folder first."
        )
        raise SystemExit(1)

    model_names = set(anki.fetch_model_names())
    if note_type_name not in model_names:
        available = ", ".join(sorted(model_names)) or "(none)"
        logger.error(
            f"Note type '{note_type_name}' not found in Anki. Available: {available}"
        )
        raise SystemExit(1)

    state = anki.fetch_model_states([note_type_name]).get(note_type_name)
    if state is None:
        logger.error(f"Could not fetch note type state for '{note_type_name}'.")
        raise SystemExit(1)
    try:
        styling_css = _normalize_styling_payload(
            state.get("styling"),
            note_type_name=note_type_name,
        )
    except ValueError as error:
        logger.error(str(error))
        raise SystemExit(1) from error

    try:
        existing_configs = fs.load_note_type_configs(note_types_dir)
    except ValueError as error:
        logger.error(str(error))
        raise SystemExit(1) from error

    label_to_field_name, label_to_identifying = _build_global_label_constraints(
        existing_configs
    )

    fields = []
    used_labels: set[str] = set()
    logger.info(f"Copying note type '{note_type_name}' from Anki.")
    for field_name in state["fields"]:
        while True:
            label = _prompt_field_label(field_name, used_labels=used_labels)
            identifying = _prompt_field_identifying(field_name)
            try:
                _validate_global_label_reuse(
                    label=label,
                    field_name=field_name,
                    identifying=identifying,
                    label_to_field_name=label_to_field_name,
                    label_to_identifying=label_to_identifying,
                )
            except ValueError as error:
                logger.error(str(error))
                continue
            used_labels.add(label)
            label_to_field_name[label] = field_name
            label_to_identifying[label] = identifying
            fields.append(Field(name=field_name, label=label, identifying=identifying))
            break

    templates = state["templates"]
    is_choice = any(_is_choice_field(field_name) for field_name in state["fields"])
    is_cloze = any(
        "{{cloze:" in str(template_data.get(template_side, "")).lower()
        for template_data in templates.values()
        for template_side in ("Front", "Back")
    )
    candidate_config = NoteTypeConfig(
        name=note_type_name,
        fields=fields,
        css=styling_css,
        is_choice=is_choice,
        is_cloze=is_cloze,
        templates=[
            {
                "Name": template_name,
                "Front": str(template_data.get("Front", "")),
                "Back": str(template_data.get("Back", "")),
            }
            for template_name, template_data in templates.items()
        ],
    )

    try:
        NoteTypeConfig.validate_configs([*existing_configs, candidate_config])
    except ValueError as error:
        logger.error(f"Invalid note type configuration: {error}")
        raise SystemExit(1) from error

    note_types_dir.mkdir(parents=True, exist_ok=True)
    destination_dir.mkdir(parents=True, exist_ok=False)

    yaml_templates, template_files = _build_template_output_files(templates)
    yaml_data = {
        "styling": "Styling.css",
        "templates": yaml_templates,
        "fields": [
            {
                "name": field_config.name,
                "label": field_config.label,
                "identifying": field_config.identifying,
            }
            for field_config in fields
        ],
    }
    if is_choice:
        yaml_data["is_choice"] = True
    if is_cloze:
        yaml_data["is_cloze"] = True

    for filename, content in template_files.items():
        (destination_dir / filename).write_text(content, encoding="utf-8")

    styling_path = destination_dir / "Styling.css"
    styling_path.write_text(styling_css, encoding="utf-8")

    config_path = destination_dir / "note_type.yaml"
    config_path.write_text(
        yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    identifying_count = sum(1 for field_config in fields if field_config.identifying)
    logger.info(
        f"Copied note type '{note_type_name}' with {len(fields)} fields "
        f"({identifying_count} identifying)."
    )
    logger.info(f"Saved to: {clickable_path(destination_dir)}")
    logger.info(f"Config: {clickable_path(config_path)}")
