"""Helper utilities for note-type related CLI behavior."""

from __future__ import annotations

import logging
import re

from ankiops.models import NoteTypeConfig

logger = logging.getLogger(__name__)
_VALID_FIELD_PREFIX_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9]*:$")


def _parse_identifying_answer(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"y", "yes"}:
        return True
    if normalized in {"n", "no"}:
        return False
    raise ValueError("Please enter 'y' or 'n'.")


def _validate_prefix_input(prefix: str, *, used_prefixes: set[str]) -> str:
    normalized = prefix.strip()
    if not normalized:
        raise ValueError("Prefix cannot be empty.")
    if not _VALID_FIELD_PREFIX_PATTERN.match(normalized):
        raise ValueError(
            "Prefix must start with a letter, contain only letters/numbers, "
            "and end with ':'."
        )
    if normalized in used_prefixes:
        raise ValueError(f"Prefix '{normalized}' is already used in this note type.")
    return normalized


def _prompt_field_prefix(field_name: str, *, used_prefixes: set[str]) -> str:
    while True:
        raw = input(f"Field '{field_name}' prefix: ").strip()
        try:
            return _validate_prefix_input(raw, used_prefixes=used_prefixes)
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


def _format_owner_entry(
    note_type_name: str,
    field_name: str,
    *,
    show_identifying: bool,
) -> str:
    suffix = " [IDENTIFYING]" if show_identifying else ""
    return f"{note_type_name}.{field_name}{suffix}"


def _log_note_type_prefix_info(note_type_configs: list[NoteTypeConfig]) -> None:
    prefix_index: dict[str, list[tuple[str, str, bool]]] = {}
    for config in sorted(note_type_configs, key=lambda config_item: config_item.name):
        for field_config in config.fields:
            if field_config.prefix is None:
                continue
            entries = prefix_index.setdefault(field_config.prefix, [])
            entries.append(
                (
                    config.name,
                    field_config.name,
                    field_config.identifying,
                )
            )

    logger.info("Taken prefixes:")
    for prefix in sorted(prefix_index):
        entries = sorted(prefix_index[prefix], key=lambda entry: (entry[0], entry[1]))
        owners = ", ".join(
            _format_owner_entry(
                note_type_name=note_type_name,
                field_name=field_name,
                show_identifying=show_identifying,
            )
            for note_type_name, field_name, show_identifying in entries
        )
        logger.info(f"  {prefix:<4} -> {owners}")
    logger.info("")
    logger.info("Free prefixes: any valid prefix not listed above.")
    logger.info("")

    logger.info("By note type:")
    sorted_configs = sorted(note_type_configs, key=lambda config_item: config_item.name)
    for config_index, config in enumerate(sorted_configs):
        prefix_parts = []
        base_identifying_prefixes = []
        for field_config in config.fields:
            if field_config.prefix is None:
                continue
            is_choice_field = config.is_choice and _is_choice_field(field_config.name)
            prefix_parts.append(field_config.prefix)
            if field_config.identifying and not is_choice_field:
                base_identifying_prefixes.append(field_config.prefix)
        logger.info(f"  {config.name}: {', '.join(prefix_parts)}")
        if config.is_choice:
            base_identifying = ", ".join(base_identifying_prefixes) or "(none)"
            logger.info(
                "    IDENTIFYING rule: base IDENTIFYING prefixes "
                f"({base_identifying}) + any one Choice n prefix"
            )
        else:
            identifying_rendered = ", ".join(base_identifying_prefixes) or "(none)"
            logger.info(f"    IDENTIFYING prefixes: {identifying_rendered}")
        if config_index < len(sorted_configs) - 1:
            logger.info("")


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
