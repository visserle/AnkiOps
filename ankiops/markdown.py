"""AnkiOps Markdown deck file format."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ankiops.note_types import (
    ANKIOPS_KEY_FIELD,
    NoteType,
    build_label_to_field_map,
)
from ankiops.notes import Note, normalize_tags

NOTE_SEPARATOR = "\n\n---\n\n"

NOTE_KEY_COMMENT_RE = re.compile(r"^\s*<!--\s*note_key:\s*([a-zA-Z0-9-]+)\s*-->\s*$")
NOTE_TYPE_COMMENT_RE = re.compile(r"^\s*<!--\s*note_type:\s*(.*?)\s*-->\s*$")
TAGS_COMMENT_RE = re.compile(r"^\s*<!--\s*tags:\s*(.*?)\s*-->\s*$")
CODE_FENCE_RE = re.compile(r"^(```|~~~)")
FIELD_LABEL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
FIELD_LABEL_CANDIDATE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*:)(?:\s|$)")


@dataclass
class DeckFile:
    """A parsed Markdown deck file, detached from I/O."""

    file_path: Path
    raw_content: str
    notes: list[Note]


def format_note_key_comment(note_key: str) -> str:
    return f"<!-- note_key: {note_key} -->"


def format_note_type_comment(note_type: str) -> str:
    return f"<!-- note_type: {note_type} -->"


def parse_note_key_comment(line: str) -> str | None:
    match = NOTE_KEY_COMMENT_RE.match(line)
    return match.group(1) if match else None


def parse_note_type_comment(line: str) -> str | None:
    match = NOTE_TYPE_COMMENT_RE.match(line)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def is_note_type_comment(line: str) -> bool:
    return NOTE_TYPE_COMMENT_RE.match(line) is not None


def is_code_fence_line(line: str) -> bool:
    return CODE_FENCE_RE.match(line) is not None


def parse_tags_comment(line: str) -> tuple[str, ...] | None:
    """Parse a full-line tags comment, returning None when it is not one."""
    match = TAGS_COMMENT_RE.match(line)
    if not match:
        return None
    return normalize_tags(match.group(1))


def format_tags_comment(tags: Iterable[str] | str | None) -> str | None:
    """Format tags as a Markdown metadata comment."""
    normalized = normalize_tags(tags)
    if not normalized:
        return None
    return f"<!-- tags: {' '.join(normalized)} -->"


def read_deck_file(
    file_path: Path,
    *,
    note_types: list[NoteType],
    context_root: Path | None = None,
) -> DeckFile:
    raw_content = file_path.read_text(encoding="utf-8")
    parsed_notes = []
    config_by_name = {config.name: config for config in note_types}
    label_to_field = build_label_to_field_map(note_types)
    display_path = _display_parse_error_path(file_path, context_root=context_root)
    separator_newlines = NOTE_SEPARATOR.count("\n")
    block_start = 0
    block_start_line = 1

    while True:
        separator_index = raw_content.find(NOTE_SEPARATOR, block_start)
        if separator_index == -1:
            block = raw_content[block_start:]
            next_block_start = -1
        else:
            block = raw_content[block_start:separator_index]
            next_block_start = separator_index + len(NOTE_SEPARATOR)

        stripped_block = block.strip()
        if not stripped_block or not stripped_block.replace("-", ""):
            if next_block_start == -1:
                break
            block_start_line += block.count("\n") + separator_newlines
            block_start = next_block_start
            continue

        leading_trimmed_len = len(block) - len(block.lstrip())
        leading_line_offset = block[:leading_trimmed_len].count("\n")
        note_start_line = block_start_line + leading_line_offset

        lines = stripped_block.split("\n")
        note_key: str | None = None
        explicit_note_type: str | None = None
        tags: tuple[str, ...] = ()
        fields: dict[str, str] = {}
        current_field: str | None = None
        current_content: list[str] = []
        in_code_block = False
        seen: set[str] = set()
        seen_labels: set[str] = set()
        first_field_line: int | None = None

        for offset, line in enumerate(lines):
            current_line_number = note_start_line + offset
            stripped = line.lstrip()
            if is_code_fence_line(stripped):
                in_code_block = not in_code_block
                if current_field:
                    current_content.append(line)
                continue

            if in_code_block:
                if current_field:
                    current_content.append(line)
                continue

            parsed_note_key = parse_note_key_comment(line)
            if parsed_note_key is not None:
                note_key = parsed_note_key
                continue

            parsed_note_type = parse_note_type_comment(line)
            if parsed_note_type is not None:
                explicit_note_type = parsed_note_type
                continue

            tag_values = parse_tags_comment(line)
            if tag_values is not None:
                tags = tag_values
                continue

            matched_field = None
            for label, field_name in label_to_field.items():
                if line.startswith(label + " ") or line == label:
                    if field_name in seen:
                        raise ValueError(
                            _with_parse_error_context(
                                f"Duplicate field '{label}'.",
                                display_path=display_path,
                                line_number=current_line_number,
                            )
                        )

                    seen.add(field_name)
                    seen_labels.add(label)
                    if current_field:
                        fields[current_field] = "\n".join(current_content).strip()

                    matched_field = field_name
                    current_content = (
                        [line[len(label) + 1 :]] if line.startswith(label + " ") else []
                    )
                    current_field = field_name
                    if first_field_line is None:
                        first_field_line = current_line_number
                    break

            if matched_field is None and current_field:
                current_content.append(line)
            elif matched_field is None:
                label_match = FIELD_LABEL_CANDIDATE_RE.match(stripped)
                if label_match:
                    unknown_label = label_match.group(1)
                    if unknown_label not in label_to_field:
                        raise ValueError(
                            _with_parse_error_context(
                                (
                                    f"Unknown field label '{unknown_label}'. "
                                    "Please check your note type labels. "
                                    "Use `ankiops note-types` to list "
                                    "defined labels."
                                ),
                                display_path=display_path,
                                line_number=current_line_number,
                            )
                        )

        if current_field:
            fields[current_field] = "\n".join(current_content).strip()

        if not fields:
            raise ValueError(
                _with_parse_error_context(
                    (
                        "Found content but no valid field labels in block "
                        f"starting with: '{block.strip()[:50]}...'"
                    ),
                    display_path=display_path,
                    line_number=note_start_line,
                )
            )

        if explicit_note_type is not None:
            note_type = explicit_note_type
            if note_type not in config_by_name:
                raise ValueError(
                    _with_parse_error_context(
                        f"Unknown note type '{note_type}'.",
                        display_path=display_path,
                        line_number=first_field_line or note_start_line,
                    )
                )
        else:
            try:
                note_type = infer_note_type(note_types, fields, labels=seen_labels)
            except ValueError as error:
                raise ValueError(
                    _with_parse_error_context(
                        str(error),
                        display_path=display_path,
                        line_number=first_field_line or note_start_line,
                    )
                ) from error

        note = Note(
            note_key=note_key,
            note_type=note_type,
            fields=fields,
            tags=tags,
        )

        errors = note.validate(config_by_name[note_type])
        if errors:
            raise ValueError(
                _with_parse_error_context(
                    "Invalid note in block:\n  " + "\n  ".join(errors),
                    display_path=display_path,
                    line_number=first_field_line or note_start_line,
                )
            )

        parsed_notes.append(note)

        if next_block_start == -1:
            break
        block_start_line += block.count("\n") + separator_newlines
        block_start = next_block_start

    return DeckFile(
        file_path=file_path,
        raw_content=raw_content,
        notes=parsed_notes,
    )


def write_deck_file(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def find_deck_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.md"), key=lambda path: path.name)


def render_notes_to_markdown(
    notes: list[Note],
    note_types_by_name: dict[str, NoteType],
) -> str:
    content_parts: list[str] = []

    for note in notes:
        parts = []
        if note.note_key:
            parts.append(format_note_key_comment(note.note_key))
        parts.append(format_note_type_comment(note.note_type))
        tags_comment = format_tags_comment(note.tags)
        if tags_comment:
            parts.append(tags_comment)

        note_type = note_types_by_name[note.note_type]
        for field_config in note_type.fields:
            if (
                field_config.label
                and field_config.name in note.fields
                and note.fields[field_config.name]
            ):
                lines = note.fields[field_config.name].split("\n")
                parts.append(f"{field_config.label} {lines[0]}")
                if len(lines) > 1:
                    parts.extend(lines[1:])

        content_parts.append("\n".join(parts))

    return NOTE_SEPARATOR.join(content_parts) + "\n"


def infer_note_type(
    note_types: list[NoteType],
    fields: dict[str, str],
    *,
    labels: set[str] | None = None,
) -> str:
    note_fields = {
        field_name
        for field_name in fields.keys()
        if field_name != ANKIOPS_KEY_FIELD.name
    }
    note_labels = set(labels) if labels is not None else None
    candidates: list[str] = []

    for config in note_types:
        type_all_fields = {field_config.name for field_config in config.fields}
        if not note_fields.issubset(type_all_fields):
            continue

        if note_labels is not None:
            type_all_labels = {
                str(field_config.label)
                for field_config in config.fields
                if field_config.label is not None
            }
            if not note_labels.issubset(type_all_labels):
                continue

        if config.is_choice:
            base_ident_fields = {
                field_config.name
                for field_config in config.fields
                if field_config.identifying
                and "choice" not in field_config.name.lower()
            }
            choice_fields = {
                field_config.name
                for field_config in config.fields
                if "choice" in field_config.name.lower()
            }
            if not base_ident_fields.issubset(note_fields):
                continue
            if not note_fields.intersection(choice_fields):
                continue

            if note_labels is not None:
                base_ident_labels = {
                    str(field_config.label)
                    for field_config in config.fields
                    if field_config.identifying
                    and field_config.label is not None
                    and "choice" not in field_config.name.lower()
                }
                choice_labels = {
                    str(field_config.label)
                    for field_config in config.fields
                    if field_config.label is not None
                    and "choice" in field_config.name.lower()
                }
                if not base_ident_labels.issubset(note_labels):
                    continue
                if not note_labels.intersection(choice_labels):
                    continue
        else:
            type_ident_fields = {
                field_config.name
                for field_config in config.fields
                if field_config.identifying
            }
            if not type_ident_fields.issubset(note_fields):
                continue
            if note_labels is not None:
                type_ident_labels = {
                    str(field_config.label)
                    for field_config in config.fields
                    if field_config.identifying and field_config.label is not None
                }
                if not type_ident_labels.issubset(note_labels):
                    continue

        candidates.append(config.name)

    if not candidates:
        raise ValueError("Cannot determine note type from fields: " + ", ".join(fields))
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous note type: matches multiple types: {', '.join(candidates)}"
        )
    return candidates[0]


def _display_parse_error_path(
    file_path: Path,
    *,
    context_root: Path | None,
) -> str:
    if context_root is None:
        return file_path.name

    try:
        return str(file_path.resolve().relative_to(context_root.resolve()))
    except Exception:
        return file_path.name


def _with_parse_error_context(
    message: str,
    *,
    display_path: str,
    line_number: int,
) -> str:
    return f"{message} (file: {display_path}, line: {line_number})"
