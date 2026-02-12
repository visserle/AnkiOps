"""Shared helpers for parsing and formatting markdown note blocks.

This module contains small utilities used by both import and export
paths: parsing a note block from markdown, extracting note blocks from
existing files, and formatting a note back to markdown.
"""

import logging
import re
from dataclasses import dataclass

from deckops.config import (
    ALL_PREFIX_TO_FIELD,
    NOTE_SEPARATOR,
    NOTE_TYPE_UNIQUE_PREFIXES,
    NOTE_TYPES,
)

logger = logging.getLogger(__name__)


@dataclass
class ParsedNote:
    note_id: int | None
    note_type: str  # "DeckOpsQA" or "DeckOpsCloze"
    fields: dict[str, str]
    raw_content: str
    line_number: int


def extract_deck_id(content: str) -> tuple[int | None, str]:
    """Extract deck_id from the first line and return (deck_id, remaining content)."""
    match = re.match(r"<!--\s*deck_id:\s*(\d+)\s*-->\n?", content)
    if match:
        return int(match.group(1)), content[match.end() :]
    return None, content


def extract_note_blocks(content: str) -> dict[str, str]:
    """Extract identified note blocks from content.

    Keys are ID strings like "note_id: 123".
    """
    _, content = extract_deck_id(content)
    blocks = content.split(NOTE_SEPARATOR)
    notes: dict[str, str] = {}
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        match = re.match(r"<!--\s*(note_id:\s*\d+)\s*-->", stripped)
        if match:
            key = re.sub(r"\s+", " ", match.group(1))
            notes[key] = stripped
    return notes


def _detect_note_type(fields):
    """Detect note type from unique field prefixes.

    Raises ValueError if no unique prefix (Q:, A:, T:) is found.
    """
    for field_name in fields:
        for prefix, note_type in NOTE_TYPE_UNIQUE_PREFIXES.items():
            if ALL_PREFIX_TO_FIELD.get(prefix) == field_name:
                return note_type
    raise ValueError(
        "Cannot determine note type: no Q:, A:, or T: field found. "
        "Every note must have at least one unique field prefix."
    )


def parse_note_block(block: str, line_number: int) -> ParsedNote:
    lines = block.strip().split("\n")
    note_id = None
    fields: dict[str, str] = {}
    current_field = None
    current_content: list[str] = []
    in_code_block = False
    seen_fields: dict[str, int] = {}  # field_name -> line offset where first seen

    for line_offset, line in enumerate(lines):
        # Track fenced code blocks (``` or ~~~) to avoid detecting
        # Q:/A:/T: prefixes inside code examples
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            if current_field:
                current_content.append(line)
            continue

        note_id_match = re.match(r"<!--\s*note_id:\s*(\d+)\s*-->", line)
        if note_id_match:
            note_id = int(note_id_match.group(1))
            continue

        # Inside code blocks, don't detect field prefixes
        if in_code_block:
            if current_field:
                current_content.append(line)
            continue

        new_field = None
        for prefix, field_name in ALL_PREFIX_TO_FIELD.items():
            if (
                line.startswith(prefix + " ")
                or line.startswith(prefix)
                and len(line) == len(prefix)
            ):
                # Check for duplicate field marker
                if field_name in seen_fields:
                    # Build ID reference for better error context
                    id_ref = f"note_id: {note_id}" if note_id is not None else ""

                    # Build context for error message
                    if id_ref:
                        context = f"in {id_ref}"
                    else:
                        first_line = line_number + seen_fields[field_name]
                        current_line = line_number + line_offset
                        context = (
                            f"at line {current_line} (first seen at line {first_line})"
                        )
                    msg = (
                        f"Duplicate field '{prefix}' {context}. "
                        f"Did you forget to end the previous note with a blank line, "
                        f"three dashes '---' and another blank line, or is "
                        f"there an accidental duplicate prefix? "
                    )
                    logger.error(msg)
                    raise ValueError(msg)

                new_field = field_name
                seen_fields[field_name] = line_offset
                if current_field:
                    fields[current_field] = "\n".join(current_content).strip()

                if line.startswith(prefix + " "):
                    current_content = [line[len(prefix) + 1 :]]
                else:
                    current_content = []
                current_field = new_field
                break

        if new_field is None and current_field:
            current_content.append(line)

    if current_field:
        fields[current_field] = "\n".join(current_content).strip()

    note_type = _detect_note_type(fields)

    return ParsedNote(
        note_id=note_id,
        note_type=note_type,
        fields=fields,
        raw_content=block,
        line_number=line_number,
    )


def validate_note(note: ParsedNote) -> list[str]:
    """Validate that all mandatory fields for the note type are present.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []
    note_config = NOTE_TYPES.get(note.note_type)
    if not note_config:
        errors.append(f"Unknown note type '{note.note_type}'")
        return errors

    for field_name, prefix, mandatory in note_config["field_mappings"]:
        if mandatory and not note.fields.get(field_name):
            errors.append(f"Missing mandatory field '{field_name}' ({prefix})")

    # Cloze notes must contain at least one cloze deletion in the Text field
    cloze_pattern = re.compile(r"\{\{c\d+::")
    if note.note_type == "DeckOpsCloze":
        text = note.fields.get("Text", "")
        if text and not cloze_pattern.search(text):
            errors.append(
                "DeckOpsCloze note must contain cloze syntax "
                "(e.g. {{c1::answer}}) in the T: field"
            )

    return errors


def format_note(
    note_id: int, note: dict, converter, note_type: str = "DeckOpsQA"
) -> str:
    note_config = NOTE_TYPES[note_type]
    field_mappings = note_config["field_mappings"]

    lines = [f"<!-- note_id: {note_id} -->"]
    fields = note["fields"]

    for field_name, prefix, mandatory in field_mappings:
        field_data = fields.get(field_name)
        if field_data:
            markdown = converter.convert(field_data.get("value", ""))
            if markdown or mandatory:
                lines.append(f"{prefix} {markdown}")

    return "\n".join(lines)


def sanitize_filename(deck_name: str) -> str:
    """Sanitize deck name for use as filename.

    Raises:
        ValueError: If deck name contains invalid filename characters.
    """
    # Check for invalid characters before sanitization
    invalid_chars = ["/", "\\", "?", "*", "|", '"', "<", ">", ":"]
    # Note: We allow '::' as it's Anki's hierarchy separator and will be replaced
    invalid_in_name = [c for c in invalid_chars if c in deck_name and c != ":"]

    if invalid_in_name:
        raise ValueError(
            f"Deck name '{deck_name}' contains invalid filename characters: {invalid_in_name}\n"
            f'Please rename the deck in Anki to remove these characters: / \\ ? * | " < >'
        )

    # Check for Windows reserved names
    reserved_names = [
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    ]
    # Get the base name (before any :: hierarchy separators)
    base_name = deck_name.split("::")[0].upper()
    if base_name in reserved_names:
        raise ValueError(
            f"Deck name '{deck_name}' starts with Windows reserved name '{base_name}'.\n"
            f"Please rename the deck in Anki to avoid: {', '.join(reserved_names)}"
        )

    return deck_name.replace("::", "__")
