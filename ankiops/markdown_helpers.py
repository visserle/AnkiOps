"""Shared helpers for parsing and formatting markdown note blocks.

Used by both import (markdown_to_anki) and export (anki_to_markdown) paths.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ankiops.config import (
    ALL_PREFIX_TO_FIELD,
    NOTE_SEPARATOR,
    NOTE_TYPES,
)

logger = logging.getLogger(__name__)

_CLOZE_PATTERN = re.compile(r"\{\{c\d+::")
_NOTE_ID_PATTERN = re.compile(r"<!--\s*note_id:\s*(\d+)\s*-->")
_DECK_ID_PATTERN = re.compile(r"<!--\s*deck_id:\s*(\d+)\s*-->\n?")
_CODE_FENCE_PATTERN = re.compile(r"^(```|~~~)")

_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class ParsedNote:
    note_id: int | None
    note_type: str
    fields: dict[str, str]
    raw_content: str


@dataclass
class InvalidID:
    """An ID from markdown that cannot be matched to an existing Anki entity."""

    id_value: int
    id_type: str  # "deck_id" or "note_id"
    file_path: Path
    context: str  # additional context (e.g., note identifier)


@dataclass
class FileState:
    """All data parsed from one markdown file in a single read."""

    file_path: Path
    raw_content: str
    deck_id: int | None
    parsed_notes: list[ParsedNote]

    @staticmethod
    def from_file(file_path: Path) -> "FileState":
        raw_content = file_path.read_text(encoding="utf-8")
        deck_id, remaining = extract_deck_id(raw_content)
        blocks = remaining.split(NOTE_SEPARATOR)
        parsed_notes = [parse_note_block(block) for block in blocks if block.strip()]
        return FileState(
            file_path=file_path,
            raw_content=raw_content,
            deck_id=deck_id,
            parsed_notes=parsed_notes,
        )


def note_identifier(note: ParsedNote) -> str:
    """Stable identifier for error messages (note_id or first content line)."""
    if note.note_id:
        return f"note_id: {note.note_id}"
    first_line = note.raw_content.strip().split("\n")[0][:60]
    return f"'{first_line}...'"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def extract_deck_id(content: str) -> tuple[int | None, str]:
    """Extract deck_id from the first line and return (deck_id, remaining)."""
    match = _DECK_ID_PATTERN.match(content)
    if match:
        return int(match.group(1)), content[match.end() :]
    return None, content


def extract_note_blocks(cards_content: str) -> dict[str, str]:
    """Extract identified note blocks from content.

    Args:
        cards_content: Content with deck_id already stripped
            (the output of extract_deck_id).

    Returns {"note_id: 123": block_content, ...}.
    """
    notes: dict[str, str] = {}
    for block in cards_content.split(NOTE_SEPARATOR):
        stripped = block.strip()
        if not stripped:
            continue
        match = _NOTE_ID_PATTERN.match(stripped)
        if match:
            notes[f"note_id: {match.group(1)}"] = stripped
    return notes


def has_untracked_notes(cards_content: str) -> bool:
    """Check if content has note blocks without note_id comments.

    Args:
        cards_content: Content with deck_id already stripped
            (the output of extract_deck_id).

    Returns True if there are notes that haven't been imported to Anki yet.
    """
    for block in cards_content.split(NOTE_SEPARATOR):
        stripped = block.strip()
        if not stripped:
            continue
        # Check if block has any field prefixes but no note_id
        has_fields = any(
            stripped.startswith(prefix + " ") or ("\n" + prefix + " ") in stripped
            for prefix in ALL_PREFIX_TO_FIELD.keys()
        )
        has_note_id = _NOTE_ID_PATTERN.match(stripped) is not None

        if has_fields and not has_note_id:
            return True
    return False


def _detect_note_type(fields: dict[str, str]) -> str:
    """Detect note type from parsed fields (most specific first)."""
    if "Choice 1" in fields:
        return "AnkiOpsChoice"
    if "Text" in fields:
        return "AnkiOpsCloze"
    if "Question" in fields or "Answer" in fields:
        return "AnkiOpsQA"
    raise ValueError("Cannot determine note type: no Q:, A:, T:, or C1: field found")


def parse_note_block(block: str) -> ParsedNote:
    """Parse a raw markdown block into a ParsedNote."""
    lines = block.strip().split("\n")
    note_id: int | None = None
    fields: dict[str, str] = {}
    current_field: str | None = None
    current_content: list[str] = []
    in_code_block = False
    seen: set[str] = set()

    for line in lines:
        stripped = line.lstrip()

        # Track fenced code blocks to avoid detecting prefixes inside code
        if _CODE_FENCE_PATTERN.match(stripped):
            in_code_block = not in_code_block
            if current_field:
                current_content.append(line)
            continue

        # Note ID comment
        id_match = _NOTE_ID_PATTERN.match(line)
        if id_match:
            note_id = int(id_match.group(1))
            continue

        # Inside code blocks, don't detect field prefixes
        if in_code_block:
            if current_field:
                current_content.append(line)
            continue

        # Try to match a field prefix
        matched_field = None
        for prefix, field_name in ALL_PREFIX_TO_FIELD.items():
            if line.startswith(prefix + " ") or line == prefix:
                # Duplicate field check
                if field_name in seen:
                    ctx = f"in note_id: {note_id}" if note_id else "in this note"
                    msg = (
                        f"Duplicate field '{prefix}' {ctx}. "
                        f"Did you forget to end the previous note with "
                        f"'\\n\\n---\\n\\n' "
                        f"or is there an accidental duplicate prefix?"
                    )
                    logger.error(msg)
                    raise ValueError(msg)

                seen.add(field_name)
                if current_field:
                    fields[current_field] = "\n".join(current_content).strip()

                matched_field = field_name
                current_content = (
                    [line[len(prefix) + 1 :]] if line.startswith(prefix + " ") else []
                )
                current_field = field_name
                break

        if matched_field is None and current_field:
            current_content.append(line)

    if current_field:
        fields[current_field] = "\n".join(current_content).strip()

    return ParsedNote(
        note_id=note_id,
        note_type=_detect_note_type(fields),
        fields=fields,
        raw_content=block,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_note(note: ParsedNote) -> list[str]:
    """Validate mandatory fields and note-type-specific rules.

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

    if note.note_type == "AnkiOpsCloze":
        text = note.fields.get("Text", "")
        if text and not _CLOZE_PATTERN.search(text):
            errors.append(
                "AnkiOpsCloze note must contain cloze syntax "
                "(e.g. {{c1::answer}}) in the T: field"
            )

    if note.note_type == "AnkiOpsChoice":
        errors.extend(_validate_choice_answers(note))

    return errors


def _validate_choice_answers(note: ParsedNote) -> list[str]:
    """Validate AnkiOpsChoice answer format and range."""
    answer = note.fields.get("Answer", "")
    if not answer:
        return []

    parts = [p.strip() for p in answer.split(",")]
    try:
        answer_ints = [int(p) for p in parts]
    except ValueError:
        return [
            "AnkiOpsChoice answer (A:) must contain integers "
            "(e.g. '1' for single choice or '1, 2, 3' for multiple choice)"
        ]

    max_choice = max(
        (i for i in range(1, 8) if note.fields.get(f"Choice {i}")),
        default=0,
    )
    for n in answer_ints:
        if n < 1 or n > max_choice:
            return [
                f"AnkiOpsChoice answer contains '{n}' but only "
                f"{max_choice} choice(s) are provided"
            ]
    return []


def validate_markdown_ids(
    file_states: list[FileState],
    valid_deck_ids: set[int],
    valid_note_ids: set[int],
) -> list[InvalidID]:
    """Check all deck and note IDs in markdown files against valid ID sets.

    Args:
        file_states: List of parsed file states
        valid_deck_ids: Set of deck IDs that exist in Anki
        valid_note_ids: Set of note IDs that exist in Anki

    Returns:
        List of IDs that exist in markdown but not in the valid sets
    """
    invalid_ids: list[InvalidID] = []

    for fs in file_states:
        # Check deck_id
        if fs.deck_id is not None and fs.deck_id not in valid_deck_ids:
            invalid_ids.append(
                InvalidID(
                    id_value=fs.deck_id,
                    id_type="deck_id",
                    file_path=fs.file_path,
                    context=f"deck_id in {fs.file_path.name}",
                )
            )

        # Check note_ids
        for note in fs.parsed_notes:
            if note.note_id is not None and note.note_id not in valid_note_ids:
                invalid_ids.append(
                    InvalidID(
                        id_value=note.note_id,
                        id_type="note_id",
                        file_path=fs.file_path,
                        context=f"{note_identifier(note)} in {fs.file_path.name}",
                    )
                )

    return invalid_ids


# ---------------------------------------------------------------------------
# Formatting (Anki → Markdown)
# ---------------------------------------------------------------------------


def format_note(
    note_id: int,
    note: dict,
    converter,
    note_type: str = "AnkiOpsQA",
) -> str:
    """Format an Anki note dict into a markdown block.

    ``note`` is the raw AnkiConnect notesInfo dict with
    ``note["fields"]["FieldName"]["value"]`` structure.
    """
    field_mappings = NOTE_TYPES[note_type]["field_mappings"]
    lines = [f"<!-- note_id: {note_id} -->"]

    for field_name, prefix, mandatory in field_mappings:
        field_data = note["fields"].get(field_name)
        if field_data:
            md = converter.convert(field_data.get("value", ""))
            if md or mandatory:
                lines.append(f"{prefix} {md}")

    return "\n".join(lines)


def convert_fields_to_html(
    fields: dict[str, str],
    converter,
) -> dict[str, str]:
    """Convert all field values from markdown to HTML.

    Args:
        fields: Dictionary mapping field names to markdown content
        converter: MarkdownToHTML converter instance

    Returns:
        Dictionary mapping field names to HTML content
    """
    return {name: converter.convert(content) for name, content in fields.items()}


def sanitize_filename(deck_name: str) -> str:
    """Convert deck name to a safe filename (``::`` → ``__``).

    Raises ValueError for invalid characters or Windows reserved names.
    """
    invalid = [c for c in r'/\?*|"<>' if c in deck_name and c != ":"]
    if invalid:
        raise ValueError(
            f"Deck name '{deck_name}' contains invalid filename characters: "
            f"{invalid}\nPlease rename the deck in Anki to remove these."
        )

    base = deck_name.split("::")[0].upper()
    if base in _WINDOWS_RESERVED:
        raise ValueError(
            f"Deck name '{deck_name}' starts with Windows reserved name "
            f"'{base}'.\nPlease rename the deck in Anki."
        )

    return deck_name.replace("::", "__")
