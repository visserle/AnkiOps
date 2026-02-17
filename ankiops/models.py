"""Core data models for AnkiOps.

Provides typed data classes for both the markdown side (Note, FileState)
and the Anki side (AnkiNote, AnkiState), plus sync result types
(SyncResult, ImportSummary, ExportSummary) shared by import/export paths.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ankiops.markdown_converter import MarkdownToHTML

from ankiops.anki_client import invoke
from ankiops.config import NOTE_SEPARATOR
from ankiops.note_type_config import registry

_CLOZE_PATTERN = re.compile(r"\{\{c\d+::")
_NOTE_KEY_PATTERN = re.compile(r"<!--\s*note_key:\s*([a-zA-Z0-9-]+)\s*-->")
_DECK_KEY_PATTERN = re.compile(r"<!--\s*deck_key:\s*([a-zA-Z0-9-]+)\s*-->\n?")
_CODE_FENCE_PATTERN = re.compile(r"^(```|~~~)")


logger = logging.getLogger(__name__)


@dataclass
class FileState:
    """All data parsed from one markdown file in a single read.

    Unified class used by both import and export paths.
    """

    file_path: Path
    raw_content: str
    deck_key: str | None
    parsed_notes: list[Note]

    @staticmethod
    def extract_deck_key(content: str) -> tuple[str | None, str]:
        """Extract deck_key from the first line.

        Returns (deck_key, remaining_content).
        """
        match = _DECK_KEY_PATTERN.match(content)
        if match:
            return match.group(1), content[match.end() :]

        return None, content

    @staticmethod
    def extract_note_blocks(cards_content: str) -> dict[str, str]:
        """Extract identified note blocks from content.

        Args:
            cards_content: Content with deck_key already stripped.

        Returns {"note_key: a1b2c3d4...": block_content, ...}.
        """

        notes: dict[str, str] = {}
        for block in cards_content.split(NOTE_SEPARATOR):
            stripped = block.strip()
            if not stripped:
                continue
            match = _NOTE_KEY_PATTERN.match(stripped)
            if match:
                notes[f"note_key: {match.group(1)}"] = stripped
        return notes

    @staticmethod
    def from_file(file_path: Path) -> FileState:
        """Read and parse a markdown file."""
        raw_content = file_path.read_text(encoding="utf-8")
        deck_key, remaining = FileState.extract_deck_key(raw_content)
        blocks = remaining.split(NOTE_SEPARATOR)
        parsed_notes = []
        for block in blocks:
            # Skip empty or dashes-only blocks
            # (whitespace/trailing separators)
            if not block.strip() or set(block.strip()) <= {"-"}:
                continue

            note = Note.from_block(block)
            parsed_notes.append(note)

        return FileState(
            file_path=file_path,
            raw_content=raw_content,
            deck_key=deck_key,
            parsed_notes=parsed_notes,
        )

    @property
    def note_keys(self) -> set[str]:
        """All note keys present in this file."""
        return {n.note_key for n in self.parsed_notes if n.note_key is not None}

    @property
    def has_untracked(self) -> bool:
        """True if the file contains notes without a note_key."""
        return any(n.note_key is None for n in self.parsed_notes)

    @property
    def existing_notes(self) -> list[Note]:
        """Notes that already have a note_key."""
        return [n for n in self.parsed_notes if n.note_key is not None]

    @property
    def new_notes(self) -> list[Note]:
        """Notes that do not have a note_key (newly created)."""
        return [n for n in self.parsed_notes if n.note_key is None]

    @property
    def existing_blocks(self) -> dict[str, str]:
        """Identified note blocks keyed by ``"note_key: a1b2..."``.

        Used by the export path for block-level text comparison.
        Derived from the file's raw content (not from individual notes).
        """
        _, remaining = FileState.extract_deck_key(self.raw_content)
        return FileState.extract_note_blocks(remaining)

    def validate_no_duplicate_first_lines(
        self,
        key_assignments: list[tuple[Note, str]],
    ) -> None:
        """Raise if new notes share a first line.

        Would break text-based key insertion.
        """
        first_lines: dict[str, list[str]] = {}
        for note, _ in key_assignments:
            if note.note_key is not None:
                continue
            first_line = note.first_line
            first_lines.setdefault(first_line, []).append(note.identifier)

        duplicates = {line: ids for line, ids in first_lines.items() if len(ids) > 1}
        if duplicates:
            msg = f"ERROR: Duplicate first lines detected in {self.file_path.name}:\n"
            for first_line, ids in duplicates.items():
                msg += f"  '{first_line[:60]}...' in notes: {', '.join(ids)}\n"
            msg += (
                "Cannot safely assign keys. "
                "Please ensure each note has a unique first line."
            )
            raise ValueError(msg)


@dataclass
class Note:
    """A note parsed from a markdown block.

    Single class for all note types.  The ``note_type`` field (e.g.
    ``"AnkiOpsQA"``, ``"AnkiOpsCloze"``) drives behaviour via config lookup.
    """

    note_key: str | None
    note_type: str
    fields: dict[str, str]  # {field_name: markdown_content}

    @staticmethod
    def infer_note_type(fields: dict[str, str]) -> str:
        """Infer note type from parsed fields.

        Finds the note type that contains all fields present in the note.
        If multiple types match (e.g. QA matches {Q, A}, Choice matches {Q, A, C1}),
        prefers the "tightest" fit (smallest number of defined fields).
        """
        # Exclude common fields (Extra, More, etc.) from logic
        # as they do not distinguish note types.
        common_names = registry._COMMON_NAMES
        note_fields = {k for k in fields.keys() if k not in common_names}
        candidates = []

        for name in registry.supported_note_types:
            config = registry.get(name)
            type_fields = set(config.identifying_field_names)

            # Check 1: Note fields must be a subset of the Type's fields
            if note_fields.issubset(type_fields):
                candidates.append((name, len(type_fields)))

        if not note_fields:
            raise ValueError("Cannot determine note type: only common fields found")

        if not candidates:
            raise ValueError(
                "Cannot determine note type from fields: " + ", ".join(fields.keys())
            )

        # Sort by size (ascending).
        # {Q, A} fits QA (size 2) and Choice (size 10). QA comes first.
        # {Q, A, C1} fits Choice (size 10). QA rejects it. Choice comes first.
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    @staticmethod
    def from_block(block: str) -> Note:
        """Parse a raw markdown block into a Note."""
        lines = block.strip().split("\n")
        note_key: str | None = None
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

            # Key comment
            key_match = _NOTE_KEY_PATTERN.match(line)
            if key_match:
                note_key = key_match.group(1)
                continue

            # Inside code blocks, don't detect field prefixes
            if in_code_block:
                if current_field:
                    current_content.append(line)
                continue

            # Try to match a field prefix
            matched_field = None
            for prefix, field_name in registry.prefix_to_field.items():
                if line.startswith(prefix + " ") or line == prefix:
                    # Duplicate field check
                    if field_name in seen:
                        ctx = f"in note_key: {note_key}" if note_key else "in this note"
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
                        [line[len(prefix) + 1 :]]
                        if line.startswith(prefix + " ")
                        else []
                    )
                    current_field = field_name
                    break

            if matched_field is None and current_field:
                current_content.append(line)

        if current_field:
            fields[current_field] = "\n".join(current_content).strip()

        if not fields:
            # We reached here with a non-empty block
            # (guaranteed by from_file filtering) but found
            # no fields — block has content but no prefixes.
            raise ValueError(
                f"Found content but no valid field prefixes (e.g. 'Q: ', 'A: ') "
                f"in block starting with: '{block.strip()[:50]}...'"
            )

        return Note(
            note_key=note_key,
            note_type=Note.infer_note_type(fields),
            fields=fields,
        )

    @property
    def first_line(self) -> str:
        """First content line of the note block (prefix + first line of content).

        Used for text-based note_id insertion in ``_flush_writes`` and
        for duplicate detection.  Reconstructed from parsed fields.
        """
        for field_name, content in self.fields.items():
            prefix = registry.field_to_prefix.get(field_name, "")
            first_content = content.split("\n")[0] if content else ""
            if prefix and first_content:
                return f"{prefix} {first_content}"
            if prefix:
                return prefix
        return ""

    @property
    def identifier(self) -> str:
        """Stable identifier for error messages."""
        if self.note_key is not None:
            return f"note_key: {self.note_key}"
        return f"'{self.first_line[:60]}...'"

    def validate(self) -> list[str]:
        """Validate mandatory fields and note-type-specific rules.

        Returns a list of error messages (empty if valid).
        """
        errors: list[str] = []
        try:
            config = registry.get(self.note_type)
        except KeyError:
            errors.append(f"Unknown note type '{self.note_type}'")
            return errors

        choice_count = 0
        choice_fields = []

        # Check if this is a Choice-like note type
        # We detect this by checking if it has choice-like fields in its definition
        has_choices = any("Choice" in f.name for f in config.fields)

        identifying = registry.identifying_fields.get(self.note_type, [])

        for field_name, prefix in identifying:
            if "Choice" in field_name:
                choice_fields.append(field_name)
                if self.fields.get(field_name):
                    choice_count += 1
                continue  # Skip individual Choice field check in mandatory loop

            # Non-Choice fields (Question, Answer, etc.) are mandatory if defined in the type
            if not self.fields.get(field_name):
                errors.append(f"Missing mandatory field '{field_name}' ({prefix})")

        if config.is_cloze:
            text = self.fields.get("Text", "")
            if text and not _CLOZE_PATTERN.search(text):
                errors.append(
                    f"{self.note_type} note must contain cloze syntax "
                    "(e.g. {{c1::answer}}) in the T: field"
                )

        if has_choices:
            if choice_count < 2:
                errors.append(f"{self.note_type} note must have at least 2 choices")
            errors.extend(self._validate_choice_answers(choice_fields))

        return errors

    def _validate_choice_answers(self, choice_fields: list[str]) -> list[str]:
        """Validate AnkiOpsChoice answer format and range."""
        answer = self.fields.get("Answer", "")
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
            (int(f.split()[-1]) for f in choice_fields if self.fields.get(f)),
            default=0,
        )
        for n in answer_ints:
            if n < 1 or n > max_choice:
                return [
                    f"AnkiOpsChoice answer contains '{n}' but only "
                    f"{max_choice} choice(s) are provided"
                ]
        return []

    def to_html(self, converter: "MarkdownToHTML") -> dict[str, str]:
        """Convert all field values from markdown to HTML.

        The returned dict contains an entry for every field defined by
        this note type.  Fields absent from ``self.fields`` get an empty
        string, so that Anki clears them when the user removes an
        optional field from the markdown.

        Args:
            converter: MarkdownToHTML converter instance

        Returns:
            Dictionary mapping field names to HTML content.
        """
        html = {
            name: converter.convert(content) for name, content in self.fields.items()
        }

        for field_name, _ in registry.note_config.get(self.note_type, []):
            html.setdefault(field_name, "")

        return html

    def html_fields_match(
        self, html_fields: dict[str, str], anki_note: AnkiNote
    ) -> bool:
        """Check if converted HTML fields match an AnkiNote's fields.

        Args:
            html_fields: Output of ``self.to_html(converter)``.
            anki_note: The Anki-side note to compare against.

        Returns:
            True if no update is needed.
        """
        return all(anki_note.fields.get(k) == v for k, v in html_fields.items())


@dataclass
class AnkiState:
    """All Anki-side data, fetched once.

    Built by ``AnkiState.fetch()`` with 3-4 API calls:
      1. deckNamesAndIds
      2. findCards  (all AnkiOps cards)
      3. cardsInfo  (details for found cards)
      4. notesInfo  (details for discovered note IDs)
    """

    deck_ids_by_name: dict[str, int]
    deck_names_by_id: dict[int, str]
    notes_by_id: dict[int, AnkiNote]  # note_id -> typed AnkiNote
    cards_by_id: dict[int, dict]  # card_id -> raw AnkiConnect card dict
    note_ids_by_deck_name: dict[str, set[int]]  # deck_name -> {note_id, ...}

    @staticmethod
    def fetch() -> AnkiState:
        deck_ids_by_name = invoke("deckNamesAndIds")
        deck_names_by_id = {v: k for k, v in deck_ids_by_name.items()}

        query = " OR ".join(f"note:{nt}" for nt in registry.supported_note_types)
        all_card_ids = invoke("findCards", query=query)

        cards_by_id: dict[int, dict] = {}
        note_ids_by_deck_name: dict[str, set[int]] = {}
        all_note_ids: set[int] = set()

        if all_card_ids:
            for card in invoke("cardsInfo", cards=all_card_ids):
                cards_by_id[card["cardId"]] = card
                note_ids_by_deck_name.setdefault(card["deckName"], set()).add(
                    card["note"]
                )
                all_note_ids.add(card["note"])

        notes_by_id: dict[int, AnkiNote] = {}
        if all_note_ids:
            for note in invoke("notesInfo", notes=list(all_note_ids)):
                if not note:
                    continue
                model = note.get("modelName")
                if model and model not in registry.supported_note_types:
                    raise ValueError(
                        f"Safety check failed: Note {note['noteId']} has template "
                        f"'{model}' but expected a AnkiOps note type. "
                        f"AnkiOps will never modify notes with non-AnkiOps templates."
                    )
                anki_note = AnkiNote.from_raw(note)
                notes_by_id[anki_note.note_id] = anki_note

        return AnkiState(
            deck_ids_by_name=deck_ids_by_name,
            deck_names_by_id=deck_names_by_id,
            notes_by_id=notes_by_id,
            cards_by_id=cards_by_id,
            note_ids_by_deck_name=note_ids_by_deck_name,
        )


@dataclass
class AnkiNote:
    """A note as it exists in Anki (wrapping the raw AnkiConnect dict).

    Provides typed access to note data instead of raw dict indexing.
    """

    note_id: int
    note_type: str  # modelName
    fields: dict[str, str]  # {field_name: value} (HTML content, extracted)
    card_ids: list[int]

    @staticmethod
    def from_raw(raw_note: dict) -> AnkiNote:
        """Create an AnkiNote from a raw AnkiConnect notesInfo dict.

        Extracts fields from raw_note["fields"][field_name]["value"] structure.
        """
        return AnkiNote(
            note_id=raw_note["noteId"],
            note_type=raw_note.get("modelName", ""),
            fields={name: data["value"] for name, data in raw_note["fields"].items()},
            card_ids=raw_note.get("cards", []),
        )

    def to_markdown(self, converter, note_key: str) -> str:
        """Format this Anki note as a markdown block.

        Args:
            converter: HTMLToMarkdown converter instance
            note_key: The key to write in the markdown header

        Returns:
            Markdown block string starting with ``<!-- note_key: ... -->``.
        """
        note_config = registry.note_config[self.note_type]
        lines = [f"<!-- note_key: {note_key} -->"]

        for field_name, prefix in note_config:
            value = self.fields.get(field_name, "")
            if value:
                md = converter.convert(value)
                if md:
                    lines.append(f"{prefix} {md}")

        return "\n".join(lines)


class ChangeType(Enum):
    CREATE = auto()
    UPDATE = auto()
    DELETE = auto()
    MOVE = auto()
    SKIP = auto()
    CONFLICT = auto()


@dataclass
class Change:
    change_type: ChangeType
    entity_id: int | None
    entity_repr: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncResult:
    """Result of syncing a single file or deck.

    Used by both import (markdown → Anki) and export (Anki → markdown) paths.
    """

    deck_name: str
    file_path: Path | None
    changes: list[Change] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def note_count(self) -> int:
        """Total valid notes processed (excluding deletes/conflicts)."""
        return sum(
            1
            for c in self.changes
            if c.change_type
            in (ChangeType.CREATE, ChangeType.UPDATE, ChangeType.MOVE, ChangeType.SKIP)
        )

    @property
    def updated_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.UPDATE)

    @property
    def created_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.CREATE)

    @property
    def deleted_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.DELETE)

    @property
    def moved_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.MOVE)

    @property
    def skipped_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.SKIP)


@dataclass
class UntrackedDeck:
    """An Anki deck with AnkiOps notes but no matching markdown file."""

    deck_name: str
    deck_id: int
    note_ids: list[int]


@dataclass
class ImportSummary:
    """Aggregate result of a full collection import."""

    file_results: list[SyncResult]
    untracked_decks: list[UntrackedDeck]


@dataclass
class ExportSummary:
    """Aggregate result of a full collection export."""

    deck_results: list[SyncResult]
    renamed_files: int
    deleted_deck_files: int
    deleted_orphan_notes: int
