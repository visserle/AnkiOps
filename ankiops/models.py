"""Core data models for AnkiOps.

Provides typed data classes for both the markdown side (Note, FileState)
and the Anki side (AnkiNote, AnkiState), plus sync result types
(SyncResult, ImportSummary, ExportSummary) shared by import/export paths.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ankiops.anki_client import invoke
from ankiops.config import (
    MANDATORY_FIELDS,
    NOTE_CONFIG,
    NOTE_SEPARATOR,
    SUPPORTED_NOTE_TYPES,
)

# prefix -> field name mapping for all note types, e.g. "Q:" -> "Question"
# Dynamically built from NOTE_CONFIG to avoid duplication
PREFIX_TO_FIELD: dict[str, str] = {}
for _field_mappings in NOTE_CONFIG.values():
    for _field_name, _prefix in _field_mappings:
        PREFIX_TO_FIELD[_prefix] = _field_name
FIELD_TO_PREFIX = {v: k for k, v in PREFIX_TO_FIELD.items()}

_CLOZE_PATTERN = re.compile(r"\{\{c\d+::")
_NOTE_ID_PATTERN = re.compile(r"<!--\s*note_id:\s*(\d+)\s*-->")
_DECK_ID_PATTERN = re.compile(r"<!--\s*deck_id:\s*(\d+)\s*-->\n?")
_CODE_FENCE_PATTERN = re.compile(r"^(```|~~~)")


logger = logging.getLogger(__name__)


@dataclass
class InvalidID:
    """An ID from markdown that cannot be matched to an existing Anki entity."""

    id_value: int
    id_type: str  # "deck_id" or "note_id"
    file_path: Path
    context: str  # additional context (e.g., note identifier)


@dataclass
class FileState:
    """All data parsed from one markdown file in a single read.

    Unified class used by both import and export paths.
    """

    file_path: Path
    raw_content: str
    deck_id: int | None
    parsed_notes: list[Note]

    @staticmethod
    def extract_deck_id(content: str) -> tuple[int | None, str]:
        """Extract deck_id from the first line and return (deck_id, remaining)."""
        match = _DECK_ID_PATTERN.match(content)
        if match:
            return int(match.group(1)), content[match.end() :]
        return None, content

    @staticmethod
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

    @staticmethod
    def from_file(file_path: Path) -> FileState:
        """Read and parse a markdown file."""
        raw_content = file_path.read_text(encoding="utf-8")
        deck_id, remaining = FileState.extract_deck_id(raw_content)
        blocks = remaining.split(NOTE_SEPARATOR)
        parsed_notes = [Note.from_block(block) for block in blocks if block.strip()]
        return FileState(
            file_path=file_path,
            raw_content=raw_content,
            deck_id=deck_id,
            parsed_notes=parsed_notes,
        )

    @property
    def note_ids(self) -> set[int]:
        """All note IDs present in this file."""
        return {n.note_id for n in self.parsed_notes if n.note_id is not None}

    @property
    def has_untracked(self) -> bool:
        """True if the file contains notes without a note_id."""
        return any(n.note_id is None for n in self.parsed_notes)

    @property
    def existing_blocks(self) -> dict[str, str]:
        """Identified note blocks keyed by ``"note_id: 123"``.

        Used by the export path for block-level text comparison.
        Derived from the file's raw content (not from individual notes).
        """
        _, remaining = FileState.extract_deck_id(self.raw_content)
        return FileState.extract_note_blocks(remaining)

    @staticmethod
    def validate_ids(
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
            if fs.deck_id is not None and fs.deck_id not in valid_deck_ids:
                invalid_ids.append(
                    InvalidID(
                        id_value=fs.deck_id,
                        id_type="deck_id",
                        file_path=fs.file_path,
                        context=f"deck_id in {fs.file_path.name}",
                    )
                )

            for note in fs.parsed_notes:
                if note.note_id is not None and note.note_id not in valid_note_ids:
                    invalid_ids.append(
                        InvalidID(
                            id_value=note.note_id,
                            id_type="note_id",
                            file_path=fs.file_path,
                            context=f"{note.identifier} in {fs.file_path.name}",
                        )
                    )

        return invalid_ids

    def validate_no_duplicate_first_lines(
        self,
        id_assignments: list[tuple[Note, int]],
    ) -> None:
        """Raise if new notes share a first line (would break text-based ID insertion)."""
        first_lines: dict[str, list[str]] = {}
        for note, _ in id_assignments:
            if note.note_id is not None:
                continue
            first_line = note.first_line
            first_lines.setdefault(first_line, []).append(note.identifier)

        duplicates = {line: ids for line, ids in first_lines.items() if len(ids) > 1}
        if duplicates:
            msg = f"ERROR: Duplicate first lines detected in {self.file_path.name}:\n"
            for first_line, ids in duplicates.items():
                msg += f"  '{first_line[:60]}...' in notes: {', '.join(ids)}\n"
            msg += (
                "Cannot safely assign IDs. "
                "Please ensure each note has a unique first line."
            )
            raise ValueError(msg)


@dataclass
class Note:
    """A note parsed from a markdown block.

    Single class for all note types.  The ``note_type`` field (e.g.
    ``"AnkiOpsQA"``, ``"AnkiOpsCloze"``) drives behaviour via config lookup.
    """

    note_id: int | None
    note_type: str
    fields: dict[str, str]  # {field_name: markdown_content}

    @staticmethod
    def infer_note_type(fields: dict[str, str]) -> str:
        """Infer note type from parsed fields.

        Checks all note types from most specific to least specific,
        with AnkiOpsQA as the generic catch-all.
        """
        items = []
        for note_type, mandatory_fields in MANDATORY_FIELDS.items():
            if note_type == "AnkiOpsChoice":
                continue
            elif note_type == "AnkiOpsQA":
                identifying = {"Question", "Answer"}
            else:
                identifying = {f[0] for f in mandatory_fields}
            items.append((note_type, identifying))

        # Sort by number of identifying fields (descending), then name
        # But ensure AnkiOpsQA is always last
        items.sort(key=lambda x: (x[0] == "AnkiOpsQA", -len(x[1]), x[0]))

        # Special case: Choice notes are inferred by Question + ANY Choice field
        if "Question" in fields and any(k.startswith("Choice ") for k in fields):
            return "AnkiOpsChoice"

        for note_type, identifying in items:
            if identifying.issubset(fields.keys()):
                return note_type

        raise ValueError(
            "Cannot determine note type from fields: " + ", ".join(fields.keys())
        )

    @staticmethod
    def from_block(block: str) -> Note:
        """Parse a raw markdown block into a Note."""
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
            for prefix, field_name in PREFIX_TO_FIELD.items():
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

        return Note(
            note_id=note_id,
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
            prefix = FIELD_TO_PREFIX.get(field_name, "")
            first_content = content.split("\n")[0] if content else ""
            if prefix and first_content:
                return f"{prefix} {first_content}"
            if prefix:
                return prefix
        return ""

    @property
    def identifier(self) -> str:
        """Stable identifier for error messages."""
        if self.note_id:
            return f"note_id: {self.note_id}"
        return f"'{self.first_line[:60]}...'"

    def validate(self) -> list[str]:
        """Validate mandatory fields and note-type-specific rules.

        Returns a list of error messages (empty if valid).
        """
        errors: list[str] = []
        if self.note_type not in NOTE_CONFIG:
            errors.append(f"Unknown note type '{self.note_type}'")
            return errors

        choice_count = 0
        choice_fields = []
        for field_name, prefix in MANDATORY_FIELDS.get(self.note_type, []):
            if "Choice" in field_name:
                choice_fields.append(field_name)
                if self.fields.get(field_name):
                    choice_count += 1
                continue # Skip individual Choice field check in mandatory loop

            if not self.fields.get(field_name):
                errors.append(f"Missing mandatory field '{field_name}' ({prefix})")

        if self.note_type == "AnkiOpsCloze":
            text = self.fields.get("Text", "")
            if text and not _CLOZE_PATTERN.search(text):
                errors.append(
                    "AnkiOpsCloze note must contain cloze syntax "
                    "(e.g. {{c1::answer}}) in the T: field"
                )

        if self.note_type == "AnkiOpsChoice":
            if choice_count < 2:
                errors.append("AnkiOpsChoice note must have at least 2 choices")
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

    def to_html(self, converter) -> dict[str, str]:
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

        for field_name, _ in NOTE_CONFIG.get(self.note_type, []):
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

    deck_names_and_ids: dict[str, int]
    id_to_deck_name: dict[int, str]
    notes: dict[int, AnkiNote]  # note_id -> typed AnkiNote
    cards: dict[int, dict]  # card_id -> raw AnkiConnect card dict
    deck_note_ids: dict[str, set[int]]  # deck_name -> {note_id, ...}

    @staticmethod
    def fetch() -> AnkiState:
        deck_names_and_ids = invoke("deckNamesAndIds")
        id_to_deck_name = {v: k for k, v in deck_names_and_ids.items()}

        query = " OR ".join(f"note:{nt}" for nt in SUPPORTED_NOTE_TYPES)
        all_card_ids = invoke("findCards", query=query)

        cards: dict[int, dict] = {}
        deck_note_ids: dict[str, set[int]] = {}
        all_note_ids: set[int] = set()

        if all_card_ids:
            for card in invoke("cardsInfo", cards=all_card_ids):
                cards[card["cardId"]] = card
                deck_note_ids.setdefault(card["deckName"], set()).add(card["note"])
                all_note_ids.add(card["note"])

        notes: dict[int, AnkiNote] = {}
        if all_note_ids:
            for note in invoke("notesInfo", notes=list(all_note_ids)):
                if not note:
                    continue
                model = note.get("modelName")
                if model and model not in SUPPORTED_NOTE_TYPES:
                    raise ValueError(
                        f"Safety check failed: Note {note['noteId']} has template "
                        f"'{model}' but expected a AnkiOps template. "
                        f"AnkiOps will never modify notes with non-AnkiOps templates."
                    )
                anki_note = AnkiNote.from_raw(note)
                notes[anki_note.note_id] = anki_note

        return AnkiState(
            deck_names_and_ids=deck_names_and_ids,
            id_to_deck_name=id_to_deck_name,
            notes=notes,
            cards=cards,
            deck_note_ids=deck_note_ids,
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

    def to_markdown(self, converter) -> str:
        """Format this Anki note as a markdown block.

        Args:
            converter: HTMLToMarkdown converter instance

        Returns:
            Markdown block string starting with ``<!-- note_id: ... -->``.
        """
        note_config = NOTE_CONFIG[self.note_type]
        lines = [f"<!-- note_id: {self.note_id} -->"]

        for field_name, prefix in note_config:
            value = self.fields.get(field_name, "")
            if value:
                md = converter.convert(value)
                if md:
                    lines.append(f"{prefix} {md}")

        return "\n".join(lines)


@dataclass
class SyncResult:
    """Result of syncing a single file or deck.

    Used by both import (markdown → Anki) and export (Anki → markdown) paths.
    """

    deck_name: str
    file_path: Path | None
    total_notes: int
    updated: int
    created: int
    deleted: int
    moved: int
    skipped: int
    errors: list[str] = field(default_factory=list)
    renamed_from: str | None = None


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
