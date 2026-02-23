"""Core Domain Models for AnkiOps.

These classes are pure data representations with no dependencies on
external systems (AnkiConnect, file system, SQLite).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

_CLOZE_PATTERN = re.compile(r"\{\{c\d+::")


@dataclass(frozen=True)
class Field:
    """Definition of a field in a note type."""

    name: str
    prefix: str | None
    identifying: bool


ANKIOPS_KEY_FIELD = Field("AnkiOps Key", None, identifying=False)


@dataclass
class NoteTypeConfig:
    """Pure data configuration for a single note type."""

    name: str
    fields: list[Field]
    css: str = ""
    is_cloze: bool = False
    is_choice: bool = False
    templates: list[dict[str, str]] = field(default_factory=list)

    @property
    def identifying_prefixes(self) -> set[str]:
        """Return prefixes of identifying fields."""
        return {
            str(f.prefix) for f in self.fields if f.prefix is not None and f.identifying
        }

    @classmethod
    def validate_configs(cls, configs: list[NoteTypeConfig]) -> None:
        """Validate a set of note type configurations to ensure data integrity,
        global consistency of field mappings, strict built-in reservations,
        and lack of ambiguity.
        """
        reserved_names = {ANKIOPS_KEY_FIELD.name}

        # Gather global prefix mappings and built-in prefixes first
        prefix_to_field: dict[str, str] = {}
        built_in_prefixes: set[str] = set()

        for config in configs:
            is_builtin = config.name.startswith("AnkiOps")
            if is_builtin:
                built_in_prefixes.update(config.identifying_prefixes)

            # Build global prefix mapping / validate global consistency
            for field in config.fields:
                if field.prefix is not None:
                    if field.prefix in prefix_to_field:
                        existing_name = prefix_to_field[field.prefix]
                        if existing_name != field.name:
                            raise ValueError(
                                f"Prefix '{field.prefix}' matches existing prefix for "
                                f"'{existing_name}', but maps to different field "
                                f"'{field.name}' in '{config.name}'"
                            )
                    else:
                        prefix_to_field[field.prefix] = field.name

        for config in configs:
            is_builtin = config.name.startswith("AnkiOps")

            # 1. Reservation Checks (Names and Prefixes)
            for field in config.fields:
                if field.name in reserved_names:
                    if field.prefix != ANKIOPS_KEY_FIELD.prefix:
                        raise ValueError(
                            f"Note type '{config.name}' uses reserved "
                            f"field name '{field.name}' with invalid prefix. "
                            f"Expected no prefix."
                        )
                    if field.identifying != ANKIOPS_KEY_FIELD.identifying:
                        raise ValueError(
                            f"Note type '{config.name}' uses reserved "
                            f"field name '{field.name}' as identifying, which is not allowed."
                        )

                if field.prefix is not None and field.prefix in built_in_prefixes:
                    if not is_builtin:
                        raise ValueError(
                            f"Note type '{config.name}' uses reserved built-in "
                            f"prefix '{field.prefix}'"
                        )

            # 2. Distinctness Checks
            new_prefixes = config.identifying_prefixes
            for existing_config in configs:
                if existing_config.name == config.name:
                    continue

                if new_prefixes == existing_config.identifying_prefixes:
                    raise ValueError(
                        f"Note type '{config.name}' has identical identifying fields "
                        f"to '{existing_config.name}' ({new_prefixes}), "
                        "which makes inference ambiguous."
                    )

            # 3. Choice Validation
            if config.is_choice:
                has_choice_field = any(
                    "choice" in field.name.lower() for field in config.fields
                )
                if not has_choice_field:
                    raise ValueError(
                        f"Note type '{config.name}' is marked as 'is_choice', but no fields "
                        "containing the word 'choice' were found. Choice note types must "
                        "have at least one field with 'choice' in its name."
                    )


@dataclass
class Note:
    """A note parsed from markdown."""

    note_key: str | None
    note_type: str
    fields: dict[str, str]  # {field_name: markdown_content}

    @property
    def identifier(self) -> str:
        """Stable identifier for error messages."""
        if self.note_key is not None:
            return f"note_key: {self.note_key}"
        # For new notes falling back to first line
        return f"'{self.first_field_line()[:60]}...'"

    def first_field_line(self) -> str:
        """Return the first non-empty line of the first populated field."""
        for content in self.fields.values():
            if content:
                return content.split("\n")[0]
        return ""

    def validate(self, config: NoteTypeConfig) -> list[str]:
        """Validate mandatory fields and note-type-specific rules."""
        errors: list[str] = []
        choice_count: int = 0
        choice_fields = []

        has_choices = any("Choice" in f.name for f in config.fields)

        for f in config.fields:
            if "Choice" in f.name:
                choice_fields.append(f.name)
                if self.fields.get(f.name):
                    choice_count += 1
                continue

            if f.prefix is not None and f.identifying and not self.fields.get(f.name):
                errors.append(f"Missing mandatory field '{f.name}' ({f.prefix})")

        if config.is_cloze:
            has_cloze = any(
                _CLOZE_PATTERN.search(val) for val in self.fields.values() if val
            )
            if not has_cloze:
                errors.append(
                    f"{self.note_type} note must contain cloze syntax "
                    "(e.g. {{c1::answer}})"
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


@dataclass
class AnkiNote:
    """A note as it exists in Anki."""

    note_id: int
    note_type: str
    fields: dict[str, str]  # HTML content
    card_ids: list[int]


class ChangeType(Enum):
    CREATE = auto()
    UPDATE = auto()
    DELETE = auto()
    MOVE = auto()
    SKIP = auto()
    CONFLICT = auto()
    HASH = auto()
    SYNC = auto()


@dataclass
class Change:
    change_type: ChangeType
    entity_id: int | str | None
    entity_repr: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncSummary:
    total: int = 0
    created: int = 0
    updated: int = 0
    deleted: int = 0
    moved: int = 0
    skipped: int = 0
    errors: int = 0
    hashed: int = 0
    synced: int = 0

    def __add__(self, other: "SyncSummary") -> "SyncSummary":
        if not isinstance(other, SyncSummary):
            return NotImplemented
        return SyncSummary(
            **{f: getattr(self, f) + getattr(other, f) for f in self.__annotations__}
        )

    def to_dict(self) -> dict[str, int]:
        return {f: getattr(self, f) for f in self.__annotations__}

    def format(self) -> str:
        parts = []
        for key in [
            "created",
            "updated",
            "deleted",
            "moved",
            "errors",
            "hashed",
            "synced",
        ]:
            val = getattr(self, key)
            if val > 0:
                parts.append(f"{val} {key}")
        if not parts:
            return "no changes"
        return ", ".join(parts)


# Priority order for deduplicating overlapping changes on the same entity.
_CHANGE_PRIORITY = {
    ct: i
    for i, ct in enumerate(
        [
            ChangeType.SKIP,
            ChangeType.HASH,
            ChangeType.SYNC,
            ChangeType.UPDATE,
            ChangeType.CREATE,
            ChangeType.MOVE,
            ChangeType.DELETE,
            ChangeType.CONFLICT,
        ]
    )
}


def _compute_summary(
    changes: list[Change],
    errors: list[str],
    mapping: dict[ChangeType, str],
) -> SyncSummary:
    """Compute a SyncSummary from a list of changes, deduplicating by entity."""
    effective: dict[int | str, ChangeType] = {}
    for c in changes:
        eid = c.entity_id or c.entity_repr
        prev = effective.get(eid)
        if prev is None or _CHANGE_PRIORITY[c.change_type] > _CHANGE_PRIORITY[prev]:
            effective[eid] = c.change_type

    res = {"total": 0, "errors": len(errors)}
    for ct in effective.values():
        if field_name := mapping.get(ct):
            res[field_name] = res.get(field_name, 0) + 1
        if ct not in (ChangeType.DELETE, ChangeType.CONFLICT):
            res["total"] += 1
    return SyncSummary(**res)


_NOTE_MAPPING = {
    ChangeType.CREATE: "created",
    ChangeType.UPDATE: "updated",
    ChangeType.DELETE: "deleted",
    ChangeType.MOVE: "moved",
    ChangeType.SKIP: "skipped",
}

_MEDIA_MAPPING = {
    **_NOTE_MAPPING,
    ChangeType.HASH: "hashed",
    ChangeType.SYNC: "synced",
}


@dataclass
class NoteSyncResult:
    deck_name: str
    file_path: Path | None
    changes: list[Change] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def summary(self) -> SyncSummary:
        return _compute_summary(self.changes, self.errors, _NOTE_MAPPING)


@dataclass
class MediaSyncResult:
    changes: list[Change] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def summary(self) -> SyncSummary:
        return _compute_summary(self.changes, self.errors, _MEDIA_MAPPING)


@dataclass
class UntrackedDeck:
    deck_name: str
    deck_id: int
    note_ids: list[int]


@dataclass
class CollectionImportResult:
    results: list[NoteSyncResult] = field(default_factory=list)
    untracked_decks: list[UntrackedDeck] = field(default_factory=list)

    @property
    def summary(self) -> SyncSummary:
        return sum((r.summary for r in self.results), SyncSummary())


@dataclass
class CollectionExportResult:
    results: list[NoteSyncResult] = field(default_factory=list)
    extra_changes: list[Change] = field(default_factory=list)

    @property
    def summary(self) -> SyncSummary:
        base = sum((r.summary for r in self.results), SyncSummary())
        mapping = {
            ChangeType.CREATE: "created",
            ChangeType.UPDATE: "updated",
            ChangeType.DELETE: "deleted",
            ChangeType.MOVE: "moved",
            ChangeType.SKIP: "skipped",
            ChangeType.HASH: "hashed",
            ChangeType.SYNC: "synced",
        }
        for c in self.extra_changes:
            if field_name := mapping.get(c.change_type):
                setattr(base, field_name, getattr(base, field_name) + 1)
        return base


@dataclass
class Deck:
    deck_id: int
    name: str


@dataclass
class MarkdownFile:
    """Represents a parsed Markdown file, fully detached from IO."""

    file_path: Path
    raw_content: str
    notes: list[Note]
