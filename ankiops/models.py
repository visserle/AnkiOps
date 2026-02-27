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
            str(field_config.prefix)
            for field_config in self.fields
            if field_config.prefix is not None and field_config.identifying
        }

    @classmethod
    def validate_configs(cls, configs: list[NoteTypeConfig]) -> None:
        """Validate a set of note type configurations to ensure data integrity,
        local field uniqueness, strict built-in reservations, and lack
        of ambiguity.
        """
        reserved_names = {ANKIOPS_KEY_FIELD.name}
        built_in_prefixes = {
            str(field_config.prefix)
            for config in configs
            if config.name.startswith("AnkiOps")
            for field_config in config.fields
            if field_config.prefix is not None
        }
        custom_prefix_to_type: dict[str, str] = {}
        identifying_signature_to_type: dict[frozenset[str], str] = {}

        for config in configs:
            is_builtin = config.name.startswith("AnkiOps")
            seen_names: set[str] = set()
            seen_prefixes: set[str] = set()

            # 1. Reservation Checks (Names and Prefixes)
            for field_config in config.fields:
                if field_config.name in seen_names:
                    raise ValueError(
                        f"Note type '{config.name}' has duplicate field name "
                        f"'{field_config.name}'. "
                        "Field names must be unique within a note type."
                    )
                seen_names.add(field_config.name)

                if field_config.prefix is not None:
                    if field_config.prefix in seen_prefixes:
                        raise ValueError(
                            f"Note type '{config.name}' has duplicate field prefix "
                            f"'{field_config.prefix}'. "
                            "Field prefixes must be unique within a "
                            "note type."
                        )
                    seen_prefixes.add(field_config.prefix)

                if field_config.name in reserved_names:
                    if field_config.prefix != ANKIOPS_KEY_FIELD.prefix:
                        raise ValueError(
                            f"Note type '{config.name}' uses reserved "
                            f"field name '{field_config.name}' with invalid prefix. "
                            f"Expected no prefix."
                        )
                    if field_config.identifying != ANKIOPS_KEY_FIELD.identifying:
                        raise ValueError(
                            f"Note type '{config.name}' uses reserved "
                            f"field name '{field_config.name}' as identifying, "
                            "which is not allowed."
                        )

                if field_config.prefix is not None and not is_builtin:
                    if field_config.prefix in built_in_prefixes:
                        raise ValueError(
                            f"Note type '{config.name}' uses reserved built-in "
                            f"prefix '{field_config.prefix}'"
                        )

                    existing_type = custom_prefix_to_type.get(field_config.prefix)
                    if existing_type is not None and existing_type != config.name:
                        raise ValueError(
                            f"Note type '{config.name}' reuses custom prefix "
                            f"'{field_config.prefix}' already used by "
                            f"'{existing_type}'. "
                            "Prefixes must be unique across custom note types."
                        )
                    custom_prefix_to_type[field_config.prefix] = config.name

            # 2. Choice Validation
            if config.is_choice:
                has_choice_field = any(
                    "choice" in field_config.name.lower()
                    for field_config in config.fields
                )
                if not has_choice_field:
                    raise ValueError(
                        f"Note type '{config.name}' is marked as 'is_choice', "
                        "but no fields containing the word 'choice' were found. "
                        "Choice note types must have at least one field with "
                        "'choice' in its name."
                    )

            # 3. Distinctness Checks
            signature = frozenset(config.identifying_prefixes)
            existing_type = identifying_signature_to_type.get(signature)
            if existing_type is not None and existing_type != config.name:
                raise ValueError(
                    f"Note type '{config.name}' has identical identifying fields "
                    f"to '{existing_type}' ({set(signature)}), "
                    "which makes inference ambiguous."
                )
            identifying_signature_to_type[signature] = config.name


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

        has_choices = any(
            "Choice" in field_config.name for field_config in config.fields
        )

        for field_config in config.fields:
            if "Choice" in field_config.name:
                choice_fields.append(field_config.name)
                if self.fields.get(field_config.name):
                    choice_count += 1
                continue

            if (
                field_config.prefix is not None
                and field_config.identifying
                and not self.fields.get(field_config.name)
            ):
                errors.append(
                    f"Missing mandatory field '{field_config.name}' "
                    f"({field_config.prefix})"
                )

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

        parts = [answer_part.strip() for answer_part in answer.split(",")]
        try:
            answer_ints = [int(answer_part) for answer_part in parts]
        except ValueError:
            return [
                "AnkiOpsChoice answer (A:) must contain integers "
                "(e.g. '1' for single choice or '1, 2, 3' for multiple choice)"
            ]

        max_choice = max(
            (
                int(choice_field_name.split()[-1])
                for choice_field_name in choice_fields
                if self.fields.get(choice_field_name)
            ),
            default=0,
        )
        for answer_index in answer_ints:
            if answer_index < 1 or answer_index > max_choice:
                return [
                    f"AnkiOpsChoice answer contains '{answer_index}' but only "
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


# Shared mapping for projecting change types into summary counters.
_SUMMARY_FIELD_BY_CHANGE: dict[ChangeType, str] = {
    ChangeType.CREATE: "created",
    ChangeType.UPDATE: "updated",
    ChangeType.DELETE: "deleted",
    ChangeType.MOVE: "moved",
    ChangeType.SKIP: "skipped",
    ChangeType.HASH: "hashed",
    ChangeType.SYNC: "synced",
}

_NOTE_REPORTED_CHANGE_TYPES: frozenset[ChangeType] = frozenset(
    {
        ChangeType.CREATE,
        ChangeType.UPDATE,
        ChangeType.DELETE,
        ChangeType.MOVE,
        ChangeType.SKIP,
    }
)

_MEDIA_REPORTED_CHANGE_TYPES: frozenset[ChangeType] = _NOTE_REPORTED_CHANGE_TYPES | {
    ChangeType.HASH,
    ChangeType.SYNC,
}

_TOTAL_EXCLUDED_CHANGE_TYPES: frozenset[ChangeType] = frozenset(
    {ChangeType.DELETE, ChangeType.CONFLICT}
)

_NOTE_CHANGE_ORDER: tuple[ChangeType, ...] = (
    ChangeType.CREATE,
    ChangeType.UPDATE,
    ChangeType.DELETE,
    ChangeType.SKIP,
    ChangeType.MOVE,
)


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
    _FORMAT_ORDER = (
        "created",
        "updated",
        "deleted",
        "moved",
        "errors",
        "hashed",
        "synced",
    )

    def __add__(self, other: "SyncSummary") -> "SyncSummary":
        if not isinstance(other, SyncSummary):
            return NotImplemented
        return SyncSummary(
            **{
                field_name: getattr(self, field_name) + getattr(other, field_name)
                for field_name in self.__annotations__
            }
        )

    @classmethod
    def from_changes(
        cls,
        changes: list["Change"],
        errors: list[str],
        *,
        reported_change_types: frozenset[ChangeType],
    ) -> "SyncSummary":
        summary = cls(errors=len(errors))
        summary.add_changes(changes, reported_change_types=reported_change_types)
        return summary

    def add_changes(
        self,
        changes: list["Change"],
        *,
        reported_change_types: frozenset[ChangeType],
        dedupe_by_entity: bool = True,
        include_total: bool = True,
    ) -> None:
        change_types: list[ChangeType]
        if dedupe_by_entity:
            change_types = _effective_change_types(changes)
        else:
            change_types = [change.change_type for change in changes]

        for change_type in change_types:
            self.add_change_type(
                change_type,
                reported_change_types=reported_change_types,
                include_total=include_total,
            )

    def add_change_type(
        self,
        change_type: ChangeType,
        *,
        reported_change_types: frozenset[ChangeType],
        include_total: bool = True,
    ) -> None:
        if change_type in reported_change_types:
            field_name = _SUMMARY_FIELD_BY_CHANGE.get(change_type)
            if field_name:
                setattr(self, field_name, getattr(self, field_name) + 1)

        if include_total and change_type not in _TOTAL_EXCLUDED_CHANGE_TYPES:
            self.total += 1

    def to_dict(self) -> dict[str, int]:
        return {
            field_name: getattr(self, field_name) for field_name in self.__annotations__
        }

    def format(self) -> str:
        parts = []
        for key in self._FORMAT_ORDER:
            val = getattr(self, key)
            if val > 0:
                parts.append(f"{val} {key}")
        if not parts:
            return "no changes"
        return ", ".join(parts)


# Priority order for deduplicating overlapping changes on the same entity.
_CHANGE_PRIORITY = {
    change_type: priority
    for priority, change_type in enumerate(
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


def _effective_change_types(changes: list[Change]) -> list[ChangeType]:
    """Return the highest-priority change per entity."""
    effective: dict[int | str, ChangeType] = {}
    for change in changes:
        entity_id = change.entity_id or change.entity_repr
        previous = effective.get(entity_id)
        if (
            previous is None
            or _CHANGE_PRIORITY[change.change_type] > _CHANGE_PRIORITY[previous]
        ):
            effective[entity_id] = change.change_type
    return list(effective.values())


@dataclass
class ChangeBuckets:
    creates: list[Change] = field(default_factory=list)
    updates: list[Change] = field(default_factory=list)
    deletes: list[Change] = field(default_factory=list)
    skips: list[Change] = field(default_factory=list)
    moves: list[Change] = field(default_factory=list)
    others: list[Change] = field(default_factory=list)

    def add(self, change: Change) -> None:
        if change.change_type == ChangeType.CREATE:
            self.creates.append(change)
        elif change.change_type == ChangeType.UPDATE:
            self.updates.append(change)
        elif change.change_type == ChangeType.DELETE:
            self.deletes.append(change)
        elif change.change_type == ChangeType.SKIP:
            self.skips.append(change)
        elif change.change_type == ChangeType.MOVE:
            self.moves.append(change)
        else:
            self.others.append(change)

    def extend(self, changes: list[Change]) -> None:
        for change in changes:
            self.add(change)

    def ordered(
        self,
        *,
        order: tuple[ChangeType, ...] = _NOTE_CHANGE_ORDER,
        include_others: bool = True,
    ) -> list[Change]:
        ordered_changes: list[Change] = []
        for change_type in order:
            if change_type == ChangeType.CREATE:
                ordered_changes.extend(self.creates)
            elif change_type == ChangeType.UPDATE:
                ordered_changes.extend(self.updates)
            elif change_type == ChangeType.DELETE:
                ordered_changes.extend(self.deletes)
            elif change_type == ChangeType.SKIP:
                ordered_changes.extend(self.skips)
            elif change_type == ChangeType.MOVE:
                ordered_changes.extend(self.moves)

        if include_others:
            ordered_changes.extend(self.others)
        return ordered_changes


@dataclass
class SyncResult:
    name: str | None = None
    file_path: Path | None = None
    changes: list[Change] = field(default_factory=list)
    change_buckets: ChangeBuckets = field(default_factory=ChangeBuckets)
    errors: list[str] = field(default_factory=list)
    reported_change_types: frozenset[ChangeType] = _NOTE_REPORTED_CHANGE_TYPES
    checked: int = 0
    unchanged: int = 0
    missing: int = 0

    @classmethod
    def for_notes(cls, *, name: str, file_path: Path | None) -> "SyncResult":
        return cls(
            name=name,
            file_path=file_path,
            reported_change_types=_NOTE_REPORTED_CHANGE_TYPES,
        )

    @classmethod
    def for_media(cls) -> "SyncResult":
        return cls(
            name="media",
            file_path=None,
            reported_change_types=_MEDIA_REPORTED_CHANGE_TYPES,
        )

    def add_change(self, change: Change) -> None:
        self.change_buckets.add(change)

    def extend_changes_bucketed(self, changes: list[Change]) -> None:
        self.change_buckets.extend(changes)

    def materialize_changes(
        self,
        *,
        order: tuple[ChangeType, ...] = _NOTE_CHANGE_ORDER,
        include_others: bool = True,
    ) -> None:
        self.changes = self.change_buckets.ordered(
            order=order,
            include_others=include_others,
        )

    @property
    def summary(self) -> SyncSummary:
        return SyncSummary.from_changes(
            self.changes,
            self.errors,
            reported_change_types=self.reported_change_types,
        )


@dataclass
class UntrackedDeck:
    deck_name: str
    deck_id: int
    note_ids: list[int]


@dataclass
class CollectionResult:
    results: list[SyncResult] = field(default_factory=list)
    untracked_decks: list[UntrackedDeck] = field(default_factory=list)
    extra_changes: list[Change] = field(default_factory=list)
    extra_change_types: frozenset[ChangeType] = _MEDIA_REPORTED_CHANGE_TYPES
    include_extra_in_total: bool = False

    @classmethod
    def for_import(
        cls,
        *,
        results: list[SyncResult],
        untracked_decks: list[UntrackedDeck],
    ) -> "CollectionResult":
        return cls(results=results, untracked_decks=untracked_decks)

    @classmethod
    def for_export(
        cls,
        *,
        results: list[SyncResult],
        extra_changes: list[Change],
    ) -> "CollectionResult":
        return cls(
            results=results,
            extra_changes=extra_changes,
            extra_change_types=_MEDIA_REPORTED_CHANGE_TYPES,
            include_extra_in_total=False,
        )

    @property
    def summary(self) -> SyncSummary:
        base = sum(
            (sync_result.summary for sync_result in self.results),
            SyncSummary(),
        )
        if self.extra_changes:
            base.add_changes(
                self.extra_changes,
                reported_change_types=self.extra_change_types,
                dedupe_by_entity=False,
                include_total=self.include_extra_in_total,
            )
        return base


@dataclass
class MarkdownFile:
    """Represents a parsed Markdown file, fully detached from IO."""

    file_path: Path
    raw_content: str
    notes: list[Note]
