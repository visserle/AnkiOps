"""Core Domain Models for AnkiOps.

These classes are pure data representations with no dependencies on
external systems (Anki, file system, SQLite).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ankiops.tags import normalize_tags

_CLOZE_PATTERN = re.compile(r"\{\{c\d+::")


@dataclass(frozen=True)
class Field:
    """Definition of a field in a note type."""

    name: str
    label: str | None
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
    def identifying_labels(self) -> set[str]:
        """Return labels of identifying fields."""
        return {
            str(field_config.label)
            for field_config in self.fields
            if field_config.label is not None and field_config.identifying
        }

    @classmethod
    def validate_configs(cls, configs: list[NoteTypeConfig]) -> None:
        """Validate note type configs using global set invariants.

        Invariants:
        - Field names/labels are unique within each note type.
        - A label maps to a single field name globally.
        - A label has one identifying flag globally.
        - Identifying label signatures are unique across note types.
        - Choice note types have identifying base and identifying choice fields.
        """
        reserved_names = {ANKIOPS_KEY_FIELD.name}
        label_to_field_name: dict[str, str] = {}
        label_to_identifying: dict[str, bool] = {}
        identifying_signature_to_type: dict[frozenset[str], str] = {}

        for config in configs:
            seen_names: set[str] = set()
            seen_labels: set[str] = set()
            identifying_choice_labels: set[str] = set()
            identifying_non_choice_labels: set[str] = set()
            has_choice_field = False

            # 1) Local uniqueness + reserved fields + global label invariants.
            for field_config in config.fields:
                if field_config.name in seen_names:
                    raise ValueError(
                        f"Note type '{config.name}' has duplicate field name "
                        f"'{field_config.name}'. "
                        "Field names must be unique within a note type."
                    )
                seen_names.add(field_config.name)

                if field_config.label is not None:
                    if field_config.label in seen_labels:
                        raise ValueError(
                            f"Note type '{config.name}' has duplicate field label "
                            f"'{field_config.label}'. "
                            "Field labels must be unique within a "
                            "note type."
                        )
                    seen_labels.add(field_config.label)

                if field_config.name in reserved_names:
                    if field_config.label != ANKIOPS_KEY_FIELD.label:
                        raise ValueError(
                            f"Note type '{config.name}' uses reserved "
                            f"field name '{field_config.name}' with invalid label. "
                            "Expected no label."
                        )
                    if field_config.identifying != ANKIOPS_KEY_FIELD.identifying:
                        raise ValueError(
                            f"Note type '{config.name}' uses reserved "
                            f"field name '{field_config.name}' as identifying, "
                            "which is not allowed."
                        )

                if field_config.label is not None:
                    existing_name = label_to_field_name.get(field_config.label)
                    if existing_name is not None and existing_name != field_config.name:
                        raise ValueError(
                            f"Label '{field_config.label}' maps to both "
                            f"'{existing_name}' and '{field_config.name}'."
                        )
                    label_to_field_name[field_config.label] = field_config.name

                    existing_identifying = label_to_identifying.get(field_config.label)
                    if (
                        existing_identifying is not None
                        and existing_identifying != field_config.identifying
                    ):
                        raise ValueError(
                            f"Label '{field_config.label}' has conflicting "
                            "identifying flag across note types."
                        )
                    label_to_identifying[field_config.label] = field_config.identifying

                    if "choice" in field_config.name.lower():
                        has_choice_field = True
                        if field_config.identifying:
                            identifying_choice_labels.add(field_config.label)
                    elif field_config.identifying:
                        identifying_non_choice_labels.add(field_config.label)

            # 2) Choice constraints for inference safety.
            if config.is_choice:
                if not has_choice_field:
                    raise ValueError(
                        f"Note type '{config.name}' is marked as 'is_choice', "
                        "but no fields containing the word 'choice' were found. "
                        "Choice note types must have at least one field with "
                        "'choice' in its name."
                    )
                if not identifying_non_choice_labels:
                    raise ValueError(
                        f"Note type '{config.name}' is marked as 'is_choice', "
                        "but has no identifying non-choice field."
                    )
                if not identifying_choice_labels:
                    raise ValueError(
                        f"Note type '{config.name}' is marked as 'is_choice', "
                        "but has no identifying choice field."
                    )

            # 3) Distinct identifying signatures across note types.
            signature = frozenset(config.identifying_labels)
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
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.tags = normalize_tags(self.tags)

    @property
    def identifier(self) -> str:
        """Stable identifier for error messages."""
        if self.note_key is not None:
            return f"note_key: {self.note_key}"
        # For new notes falling back to first line
        first_line = self.first_field_line()
        return f"'{first_line[:60]}...'" if first_line else "unknown note"

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
            "choice" in field_config.name.lower() for field_config in config.fields
        )

        for field_config in config.fields:
            if "choice" in field_config.name.lower():
                choice_fields.append(field_config.name)
                if self.fields.get(field_config.name):
                    choice_count += 1
                continue

            if (
                field_config.label is not None
                and field_config.identifying
                and not self.fields.get(field_config.name)
            ):
                errors.append(
                    f"Missing mandatory field '{field_config.name}' "
                    f"({field_config.label})"
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
        else:
            has_cloze = any(
                _CLOZE_PATTERN.search(val) for val in self.fields.values() if val
            )
            if has_cloze:
                errors.append(
                    f"{self.note_type} note must not contain cloze syntax "
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
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.tags = normalize_tags(self.tags)


class ChangeType(Enum):
    CREATE = ("created", True, 4, True, True, True)
    UPDATE = ("updated", True, 3, True, True, False)
    CONVERT = ("converted", True, 3, True, False, False)
    DELETE = ("deleted", False, 6, True, True, True)
    MOVE = ("moved", True, 5, True, True, True)
    SKIP = ("skipped", True, 0, True, True, False)

    CONFLICT = (None, False, 7, False, False, False)
    HASH = ("hashed", True, 1, False, True, False)
    SYNC = ("synced", True, 2, False, True, False)

    def __init__(
        self,
        summary_field: str | None,
        counts_total: bool,
        priority: int,
        report_in_notes: bool,
        report_in_media: bool,
        affects_membership: bool,
    ) -> None:
        self.summary_field = summary_field
        self.counts_total = counts_total
        self.priority = priority
        self.report_in_notes = report_in_notes
        self.report_in_media = report_in_media
        self.affects_membership = affects_membership

    @classmethod
    def note_reported(cls) -> frozenset["ChangeType"]:
        return frozenset(
            change_type for change_type in cls if change_type.report_in_notes
        )

    @classmethod
    def media_reported(cls) -> frozenset["ChangeType"]:
        return frozenset(
            change_type for change_type in cls if change_type.report_in_media
        )


@dataclass
class Change:
    change_type: ChangeType
    entity_id: int | str | None
    entity_repr: str
    context: dict[str, Any] = field(default_factory=dict)


_NOTE_REPORTED_CHANGE_TYPES = ChangeType.note_reported()
_MEDIA_REPORTED_CHANGE_TYPES = ChangeType.media_reported()
_NOTE_CHANGE_ORDER: tuple[ChangeType, ...] = (
    ChangeType.CREATE,
    ChangeType.CONVERT,
    ChangeType.UPDATE,
    ChangeType.DELETE,
    ChangeType.SKIP,
    ChangeType.MOVE,
)


@dataclass
class SyncResult:
    name: str | None = None
    file_path: Path | None = None
    changes: list[Change] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    reported_change_types: frozenset[ChangeType] = _NOTE_REPORTED_CHANGE_TYPES
    checked: int = 0
    unchanged: int = 0
    missing: int = 0
    protected_keyless_notes: int = 0

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
        self.changes.append(change)

    def changes_for(self, change_type: ChangeType) -> list[Change]:
        return [change for change in self.changes if change.change_type == change_type]

    def has_changes(self, *change_types: ChangeType) -> bool:
        return any(change.change_type in change_types for change in self.changes)

    def order_changes(
        self,
        *,
        order: tuple[ChangeType, ...] = _NOTE_CHANGE_ORDER,
        include_unordered: bool = True,
    ) -> None:
        ordered_types = set(order)
        ordered_changes = [
            change
            for change_type in order
            for change in self.changes
            if change.change_type == change_type
        ]
        if include_unordered:
            ordered_changes.extend(
                change
                for change in self.changes
                if change.change_type not in ordered_types
            )
        self.changes = ordered_changes

    @property
    def summary(self) -> SyncSummary:
        return SyncSummary.from_changes(
            self.changes,
            self.errors,
            reported_change_types=self.reported_change_types,
        )


def _effective_change_types(changes: list[Change]) -> list[ChangeType]:
    """Return the highest-priority change per entity."""
    effective: dict[int | str, ChangeType] = {}
    for change in changes:
        entity_id = change.entity_id or change.entity_repr
        previous = effective.get(entity_id)
        if previous is None or change.change_type.priority > previous.priority:
            effective[entity_id] = change.change_type
    return list(effective.values())


@dataclass
class SyncSummary:
    total: int = 0
    created: int = 0
    updated: int = 0
    converted: int = 0
    deleted: int = 0
    moved: int = 0
    skipped: int = 0
    errors: int = 0
    hashed: int = 0
    synced: int = 0
    _FORMAT_ORDER = (
        "created",
        "updated",
        "converted",
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
            field_name = change_type.summary_field
            if field_name:
                setattr(self, field_name, getattr(self, field_name) + 1)

        if include_total and change_type.counts_total:
            self.total += 1

    def to_dict(self) -> dict[str, int]:
        return {
            field_name: getattr(self, field_name)
            for field_name in type(self).__annotations__
        }

    @staticmethod
    def format_change_counts(**counts: int) -> str:
        """Format non-zero change counts into a compact string."""
        parts = []
        for label_key, count_value in counts.items():
            if count_value <= 0 or label_key == "total":
                continue
            label = (
                label_key[:-1]
                if count_value == 1 and label_key.endswith("s")
                else label_key
            )
            parts.append(f"{count_value} {label}")
        return ", ".join(parts) if parts else "no changes"

    def format(self) -> str:
        ordered_counts = {key: getattr(self, key) for key in self._FORMAT_ORDER}
        return self.format_change_counts(**ordered_counts)


@dataclass
class UntrackedDeck:
    deck_name: str
    deck_id: int
    note_ids: list[int]


@dataclass
class ProtectedNoteGroup:
    deck_name: str
    note_count: int


@dataclass
class CollectionResult:
    results: list[SyncResult] = field(default_factory=list)
    untracked_decks: list[UntrackedDeck] = field(default_factory=list)
    protected_note_groups: list[ProtectedNoteGroup] = field(default_factory=list)
    extra_changes: list[Change] = field(default_factory=list)
    extra_change_types: frozenset[ChangeType] = _MEDIA_REPORTED_CHANGE_TYPES
    include_extra_in_total: bool = False

    @classmethod
    def for_import(
        cls,
        *,
        results: list[SyncResult],
        untracked_decks: list[UntrackedDeck],
        protected_note_groups: list[ProtectedNoteGroup] | None = None,
    ) -> "CollectionResult":
        return cls(
            results=results,
            untracked_decks=untracked_decks,
            protected_note_groups=protected_note_groups or [],
        )

    @classmethod
    def for_export(
        cls,
        *,
        results: list[SyncResult],
        extra_changes: list[Change],
        protected_note_groups: list[ProtectedNoteGroup] | None = None,
    ) -> "CollectionResult":
        return cls(
            results=results,
            protected_note_groups=protected_note_groups or [],
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
