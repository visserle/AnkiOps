"""Reports produced by AnkiOps sync operations."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any


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
class SyncReport:
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
    def for_notes(cls, *, name: str, file_path: Path | None) -> "SyncReport":
        return cls(
            name=name,
            file_path=file_path,
            reported_change_types=_NOTE_REPORTED_CHANGE_TYPES,
        )

    @classmethod
    def for_media(cls) -> "SyncReport":
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
    def summary(self) -> "SyncSummary":
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
                field_info.name: getattr(self, field_info.name)
                + getattr(other, field_info.name)
                for field_info in fields(self)
            }
        )

    @classmethod
    def from_changes(
        cls,
        changes: list[Change],
        errors: list[str],
        *,
        reported_change_types: frozenset[ChangeType],
    ) -> "SyncSummary":
        summary = cls(errors=len(errors))
        summary.add_changes(changes, reported_change_types=reported_change_types)
        return summary

    def add_changes(
        self,
        changes: list[Change],
        *,
        reported_change_types: frozenset[ChangeType],
        dedupe_by_entity: bool = True,
        include_total: bool = True,
    ) -> None:
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
            field_info.name: getattr(self, field_info.name)
            for field_info in fields(self)
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
class CollectionReport:
    results: list[SyncReport] = field(default_factory=list)
    untracked_decks: list[UntrackedDeck] = field(default_factory=list)
    protected_note_groups: list[ProtectedNoteGroup] = field(default_factory=list)
    extra_changes: list[Change] = field(default_factory=list)
    extra_change_types: frozenset[ChangeType] = _MEDIA_REPORTED_CHANGE_TYPES
    include_extra_in_total: bool = False

    @classmethod
    def for_import(
        cls,
        *,
        results: list[SyncReport],
        untracked_decks: list[UntrackedDeck],
        protected_note_groups: list[ProtectedNoteGroup] | None = None,
    ) -> "CollectionReport":
        return cls(
            results=results,
            untracked_decks=untracked_decks,
            protected_note_groups=protected_note_groups or [],
        )

    @classmethod
    def for_export(
        cls,
        *,
        results: list[SyncReport],
        extra_changes: list[Change],
        protected_note_groups: list[ProtectedNoteGroup] | None = None,
    ) -> "CollectionReport":
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
