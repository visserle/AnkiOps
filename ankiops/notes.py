"""Note-shaped data and note-local helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from blake3 import blake3

from ankiops.note_types import NoteType

_CLOZE_PATTERN = re.compile(r"\{\{c\d+::")


def normalize_tags(tags: Iterable[str] | str | None = None) -> tuple[str, ...]:
    """Return canonical Anki tags: trimmed, deduped, and sorted."""
    if tags is None:
        return ()

    raw_tags = tags.split() if isinstance(tags, str) else tags
    normalized = {
        tag.strip() for tag in raw_tags if isinstance(tag, str) and tag.strip()
    }
    return tuple(sorted(normalized))


def _stable_payload(
    note_type: str,
    fields: dict[str, str],
    *,
    tags: Iterable[str] | str | None = (),
) -> bytes:
    payload = {
        "note_type": note_type,
        "fields": fields,
        "tags": list(normalize_tags(tags)),
    }
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def note_fingerprint(
    note_type: str,
    fields: dict[str, str],
    *,
    tags: Iterable[str] | str | None = (),
) -> str:
    """Compute a stable note-level fingerprint from type + fields + tags."""
    return blake3(_stable_payload(note_type, fields, tags=tags)).hexdigest(length=8)


@dataclass
class Note:
    """A note parsed from Markdown."""

    note_key: str | None
    note_type: str
    fields: dict[str, str]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.tags = normalize_tags(self.tags)

    @property
    def identifier(self) -> str:
        """Stable identifier for error messages."""
        if self.note_key is not None:
            return f"note_key: {self.note_key}"
        first_line = self.first_field_line()
        return f"'{first_line[:60]}...'" if first_line else "unknown note"

    def first_field_line(self) -> str:
        """Return the first non-empty line of the first populated field."""
        for content in self.fields.values():
            if content:
                return content.split("\n")[0]
        return ""

    def validate(self, config: NoteType) -> list[str]:
        """Validate mandatory fields and note-type-specific rules."""
        errors: list[str] = []
        choice_count = 0
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
    fields: dict[str, str]
    card_ids: list[int]
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.tags = normalize_tags(self.tags)
