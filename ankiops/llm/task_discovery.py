"""Discovery selection and DB recording for task execution."""

from __future__ import annotations

import logging

from .discovery import DiscoveryItem, DiscoverySnapshot
from .llm_db import LlmDb
from .task_runtime_types import EligibleCandidate
from .task_types import LlmItemStatus


def record_discovery_snapshot(
    *,
    db: LlmDb,
    job_id: int,
    snapshot: DiscoverySnapshot,
    logger: logging.Logger,
) -> list[EligibleCandidate]:
    candidates: list[EligibleCandidate] = []
    for item in snapshot.items:
        candidate = _record_discovery_item(
            db=db,
            job_id=job_id,
            item=item,
            logger=logger,
        )
        if candidate is not None:
            candidates.append(candidate)

    db.set_discovery_counts(
        job_id=job_id,
        decks_seen=snapshot.counts.decks_seen,
        decks_matched=snapshot.counts.decks_matched,
        notes_seen=snapshot.counts.notes_seen,
    )
    return candidates


def _record_discovery_item(
    *,
    db: LlmDb,
    job_id: int,
    item: DiscoveryItem,
    logger: logging.Logger,
) -> EligibleCandidate | None:
    note_label = item.note_key or "unknown"
    note_type_label = item.note_type or "unknown"

    if item.item_status is LlmItemStatus.INVALID_NOTE:
        message = item.error_message or "Invalid note"
        db.insert_job_item(
            job_id=job_id,
            ordinal=item.ordinal,
            deck_name=item.deck_name,
            note_key=item.note_key,
            note_type=item.note_type,
            item_status=item.item_status,
            skip_reason=None,
            error_message=message,
        )
        logger.error(
            "LLM note error for %s in '%s' (%s): %s",
            note_label,
            item.deck_name,
            note_type_label,
            message,
        )
        return None

    if item.item_status is LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS:
        db.insert_job_item(
            job_id=job_id,
            ordinal=item.ordinal,
            deck_name=item.deck_name,
            note_key=item.note_key,
            note_type=item.note_type,
            item_status=item.item_status,
            skip_reason=item.skip_reason,
        )
        logger.debug(
            "  Skipped %s in '%s' (%s): no editable non-empty fields",
            note_label,
            item.deck_name,
            note_type_label,
        )
        return None

    if (
        item.item_status is LlmItemStatus.QUEUED
        and item.payload is not None
        and item.note_type_config is not None
        and item.serialized_note is not None
    ):
        item_id = db.insert_job_item(
            job_id=job_id,
            ordinal=item.ordinal,
            deck_name=item.deck_name,
            note_key=item.payload.note_key,
            note_type=item.payload.note_type,
            item_status=item.item_status,
            skip_reason=None,
        )
        return EligibleCandidate(
            item_id=item_id,
            deck_name=item.deck_name,
            payload=item.payload,
            note_type_config=item.note_type_config,
            serialized_note=item.serialized_note,
        )

    db.insert_job_item(
        job_id=job_id,
        ordinal=item.ordinal,
        deck_name=item.deck_name,
        note_key=item.note_key,
        note_type=item.note_type,
        item_status=item.item_status,
        skip_reason=item.skip_reason,
    )
    return None
