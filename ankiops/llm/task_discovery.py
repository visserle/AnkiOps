"""Discovery selection and DB recording for task execution."""

from __future__ import annotations

import logging
from typing import Any

from ankiops.models import NoteTypeConfig

from .discovery import DiscoveryItem, discover_candidates
from .llm_db import LlmDb
from .task_runtime_types import EligibleCandidate
from .task_types import LlmCandidateStatus, LlmFinalStatus, TaskConfig


def discover_and_record_candidates(
    *,
    db: LlmDb,
    job_id: int,
    data: dict[str, Any],
    task: TaskConfig,
    note_type_configs: dict[str, NoteTypeConfig],
    logger: logging.Logger,
) -> list[EligibleCandidate]:
    snapshot = discover_candidates(
        data=data,
        task=task,
        note_type_configs=note_type_configs,
    )
    _log_skipped_decks(snapshot.items, logger=logger)

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


def _log_skipped_decks(items: list[DiscoveryItem], *, logger: logging.Logger) -> None:
    skipped_deck_counts: dict[str, int] = {}
    for item in items:
        if item.candidate_status is not LlmCandidateStatus.SKIPPED_DECK_SCOPE:
            continue
        skipped_deck_counts[item.deck_name] = (
            skipped_deck_counts.get(item.deck_name, 0) + 1
        )
    for deck_name, skipped_count in skipped_deck_counts.items():
        logger.debug(
            "Skipping deck '%s' (%d notes): outside task scope",
            deck_name,
            skipped_count,
        )


def _record_discovery_item(
    *,
    db: LlmDb,
    job_id: int,
    item: DiscoveryItem,
    logger: logging.Logger,
) -> EligibleCandidate | None:
    note_label = item.note_key or "unknown"
    note_type_label = item.note_type or "unknown"

    if item.candidate_status is LlmCandidateStatus.INVALID_NOTE:
        message = item.error_message or "Invalid note"
        db.insert_job_item(
            job_id=job_id,
            ordinal=item.ordinal,
            deck_name=item.deck_name,
            note_key=item.note_key,
            note_type=item.note_type,
            candidate_status=item.candidate_status,
            skip_reason=None,
            final_status=LlmFinalStatus.NOTE_ERROR,
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

    if item.candidate_status is LlmCandidateStatus.SKIPPED_NO_EDITABLE_FIELDS:
        db.insert_job_item(
            job_id=job_id,
            ordinal=item.ordinal,
            deck_name=item.deck_name,
            note_key=item.note_key,
            note_type=item.note_type,
            candidate_status=item.candidate_status,
            skip_reason=item.skip_reason,
            final_status=LlmFinalStatus.NOT_ATTEMPTED,
        )
        logger.debug(
            "  Skipped %s in '%s' (%s): no editable non-empty fields",
            note_label,
            item.deck_name,
            note_type_label,
        )
        return None

    if (
        item.candidate_status is LlmCandidateStatus.ELIGIBLE
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
            candidate_status=item.candidate_status,
            skip_reason=None,
            final_status=LlmFinalStatus.NOT_ATTEMPTED,
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
        candidate_status=item.candidate_status,
        skip_reason=item.skip_reason,
        final_status=LlmFinalStatus.NOT_ATTEMPTED,
    )
    return None
