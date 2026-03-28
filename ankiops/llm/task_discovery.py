"""Discovery selection and DB recording for task execution."""

from __future__ import annotations

import logging
from typing import Any

from ankiops.models import NoteTypeConfig

from .discovery import DiscoveryItem, DiscoverySnapshot, discover_candidates
from .llm_db import LlmDbAdapter
from .llm_models import LlmCandidateStatus, LlmFinalStatus, TaskConfig
from .task_runtime_types import EligibleCandidate


def discover_and_record_candidates(
    *,
    db: LlmDbAdapter,
    job_id: int,
    data: dict[str, Any],
    task: TaskConfig,
    note_type_configs: dict[str, NoteTypeConfig],
    resume_source_items: dict[str, int] | None,
    logger: logging.Logger,
) -> list[EligibleCandidate]:
    snapshot = discover_candidates(
        data=data,
        task=task,
        note_type_configs=note_type_configs,
    )
    selected_items = _select_snapshot_items(
        snapshot,
        resume_source_items=resume_source_items,
    )
    _log_skipped_decks(selected_items, logger=logger)

    candidates: list[EligibleCandidate] = []
    for item in selected_items:
        candidate = _record_discovery_item(
            db=db,
            job_id=job_id,
            item=item,
            resume_source_items=resume_source_items,
            logger=logger,
        )
        if candidate is not None:
            candidates.append(candidate)

    _set_discovery_counts(
        db=db,
        job_id=job_id,
        snapshot=snapshot,
        selected_items=selected_items,
        resume_source_items=resume_source_items,
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


def _resume_source_item_id(
    item: DiscoveryItem,
    *,
    resume_source_items: dict[str, int] | None,
) -> int | None:
    if resume_source_items is None:
        return None
    note_key = item.note_key
    if note_key is None:
        return None
    return resume_source_items.get(note_key)


def _record_discovery_item(
    *,
    db: LlmDbAdapter,
    job_id: int,
    item: DiscoveryItem,
    resume_source_items: dict[str, int] | None,
    logger: logging.Logger,
) -> EligibleCandidate | None:
    resume_source_item_id = _resume_source_item_id(
        item,
        resume_source_items=resume_source_items,
    )
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
            resume_source_item_id=resume_source_item_id,
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
            resume_source_item_id=resume_source_item_id,
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
            resume_source_item_id=resume_source_item_id,
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
        resume_source_item_id=resume_source_item_id,
    )
    return None


def _set_discovery_counts(
    *,
    db: LlmDbAdapter,
    job_id: int,
    snapshot: DiscoverySnapshot,
    selected_items: list[DiscoveryItem],
    resume_source_items: dict[str, int] | None,
) -> None:
    if resume_source_items is None:
        db.set_discovery_counts(
            job_id=job_id,
            decks_seen=snapshot.counts.decks_seen,
            decks_matched=snapshot.counts.decks_matched,
            notes_seen=snapshot.counts.notes_seen,
        )
        return

    deck_names = {item.deck_name for item in selected_items}
    matched_decks = {
        item.deck_name
        for item in selected_items
        if item.candidate_status is not LlmCandidateStatus.SKIPPED_DECK_SCOPE
    }
    db.set_discovery_counts(
        job_id=job_id,
        decks_seen=len(deck_names),
        decks_matched=len(matched_decks),
        notes_seen=len(selected_items),
    )


def _select_snapshot_items(
    snapshot: DiscoverySnapshot,
    *,
    resume_source_items: dict[str, int] | None,
) -> list[DiscoveryItem]:
    if resume_source_items is None:
        return list(snapshot.items)

    selected: list[DiscoveryItem] = []
    resume_note_keys = set(resume_source_items)
    for item in snapshot.items:
        note_key = item.note_key
        if note_key is None:
            continue
        if note_key in resume_note_keys:
            selected.append(item)
    return selected
