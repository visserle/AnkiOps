"""Execute LLM tasks and apply accepted edits."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AuthenticationError
from openai.types.responses import (
    ParsedResponse,
    ResponseOutputMessage,
    ResponseOutputRefusal,
)
from pydantic import ConfigDict, create_model

from ankiops.git import git_snapshot
from ankiops.interchange import deserialize
from ankiops.note_types import NoteType
from ankiops.notes import Note, normalize_tags

from .jobs import LlmItemStatus, LlmJobStatus, LlmJobStore, TaskRunSummary
from .planning import (
    DiscoveryItem,
    DiscoverySnapshot,
    EligibleBatch,
    EligibleCandidate,
    MaterializedTaskContext,
    NotePayload,
    build_candidate_batches,
    build_note_request_payload,
    format_deck_scope,
    materialize_task_context,
    snapshot_paths_for_task,
)
from .tasks import TaskConfig

logger = logging.getLogger(__name__)
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class LlmJobResult:
    job_id: int
    status: str
    summary: TaskRunSummary
    failed: bool
    persisted: bool


@dataclass(frozen=True)
class TaskExecutionProgress:
    job_id: int
    task_name: str
    total: int
    completed: int
    in_flight: int
    queued: int
    updated: int
    unchanged: int
    skipped: int
    errors: int
    canceled: int

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 1.0
        return min(self.completed / self.total, 1.0)

    @property
    def is_finished(self) -> bool:
        return self.completed >= self.total


class ProgressReporter(Protocol):
    def __call__(self, progress: TaskExecutionProgress) -> object: ...


@dataclass(frozen=True)
class OpenAIResult:
    parsed_response: object | None
    request_json: dict[str, Any]
    parsed_response_json: dict[str, Any] | None
    response_json: str | None
    error_message: str | None
    outcome: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    fatal: bool = False


@dataclass(frozen=True)
class _BatchProcessResult:
    statuses: tuple[LlmItemStatus, ...]


@dataclass(frozen=True)
class _CandidateApplyResult:
    candidate: EligibleCandidate
    status: LlmItemStatus
    changed_fields: list[str]
    error_message: str | None = None


@dataclass
class _ExecutionProgressState:
    job_id: int
    task_name: str
    total: int
    completed: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    errors: int = 0
    canceled: int = 0

    def record_status(self, status: LlmItemStatus, *, count: int = 1) -> None:
        if count <= 0:
            return
        if status is LlmItemStatus.SUCCEEDED_UPDATED:
            self.updated += count
        elif status is LlmItemStatus.SUCCEEDED_UNCHANGED:
            self.unchanged += count
        elif status is LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS:
            self.skipped += count
        elif status is LlmItemStatus.CANCELED:
            self.canceled += count
        elif status in {
            LlmItemStatus.INVALID_NOTE,
            LlmItemStatus.NOTE_ERROR,
            LlmItemStatus.PROVIDER_ERROR,
            LlmItemStatus.FATAL_ERROR,
        }:
            self.errors += count
        else:
            return
        self.completed += count

    def snapshot(self, *, in_flight: int) -> TaskExecutionProgress:
        queued = max(self.total - self.completed - in_flight, 0)
        return TaskExecutionProgress(
            job_id=self.job_id,
            task_name=self.task_name,
            total=self.total,
            completed=self.completed,
            in_flight=in_flight,
            queued=queued,
            updated=self.updated,
            unchanged=self.unchanged,
            skipped=self.skipped,
            errors=self.errors,
            canceled=self.canceled,
        )


class LlmTaskExecutor:
    """Run one configured LLM task against serialized notes."""

    def __init__(
        self,
        *,
        collection_root: Path,
        materialized_context: MaterializedTaskContext,
        no_auto_commit: bool,
        progress_callback: ProgressReporter | None = None,
    ) -> None:
        self.collection_root = collection_root
        self.materialized_context = materialized_context
        self.no_auto_commit = no_auto_commit
        self.progress_callback = progress_callback

    async def execute(self) -> LlmJobResult:
        task_context = self.materialized_context
        task = task_context.task
        logger.debug(
            "Starting LLM task '%s' (model=%s, collection=%s, deck_scope=%s)",
            task.name,
            task.model,
            self.collection_root,
            format_deck_scope(task),
        )
        if not self.no_auto_commit:
            logger.debug("Creating pre-LLM git snapshot")
            git_snapshot(
                self.collection_root,
                action=f"LLM task {task.name}",
                paths=snapshot_paths_for_task(
                    self.collection_root,
                    task_context,
                ),
            )
        else:
            logger.debug("Auto-commit disabled (--no-auto-commit)")

        db = LlmJobStore.open(self.collection_root)
        try:
            job_id = db.start_job(
                task_name=task.name,
                model=task.model.model,
                model_id=task.model.model_id,
            )
            progress_state = _build_progress_state(
                job_id=job_id,
                task_name=task.name,
                snapshot=task_context.discovery_snapshot,
            )
            candidates = _record_discovery_snapshot(
                db=db,
                job_id=job_id,
                snapshot=task_context.discovery_snapshot,
            )
            self._emit_progress(progress_state, in_flight=0)

            fatal_error: str | None = None
            if candidates:
                try:
                    await self._execute_candidates(
                        db=db,
                        job_id=job_id,
                        task=task,
                        candidates=candidates,
                        progress_state=progress_state,
                    )
                except RuntimeError as error:
                    fatal_error = str(error)
                    logger.error(fatal_error)

            if fatal_error is not None:
                canceled = db.mark_unfinished_items_canceled(job_id=job_id)
                progress_state.record_status(LlmItemStatus.CANCELED, count=canceled)

            self._emit_progress(progress_state, in_flight=0)
            aggregate = db.aggregate_job(job_id)
            failed = fatal_error is not None or aggregate.summary.errors > 0
            persisted = False
            if not failed:
                persisted = _persist_updates(
                    db=db,
                    job_id=job_id,
                    data=task_context.serialized_data,
                    collection_root=self.collection_root,
                )

            db.finalize_job(
                job_id=job_id,
                status=LlmJobStatus.FAILED if failed else LlmJobStatus.COMPLETED,
                persisted=persisted,
                fatal_error=fatal_error if failed else None,
            )
            aggregate = db.aggregate_job(job_id)
            logger.info(aggregate.summary.format())
            logger.debug("Usage: %s", aggregate.summary.format_usage())
            logger.debug("Cost: %s", aggregate.summary.format_cost())
            return LlmJobResult(
                job_id=job_id,
                status=aggregate.status.value,
                summary=aggregate.summary,
                failed=aggregate.failed,
                persisted=aggregate.persisted,
            )
        finally:
            db.close()

    async def _execute_candidates(
        self,
        *,
        db: LlmJobStore,
        job_id: int,
        task: TaskConfig,
        candidates: list[EligibleCandidate],
        progress_state: _ExecutionProgressState,
    ) -> None:
        api_key = _resolve_api_key(task.model.api_key)
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=task.model.base_url,
            timeout=60.0,
            max_retries=0,
        )
        try:
            batches = build_candidate_batches(
                candidates,
                max_notes_per_request=task.request.max_notes_per_request,
            )
            in_flight_limit = min(max(task.model.concurrency, 1), len(batches))
            next_index = 0
            in_flight: dict[asyncio.Task[_BatchProcessResult], EligibleBatch] = {}

            def in_flight_notes() -> int:
                return sum(batch.note_count for batch in in_flight.values())

            def start(batch: EligibleBatch) -> None:
                task_handle = asyncio.create_task(
                    self._process_batch(
                        db=db,
                        job_id=job_id,
                        task=task,
                        client=client,
                        batch=batch,
                    )
                )
                in_flight[task_handle] = batch

            while next_index < in_flight_limit:
                start(batches[next_index])
                next_index += 1

            first_fatal: RuntimeError | None = None
            while in_flight:
                done, _pending = await asyncio.wait(
                    set(in_flight),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for completed in done:
                    batch = in_flight.pop(completed)
                    try:
                        result = await completed
                        for status in result.statuses:
                            progress_state.record_status(status)
                    except RuntimeError as error:
                        progress_state.record_status(
                            LlmItemStatus.FATAL_ERROR,
                            count=batch.note_count,
                        )
                        if first_fatal is None:
                            first_fatal = error
                    self._emit_progress(
                        progress_state,
                        in_flight=in_flight_notes(),
                    )

                if first_fatal is not None:
                    for pending_task in in_flight:
                        pending_task.cancel()
                    await asyncio.gather(*in_flight, return_exceptions=True)
                    raise first_fatal

                while next_index < len(batches) and len(in_flight) < in_flight_limit:
                    start(batches[next_index])
                    next_index += 1
        finally:
            await client.close()

    async def _process_batch(
        self,
        *,
        db: LlmJobStore,
        job_id: int,
        task: TaskConfig,
        client: AsyncOpenAI,
        batch: EligibleBatch,
    ) -> _BatchProcessResult:
        result = await _call_openai(
            client=client,
            task=task,
            batch=batch,
        )
        if result.parsed_response is None:
            status = (
                LlmItemStatus.FATAL_ERROR
                if result.fatal
                else LlmItemStatus.PROVIDER_ERROR
            )
            with db.write_tx():
                db.insert_request(
                    job_id=job_id,
                    item_ids=batch.item_ids,
                    outcome=result.outcome,
                    request_json=result.request_json,
                    parsed_response_json=result.parsed_response_json,
                    response_json=result.response_json,
                    error_message=result.error_message,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )
                for candidate in batch.candidates:
                    db.update_job_item_status(
                        item_id=candidate.item_id,
                        item_status=status,
                        error_message=result.error_message,
                    )
            if result.fatal:
                raise RuntimeError(result.error_message or "Fatal OpenAI error")
            for candidate in batch.candidates:
                _log_note_error(
                    candidate,
                    result.error_message or "OpenAI request failed",
                )
            return _BatchProcessResult(statuses=tuple(status for _ in batch.candidates))

        try:
            apply_results = _apply_batch_parsed_response(
                parsed_response=result.parsed_response,
                batch=batch,
            )
        except ValueError as error:
            message = str(error)
            with db.write_tx():
                db.insert_request(
                    job_id=job_id,
                    item_ids=batch.item_ids,
                    outcome="validation_error",
                    request_json=result.request_json,
                    parsed_response_json=result.parsed_response_json,
                    response_json=result.response_json,
                    error_message=message,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )
                for candidate in batch.candidates:
                    db.update_job_item_status(
                        item_id=candidate.item_id,
                        item_status=LlmItemStatus.NOTE_ERROR,
                        error_message=message,
                    )
            for candidate in batch.candidates:
                _log_note_error(candidate, message)
            return _BatchProcessResult(
                statuses=tuple(LlmItemStatus.NOTE_ERROR for _ in batch.candidates)
            )

        errors = [
            f"{item.candidate.payload.note_key}: {item.error_message}"
            for item in apply_results
            if item.error_message
        ]
        outcome = "validation_error" if errors else "success"
        request_error = "; ".join(errors) if errors else None
        with db.write_tx():
            db.insert_request(
                job_id=job_id,
                item_ids=batch.item_ids,
                outcome=outcome,
                request_json=result.request_json,
                parsed_response_json=result.parsed_response_json,
                response_json=result.response_json,
                error_message=request_error,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            for item in apply_results:
                db.update_job_item_status(
                    item_id=item.candidate.item_id,
                    item_status=item.status,
                    error_message=item.error_message,
                    changed_fields=item.changed_fields,
                )
        for item in apply_results:
            if item.error_message is not None:
                _log_note_error(item.candidate, item.error_message)
            elif item.changed_fields:
                logger.debug(
                    "  Updated %s in '%s' (%s): changed=%s",
                    item.candidate.payload.note_key,
                    item.candidate.deck_name,
                    item.candidate.payload.note_type,
                    ",".join(item.changed_fields),
                )
        return _BatchProcessResult(
            statuses=tuple(item.status for item in apply_results)
        )

    def _emit_progress(
        self,
        progress_state: _ExecutionProgressState,
        *,
        in_flight: int,
    ) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(progress_state.snapshot(in_flight=in_flight))
        except Exception:
            logger.debug("Progress callback raised an exception", exc_info=True)


async def run_task_async(
    *,
    collection_root: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
    no_auto_commit: bool = False,
    progress_callback: ProgressReporter | None = None,
) -> LlmJobResult:
    context = materialize_task_context(
        collection_root=collection_root,
        task_name=task_name,
        model_override=model_override,
        deck_override=deck_override,
    )
    return await LlmTaskExecutor(
        collection_root=collection_root,
        materialized_context=context,
        no_auto_commit=no_auto_commit,
        progress_callback=progress_callback,
    ).execute()


def run_task(
    *,
    collection_root: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
    no_auto_commit: bool = False,
    progress_callback: ProgressReporter | None = None,
) -> LlmJobResult:
    return asyncio.run(
        run_task_async(
            collection_root=collection_root,
            task_name=task_name,
            model_override=model_override,
            deck_override=deck_override,
            no_auto_commit=no_auto_commit,
            progress_callback=progress_callback,
        )
    )


def _record_discovery_snapshot(
    *,
    db: LlmJobStore,
    job_id: int,
    snapshot: DiscoverySnapshot,
) -> list[EligibleCandidate]:
    candidates: list[EligibleCandidate] = []
    for item in snapshot.items:
        candidate = _record_discovery_item(db=db, job_id=job_id, item=item)
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
    db: LlmJobStore,
    job_id: int,
    item: DiscoveryItem,
) -> EligibleCandidate | None:
    if item.item_status is LlmItemStatus.INVALID_NOTE:
        db.insert_job_item(
            job_id=job_id,
            ordinal=item.ordinal,
            source=item.source,
            deck_name=item.deck_name,
            note_key=item.note_key,
            note_type=item.note_type,
            item_status=item.item_status,
            skip_reason=None,
            error_message=item.error_message or "Invalid note",
        )
        logger.error(
            "LLM note error for %s in '%s' (%s): %s",
            item.note_key or "unknown",
            item.deck_name,
            item.note_type or "unknown",
            item.error_message or "Invalid note",
        )
        return None

    if item.item_status is LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS:
        db.insert_job_item(
            job_id=job_id,
            ordinal=item.ordinal,
            source=item.source,
            deck_name=item.deck_name,
            note_key=item.note_key,
            note_type=item.note_type,
            item_status=item.item_status,
            skip_reason=item.skip_reason,
        )
        return None

    if (
        item.payload is None
        or item.note_type_config is None
        or item.serialized_note is None
    ):
        return None

    item_id = db.insert_job_item(
        job_id=job_id,
        ordinal=item.ordinal,
        source=item.source,
        deck_name=item.deck_name,
        note_key=item.payload.note_key,
        note_type=item.payload.note_type,
        item_status=LlmItemStatus.QUEUED,
        skip_reason=None,
    )
    return EligibleCandidate(
        item_id=item_id,
        source=item.source,
        deck_name=item.deck_name,
        payload=item.payload,
        note_type_config=item.note_type_config,
        serialized_note=item.serialized_note,
    )


async def _call_openai(
    *,
    client: AsyncOpenAI,
    task: TaskConfig,
    batch: EligibleBatch,
) -> OpenAIResult:
    response_model = build_response_model(
        note_type=batch.note_type,
        payloads=batch.payloads,
    )
    instructions, user_input = _build_request_content(task=task, batch=batch)
    request_kwargs: dict[str, Any] = {
        "model": task.model.model_id,
        "instructions": instructions,
        "input": user_input,
        "text_format": response_model,
    }
    if task.request.temperature is not None:
        request_kwargs["temperature"] = task.request.temperature
    if task.request.reasoning is not None:
        request_kwargs["reasoning"] = {"effort": task.request.reasoning}
    request_json = dict(request_kwargs)
    request_json["text_format"] = response_model.__name__
    started_at = time.monotonic()
    try:
        response: ParsedResponse[object] = await client.responses.parse(
            **request_kwargs
        )
    except AuthenticationError as error:
        return _openai_error_result(
            request_json=request_json,
            error_message=f"OpenAI authentication failed: {error}",
            outcome="fatal_error",
            started_at=started_at,
            fatal=True,
        )
    except APIConnectionError as error:
        return _openai_error_result(
            request_json=request_json,
            error_message=f"OpenAI connection error: {error}",
            outcome="provider_error",
            started_at=started_at,
        )
    except APIStatusError as error:
        status_code = getattr(error, "status_code", "unknown")
        return _openai_error_result(
            request_json=request_json,
            error_message=f"OpenAI returned HTTP {status_code}: {error}",
            outcome="provider_error",
            started_at=started_at,
        )
    except Exception as error:
        return _openai_error_result(
            request_json=request_json,
            error_message=f"Unexpected OpenAI error: {error}",
            outcome="fatal_error",
            started_at=started_at,
            fatal=True,
        )

    latency_ms = round((time.monotonic() - started_at) * 1000)
    refusal = _extract_refusal_text(response)
    response_json = _response_to_json(response)
    if refusal is not None:
        return OpenAIResult(
            parsed_response=None,
            request_json=request_json,
            parsed_response_json=None,
            response_json=response_json,
            error_message=refusal,
            outcome="refusal",
            latency_ms=latency_ms,
            input_tokens=_usage_value(response, "input_tokens"),
            output_tokens=_usage_value(response, "output_tokens"),
        )

    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        return OpenAIResult(
            parsed_response=None,
            request_json=request_json,
            parsed_response_json=None,
            response_json=response_json,
            error_message="OpenAI response did not include parsed structured output",
            outcome="provider_error",
            latency_ms=latency_ms,
            input_tokens=_usage_value(response, "input_tokens"),
            output_tokens=_usage_value(response, "output_tokens"),
        )

    return OpenAIResult(
        parsed_response=parsed,
        request_json=request_json,
        parsed_response_json=parsed_response_json(parsed),
        response_json=response_json,
        error_message=None,
        outcome="success",
        latency_ms=latency_ms,
        input_tokens=_usage_value(response, "input_tokens"),
        output_tokens=_usage_value(response, "output_tokens"),
    )


def _build_request_content(
    *,
    task: TaskConfig,
    batch: EligibleBatch,
) -> tuple[str, str]:
    payload = {
        "user_prompt": task.user_prompt,
        "note_type": batch.note_type,
        "notes": [
            build_note_request_payload(candidate.payload)
            for candidate in batch.candidates
        ],
    }
    return (
        task.system_prompt.strip(),
        json.dumps(payload, ensure_ascii=False),
    )


def _openai_error_result(
    *,
    request_json: dict[str, Any],
    error_message: str,
    outcome: str,
    started_at: float,
    fatal: bool = False,
) -> OpenAIResult:
    return OpenAIResult(
        parsed_response=None,
        request_json=request_json,
        parsed_response_json=None,
        response_json=None,
        error_message=error_message,
        outcome=outcome,
        latency_ms=round((time.monotonic() - started_at) * 1000),
        input_tokens=0,
        output_tokens=0,
        fatal=fatal,
    )


def _apply_batch_parsed_response(
    *,
    parsed_response: object,
    batch: EligibleBatch,
) -> list[_CandidateApplyResult]:
    updates = parsed_updates(parsed_response)
    tag_updates = parsed_tag_updates(parsed_response)
    candidates_by_note_key = {
        candidate.payload.note_key: candidate for candidate in batch.candidates
    }
    unexpected_note_keys = sorted(
        {
            note_key
            for note_key, _field_name, _value in updates
            if note_key not in candidates_by_note_key
        }
        | {
            note_key
            for note_key, _tags in tag_updates
            if note_key not in candidates_by_note_key
        }
    )
    if unexpected_note_keys:
        raise ValueError(
            f"Model returned update for unexpected note_key '{unexpected_note_keys[0]}'"
        )

    updates_by_note_key: dict[str, list[tuple[str, str, str]]] = {
        note_key: [] for note_key in candidates_by_note_key
    }
    tag_updates_by_note_key: dict[str, list[tuple[str, list[str]]]] = {
        note_key: [] for note_key in candidates_by_note_key
    }
    for update in updates:
        updates_by_note_key[update[0]].append(update)
    for tag_update in tag_updates:
        tag_updates_by_note_key[tag_update[0]].append(tag_update)

    results: list[_CandidateApplyResult] = []
    for candidate in batch.candidates:
        try:
            changed_fields = _apply_candidate_updates(
                candidate=candidate,
                updates=updates_by_note_key[candidate.payload.note_key],
                tag_updates=tag_updates_by_note_key[candidate.payload.note_key],
            )
        except ValueError as error:
            results.append(
                _CandidateApplyResult(
                    candidate=candidate,
                    status=LlmItemStatus.NOTE_ERROR,
                    changed_fields=[],
                    error_message=str(error),
                )
            )
        else:
            results.append(
                _CandidateApplyResult(
                    candidate=candidate,
                    status=(
                        LlmItemStatus.SUCCEEDED_UPDATED
                        if changed_fields
                        else LlmItemStatus.SUCCEEDED_UNCHANGED
                    ),
                    changed_fields=changed_fields,
                )
            )
    return results


def _apply_candidate_updates(
    *,
    candidate: EligibleCandidate,
    updates: list[tuple[str, str, str]],
    tag_updates: list[tuple[str, list[str]]] | None = None,
) -> list[str]:
    seen_fields: set[str] = set()
    edits: dict[str, str] = {}
    for _note_key, field_name, value in updates:
        if field_name in seen_fields:
            raise ValueError(f"Model returned duplicate update for '{field_name}'")
        seen_fields.add(field_name)
        if field_name not in candidate.payload.editable_fields:
            if field_name in candidate.payload.read_only_fields:
                raise ValueError(
                    f"Model attempted to update read-only field '{field_name}'"
                )
            raise ValueError(
                f"Model attempted to update hidden or unknown field '{field_name}'"
            )
        edits[field_name] = value

    tag_edit: tuple[str, ...] | None = None
    if tag_updates:
        if len(tag_updates) > 1:
            raise ValueError("Model returned duplicate update for 'tags'")
        if candidate.payload.editable_tags is None:
            if candidate.payload.read_only_tags is not None:
                raise ValueError("Model attempted to update read-only tags")
            raise ValueError("Model attempted to update hidden tags")
        _note_key, raw_tags = tag_updates[0]
        tag_edit = normalize_tags(raw_tags)

    raw_fields = candidate.serialized_note.get("fields")
    if not isinstance(raw_fields, dict):
        raise ValueError("Serialized note fields must be a mapping")

    next_fields = dict(raw_fields)
    next_tags = normalize_tags(candidate.serialized_note.get("tags", ()))
    changed_fields: list[str] = []
    for field_name, value in edits.items():
        if next_fields.get(field_name, "") != value:
            next_fields[field_name] = value
            changed_fields.append(field_name)
    if tag_edit is not None and next_tags != tag_edit:
        next_tags = tag_edit
        changed_fields.append("tags")

    if not changed_fields:
        return []

    note = Note(
        note_key=candidate.payload.note_key,
        note_type=candidate.payload.note_type,
        fields=next_fields,
        tags=next_tags,
    )
    errors = [*_validate_cloze_text_fields(note, candidate.note_type_config)]
    errors.extend(note.validate(candidate.note_type_config))
    if errors:
        raise ValueError("; ".join(errors))

    candidate.serialized_note["fields"] = next_fields
    candidate.serialized_note["tags"] = list(next_tags)
    return changed_fields


def _validate_cloze_text_fields(
    note: Note,
    config: NoteType,
) -> list[str]:
    if not config.is_cloze:
        return []
    errors: list[str] = []
    field_names = _cloze_template_field_names(config)
    for field_name in sorted(field_names):
        value = note.fields.get(field_name, "")
        if "{{c" not in value:
            errors.append(
                f"{note.note_type} field '{field_name}' must contain cloze syntax "
                "(e.g. {{c1::answer}})"
            )
    return errors


def _cloze_template_field_names(config: NoteType) -> set[str]:
    """Find the actualname of the cloze field used in the cloze template."""
    cloze_template_field_pattern = re.compile(
        r"\{\{\s*(?:[A-Za-z]+:)*cloze:([^}]+?)\s*\}\}",
        re.IGNORECASE,
    )
    field_names: set[str] = set()
    for template in config.templates:
        for template_value in (template.front, template.back):
            for match in cloze_template_field_pattern.finditer(template_value):
                field_name = match.group(1).strip()
                if field_name:
                    field_names.add(field_name)
    return field_names


def _persist_updates(
    *,
    db: LlmJobStore,
    job_id: int,
    data: dict[str, Any],
    collection_root: Path,
) -> bool:
    aggregate = db.aggregate_job(job_id)
    if aggregate.summary.updated <= 0:
        return False

    deserialize(
        data,
        collection_root=collection_root,
        overwrite=True,
        quiet=True,
    )
    db.set_applied_for_updated_items(job_id=job_id)
    deck_count = len([deck for deck in data.get("decks", []) if isinstance(deck, dict)])
    logger.info(
        "Persisted %d updated note(s) across %d deck file(s)",
        aggregate.summary.updated,
        deck_count,
    )
    return True


def _build_progress_state(
    *,
    job_id: int,
    task_name: str,
    snapshot: DiscoverySnapshot,
) -> _ExecutionProgressState:
    state = _ExecutionProgressState(
        job_id=job_id,
        task_name=task_name,
        total=len(snapshot.items),
    )
    for item in snapshot.items:
        state.record_status(item.item_status)
    return state


def _resolve_api_key(configured_value: str) -> str:
    if not configured_value.startswith("$"):
        return configured_value
    env_name = configured_value[1:].strip()
    if not env_name:
        raise RuntimeError("Model api_key env reference must include a variable")
    resolved = os.environ.get(env_name)
    if not resolved:
        raise RuntimeError(f"Required environment variable '{env_name}' is not set")
    return resolved


def _usage_value(response: ParsedResponse[object], name: str) -> int:
    usage = getattr(response, "usage", None)
    value = getattr(usage, name, 0)
    return value if isinstance(value, int) else 0


def _response_to_json(response: ParsedResponse[object]) -> str | None:
    try:
        return response.model_dump_json(warnings=False)
    except Exception:
        return None


def _extract_refusal_text(response: ParsedResponse[object]) -> str | None:
    refusal_parts = [
        part.refusal
        for item in response.output
        if isinstance(item, ResponseOutputMessage)
        for part in item.content
        if isinstance(part, ResponseOutputRefusal) and part.refusal
    ]
    return "\n".join(refusal_parts) if refusal_parts else None


def _log_note_error(candidate: EligibleCandidate, message: str) -> None:
    logger.error(
        "LLM note error for %s in '%s' (%s): %s",
        candidate.payload.note_key,
        candidate.deck_name,
        candidate.payload.note_type,
        message,
    )


def build_response_model(
    *,
    note_type: str,
    payloads: list[NotePayload],
) -> type[Any]:
    """Build a strict response model for one note type and request batch."""
    if not payloads:
        raise ValueError("response model requires at least one payload")

    note_keys = sorted({payload.note_key for payload in payloads})
    editable_fields = sorted(
        {field_name for payload in payloads for field_name in payload.editable_fields}
    )
    has_editable_tags = any(payload.editable_tags is not None for payload in payloads)
    if not editable_fields and not has_editable_tags:
        raise ValueError(
            "response model requires at least one editable field or editable tags"
        )

    suffix = _safe_type_name(note_type)
    response_fields: dict[str, Any] = {}
    if editable_fields:
        update_model = create_model(
            f"{suffix}Update",
            __config__=ConfigDict(extra="forbid"),
            note_key=(_literal(note_keys), ...),
            field=(_literal(editable_fields), ...),
            value=(str, ...),
        )
        update_model_type: Any = update_model
        response_fields["updates"] = (list[update_model_type], ...)
    if has_editable_tags:
        tag_update_model = create_model(
            f"{suffix}TagUpdate",
            __config__=ConfigDict(extra="forbid"),
            note_key=(_literal(note_keys), ...),
            tags=(list[str], ...),
        )
        tag_update_model_type: Any = tag_update_model
        response_fields["tag_updates"] = (list[tag_update_model_type], ...)
    return cast(
        type[Any],
        create_model(
            f"{suffix}Response",
            __config__=ConfigDict(extra="forbid"),
            **response_fields,
        ),
    )


def parsed_updates(parsed_response: object) -> list[tuple[str, str, str]]:
    """Return ``(note_key, field, value)`` tuples from a parsed Pydantic object."""
    raw_updates = getattr(parsed_response, "updates", [])
    if not isinstance(raw_updates, list):
        raise ValueError("Parsed response is missing updates list")

    updates: list[tuple[str, str, str]] = []
    for raw_update in raw_updates:
        note_key = getattr(raw_update, "note_key", None)
        field = getattr(raw_update, "field", None)
        value = getattr(raw_update, "value", None)
        if not isinstance(note_key, str):
            raise ValueError("Parsed update note_key must be a string")
        if not isinstance(field, str):
            raise ValueError("Parsed update field must be a string")
        if not isinstance(value, str):
            raise ValueError("Parsed update value must be a string")
        updates.append((note_key, field, value))
    return updates


def parsed_tag_updates(parsed_response: object) -> list[tuple[str, list[str]]]:
    """Return ``(note_key, tags)`` tuples from a parsed Pydantic object."""
    raw_updates = getattr(parsed_response, "tag_updates", [])
    if not isinstance(raw_updates, list):
        raise ValueError("Parsed response is missing tag_updates list")

    updates: list[tuple[str, list[str]]] = []
    for raw_update in raw_updates:
        note_key = getattr(raw_update, "note_key", None)
        tags = getattr(raw_update, "tags", None)
        if not isinstance(note_key, str):
            raise ValueError("Parsed tag update note_key must be a string")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError("Parsed tag update tags must be a list of strings")
        updates.append((note_key, tags))
    return updates


def parsed_response_json(parsed_response: object) -> dict[str, object] | None:
    model_dump = getattr(parsed_response, "model_dump", None)
    if not callable(model_dump):
        return None
    value = model_dump(mode="json")
    return value if isinstance(value, dict) else None


def _literal(values: list[str]) -> object:
    if not values:
        raise ValueError("Literal requires at least one value")
    return Literal.__getitem__(tuple(values))


def _safe_type_name(value: str) -> str:
    safe = _SAFE_NAME_PATTERN.sub("_", value).strip("_")
    if not safe:
        return "Note"
    if safe[0].isdigit():
        safe = f"Note_{safe}"
    return safe
