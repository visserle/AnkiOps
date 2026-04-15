"""Execution of collection-local LLM tasks."""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator

from ankiops.collection_serializer import (
    deserialize_collection_data,
    serialize_collection,
)
from ankiops.config import NOTE_TYPES_DIR
from ankiops.fs import FileSystemAdapter
from ankiops.git import git_snapshot
from ankiops.models import Note, NoteTypeConfig

from .config_loader import is_task_config_file, load_llm_task_catalog
from .discovery import DiscoverySnapshot, discover_candidates
from .llm_db import LlmDb, LlmJobDetail, LlmJobListItem
from .llm_errors import LlmFatalError, LlmNoteError, LlmNoteErrorCategory
from .provider_client import ProviderClient
from .task_attempts import AttemptRecorder
from .task_discovery import record_discovery_snapshot
from .task_options import (
    apply_deck_override,
    format_deck_scope,
    format_request_defaults,
    resolve_model,
    resolve_serializer_scope,
)
from .task_planner import build_task_plan_result
from .task_runtime_types import EligibleCandidate
from .task_types import (
    LlmItemStatus,
    LlmJobResult,
    LlmJobStatus,
    TaskConfig,
    TaskPlanResult,
    TaskRunSummary,
)

logger = logging.getLogger(__name__)

ProviderClientFactory = Callable[[TaskConfig], Any]
CollectionSerializer = Callable[..., dict[str, Any]]
CollectionDeserializer = Callable[..., None]
Snapshotter = Callable[[Path, str], bool]


@dataclass(frozen=True)
class MaterializedTaskContext:
    task: TaskConfig
    note_type_configs: dict[str, NoteTypeConfig]
    serialized_data: dict[str, Any]
    discovery_snapshot: DiscoverySnapshot


def _default_provider_client_factory(task: TaskConfig) -> ProviderClient:
    return ProviderClient(task)


@contextmanager
def _open_llm_db(collection_dir: Path) -> Iterator[LlmDb]:
    db = LlmDb.open(collection_dir)
    try:
        yield db
    finally:
        db.close()


def _apply_note_update(
    *,
    serialized_note: dict[str, Any],
    payload,
    edits: dict[str, str],
    note_type_config: NoteTypeConfig,
) -> list[str]:
    if payload.note_key != serialized_note.get("note_key"):
        raise LlmNoteError("Update note_key did not match serialized note")

    raw_fields = serialized_note.get("fields")
    if not isinstance(raw_fields, dict):
        raise LlmNoteError("Serialized note fields must be a mapping")

    next_fields = dict(raw_fields)
    changed_fields: list[str] = []
    for field_name, value in edits.items():
        if field_name not in payload.editable_fields:
            if field_name in payload.read_only_fields:
                raise LlmNoteError(
                    f"Model attempted to update read-only field '{field_name}'"
                )
            raise LlmNoteError(
                f"Model attempted to update hidden or unknown field '{field_name}'"
            )
        if next_fields.get(field_name) != value:
            next_fields[field_name] = value
            changed_fields.append(field_name)

    if not changed_fields:
        return []

    note = Note(
        note_key=payload.note_key,
        note_type=payload.note_type,
        fields=next_fields,
    )
    errors = note.validate(note_type_config)
    if errors:
        raise LlmNoteError("; ".join(errors))

    serialized_note["fields"] = next_fields
    return changed_fields


def _load_task(
    *,
    collection_dir: Path,
    task_name: str,
) -> tuple[TaskConfig, dict[str, NoteTypeConfig]]:
    fs = FileSystemAdapter()
    note_type_configs = fs.load_note_type_configs(collection_dir / NOTE_TYPES_DIR)
    catalog = load_llm_task_catalog(
        collection_dir,
        note_type_configs=note_type_configs,
    )
    task = catalog.tasks_by_name.get(task_name)
    if task is None:
        task_errors = [
            message
            for path, message in catalog.errors.items()
            if _is_task_file_for_name(path, task_name)
        ]
        if task_errors:
            joined_errors = "\n".join(task_errors)
            raise ValueError(f"Invalid LLM task configuration:\n{joined_errors}")

        shared_errors = [
            message
            for path, message in catalog.errors.items()
            if not _is_task_file(path)
        ]
        if shared_errors:
            joined_errors = "\n".join(shared_errors)
            raise ValueError(f"Invalid LLM task configuration:\n{joined_errors}")

        raise ValueError(f"Unknown task '{task_name}'")

    config_by_name = {config.name: config for config in note_type_configs}
    return task, config_by_name


def _materialize_task_context(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None,
    deck_override: str | None,
    serialize_collection_fn: CollectionSerializer,
) -> MaterializedTaskContext:
    task, note_type_configs = _load_task(
        collection_dir=collection_dir,
        task_name=task_name,
    )
    task = apply_deck_override(task, deck_override)
    model = resolve_model(task, model_override, collection_dir=collection_dir)
    task = replace(task, model=model)

    deck, no_subdecks = resolve_serializer_scope(task)
    serialized_data = serialize_collection_fn(
        collection_dir,
        deck=deck,
        no_subdecks=no_subdecks,
        note_types_dir=collection_dir / NOTE_TYPES_DIR,
    )
    discovery_snapshot = discover_candidates(
        data=serialized_data,
        task=task,
        note_type_configs=note_type_configs,
    )
    return MaterializedTaskContext(
        task=task,
        note_type_configs=note_type_configs,
        serialized_data=serialized_data,
        discovery_snapshot=discovery_snapshot,
    )


def _is_task_file(path: str) -> bool:
    path_obj = Path(path)
    return path_obj.parent.name == "llm" and is_task_config_file(path_obj)


def _is_task_file_for_name(path: str, task_name: str) -> bool:
    path_obj = Path(path)
    return _is_task_file(path) and path_obj.stem == task_name


def _unexpected_online_execution_error(error: Exception) -> LlmFatalError:
    message = str(error).strip() or error.__class__.__name__
    return LlmFatalError(f"Unexpected online execution error: {message}")


class LlmTaskExecutor:
    """Standalone async LLM runtime pipeline with DB-backed state."""

    def __init__(
        self,
        *,
        collection_dir: Path,
        materialized_context: MaterializedTaskContext,
        no_auto_commit: bool,
        provider_client_factory: ProviderClientFactory | None = None,
        deserialize_collection_fn: CollectionDeserializer | None = None,
        snapshot_fn: Snapshotter | None = None,
    ) -> None:
        self.collection_dir = collection_dir
        self.materialized_context = materialized_context
        self.no_auto_commit = no_auto_commit
        self.provider_client_factory = (
            provider_client_factory or _default_provider_client_factory
        )
        self.deserialize_collection_fn = (
            deserialize_collection_fn or deserialize_collection_data
        )
        self.snapshot_fn = snapshot_fn or git_snapshot

    async def execute(self) -> LlmJobResult:
        task_context = self.materialized_context
        task = task_context.task
        model = task.model
        db = LlmDb.open(self.collection_dir)
        try:
            job_id = db.start_job(
                task_name=task.name,
                model=model.model,
                model_id=model.model_id,
            )

            logger.debug(
                "Starting LLM task '%s' (model=%s, model_id=%s, collection=%s, "
                "deck_scope=%s)",
                task.name,
                model,
                model.model_id,
                self.collection_dir,
                format_deck_scope(task),
            )
            logger.debug("LLM request defaults: %s", format_request_defaults(task))
            logger.debug(
                "LLM serializer scope: %s",
                format_deck_scope(task),
            )

            if not self.no_auto_commit:
                logger.debug("Creating pre-LLM git snapshot")
                self.snapshot_fn(self.collection_dir, f"llm:{task.name}")
            else:
                logger.debug("Auto-commit disabled (--no-auto-commit)")

            fatal_error: str | None = None
            provider_client = None
            attempt_recorder = AttemptRecorder(db=db, provider=model.provider)
            try:
                provider_client = self.provider_client_factory(task)
                candidates = record_discovery_snapshot(
                    db=db,
                    job_id=job_id,
                    snapshot=task_context.discovery_snapshot,
                    logger=logger,
                )
                await self._execute_candidates_online(
                    db=db,
                    task=task,
                    model_id=model.model_id,
                    provider_client=provider_client,
                    candidates=candidates,
                    attempt_recorder=attempt_recorder,
                )
            except LlmFatalError as error:
                fatal_error = str(error)
                logger.error(fatal_error)
            except Exception as error:
                fatal_error = str(_unexpected_online_execution_error(error))
                logger.error(fatal_error)
            finally:
                if provider_client is not None:
                    await provider_client.close()

            if fatal_error is not None:
                canceled = db.mark_unfinished_items_canceled(job_id=job_id)
                if canceled:
                    logger.warning(
                        "Marked %d pending item(s) as canceled due to fatal error",
                        canceled,
                    )

            persisted = self._apply_updates(
                db=db,
                job_id=job_id,
                data=task_context.serialized_data,
            )

            aggregate = db.aggregate_job(job_id)
            failed = fatal_error is not None or aggregate.summary.errors > 0
            db.finalize_job(
                job_id=job_id,
                status=LlmJobStatus.FAILED if failed else LlmJobStatus.COMPLETED,
                persisted=persisted,
                fatal_error=fatal_error if failed else None,
            )
            aggregate = db.aggregate_job(job_id)

            self._log_summary(aggregate.summary)
            return LlmJobResult(
                job_id=job_id,
                status=aggregate.status.value,
                summary=aggregate.summary,
                failed=aggregate.failed,
                persisted=aggregate.persisted,
            )
        finally:
            db.close()

    @staticmethod
    def _log_summary(summary: TaskRunSummary) -> None:
        logger.info(summary.format())
        logger.debug("Usage: %s", summary.format_usage())
        logger.debug("Cost: %s", summary.format_cost())

    async def _execute_candidates_online(
        self,
        *,
        db: LlmDb,
        task: TaskConfig,
        model_id: str,
        provider_client,
        candidates: list[EligibleCandidate],
        attempt_recorder: AttemptRecorder,
    ) -> None:
        if not candidates:
            return

        in_flight_limit = min(max(task.concurrency, 1), len(candidates))
        next_index = 0
        first_fatal_error: LlmFatalError | None = None

        def _start_candidate(candidate: EligibleCandidate) -> asyncio.Task[None]:
            return asyncio.create_task(
                self._process_online_candidate(
                    db=db,
                    task=task,
                    model_id=model_id,
                    provider_client=provider_client,
                    candidate=candidate,
                    attempt_recorder=attempt_recorder,
                )
            )

        in_flight: set[asyncio.Task[None]] = set()
        while next_index < in_flight_limit:
            in_flight.add(_start_candidate(candidates[next_index]))
            next_index += 1

        while in_flight:
            done, pending = await asyncio.wait(
                in_flight,
                return_when=asyncio.FIRST_COMPLETED,
            )
            in_flight = set(pending)

            for completed in done:
                try:
                    await completed
                except LlmFatalError as error:
                    if first_fatal_error is None:
                        first_fatal_error = error
                except Exception as error:
                    if first_fatal_error is None:
                        first_fatal_error = _unexpected_online_execution_error(error)

            if first_fatal_error is not None:
                for pending_task in in_flight:
                    pending_task.cancel()
                await asyncio.gather(*in_flight, return_exceptions=True)
                raise first_fatal_error

            while next_index < len(candidates) and len(in_flight) < in_flight_limit:
                in_flight.add(_start_candidate(candidates[next_index]))
                next_index += 1

    async def _process_online_candidate(
        self,
        *,
        db: LlmDb,
        task: TaskConfig,
        model_id: str,
        provider_client,
        candidate: EligibleCandidate,
        attempt_recorder: AttemptRecorder,
    ) -> None:
        prepared_request = provider_client.prepare_attempt_request(
            note_payload=candidate.payload,
            task_prompt=task.prompt,
            request_options=task.request,
            model_id=model_id,
        )
        outcome = None

        try:
            outcome = await provider_client.generate_update(
                prepared_request=prepared_request,
                request_options=task.request,
            )
            update = outcome.update
            if update.note_key != candidate.payload.note_key:
                raise LlmNoteError("Model returned mismatched note_key")

            changed_fields = _apply_note_update(
                serialized_note=candidate.serialized_note,
                payload=candidate.payload,
                edits=update.edits,
                note_type_config=candidate.note_type_config,
            )
        except LlmFatalError as error:
            with db.write_tx():
                attempt_recorder.record_error(
                    candidate=candidate,
                    prepared_request=prepared_request,
                    outcome=outcome,
                    error=error,
                    item_status=LlmItemStatus.FATAL_ERROR,
                )
            raise
        except LlmNoteError as error:
            provider_error = error.category is LlmNoteErrorCategory.PROVIDER
            with db.write_tx():
                attempt_recorder.record_error(
                    candidate=candidate,
                    prepared_request=prepared_request,
                    outcome=outcome,
                    error=error,
                    item_status=(
                        LlmItemStatus.PROVIDER_ERROR
                        if provider_error
                        else LlmItemStatus.NOTE_ERROR
                    ),
                )
            logger.error(
                "LLM note error for %s in '%s' (%s): %s",
                candidate.payload.note_key,
                candidate.deck_name,
                candidate.payload.note_type,
                error,
            )
            return
        except Exception as error:
            logger.exception(
                "Unexpected online execution error for %s in '%s' (%s)",
                candidate.payload.note_key,
                candidate.deck_name,
                candidate.payload.note_type,
            )
            fatal_error = _unexpected_online_execution_error(error)
            with db.write_tx():
                attempt_recorder.record_error(
                    candidate=candidate,
                    prepared_request=prepared_request,
                    outcome=outcome,
                    error=fatal_error,
                    item_status=LlmItemStatus.FATAL_ERROR,
                )
            raise fatal_error from error

        if changed_fields:
            with db.write_tx():
                attempt_recorder.record_success(
                    candidate=candidate,
                    prepared_request=prepared_request,
                    outcome=outcome,
                )
                db.update_job_item_status(
                    item_id=candidate.item_id,
                    item_status=LlmItemStatus.SUCCEEDED_UPDATED,
                    changed_fields=changed_fields,
                )
            logger.debug(
                "  Updated %s in '%s' (%s): fields=%s",
                candidate.payload.note_key,
                candidate.deck_name,
                candidate.payload.note_type,
                ",".join(changed_fields),
            )
            return

        with db.write_tx():
            attempt_recorder.record_success(
                candidate=candidate,
                prepared_request=prepared_request,
                outcome=outcome,
            )
            db.update_job_item_status(
                item_id=candidate.item_id,
                item_status=LlmItemStatus.SUCCEEDED_UNCHANGED,
                changed_fields=[],
            )
        logger.debug(
            "  Unchanged %s in '%s' (%s)",
            candidate.payload.note_key,
            candidate.deck_name,
            candidate.payload.note_type,
        )

    def _apply_updates(
        self,
        *,
        db: LlmDb,
        job_id: int,
        data: dict[str, Any],
    ) -> bool:
        aggregate = db.aggregate_job(job_id)
        summary = aggregate.summary
        persisted = False

        if summary.updated > 0:
            self.deserialize_collection_fn(
                data,
                root_dir=self.collection_dir,
                note_types_dir=self.collection_dir / NOTE_TYPES_DIR,
                overwrite=True,
                quiet=True,
            )
            persisted = True
            db.set_applied_for_updated_items(job_id=job_id)
            if summary.errors:
                logger.warning(
                    "Persisted %d updated note(s) across %d deck file(s) "
                    "despite %d error(s)",
                    summary.updated,
                    len(
                        [
                            deck
                            for deck in data.get("decks", [])
                            if isinstance(deck, dict)
                        ]
                    ),
                    summary.errors,
                )
            else:
                logger.info(
                    "Persisted %d updated note(s) across %d deck file(s)",
                    summary.updated,
                    len(
                        [
                            deck
                            for deck in data.get("decks", [])
                            if isinstance(deck, dict)
                        ]
                    ),
                )

        return persisted


def plan_task(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
) -> TaskPlanResult:
    task_context = _materialize_task_context(
        collection_dir=collection_dir,
        task_name=task_name,
        model_override=model_override,
        deck_override=deck_override,
        serialize_collection_fn=serialize_collection,
    )
    return build_task_plan_result(
        task=task_context.task,
        note_type_configs=task_context.note_type_configs,
        snapshot=task_context.discovery_snapshot,
    )


async def run_task_async(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
    no_auto_commit: bool = False,
) -> LlmJobResult:
    task_context = _materialize_task_context(
        collection_dir=collection_dir,
        task_name=task_name,
        model_override=model_override,
        deck_override=deck_override,
        serialize_collection_fn=serialize_collection,
    )
    executor = LlmTaskExecutor(
        collection_dir=collection_dir,
        materialized_context=task_context,
        no_auto_commit=no_auto_commit,
    )
    return await executor.execute()


def run_task(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
    no_auto_commit: bool = False,
) -> LlmJobResult:
    return asyncio.run(
        run_task_async(
            collection_dir=collection_dir,
            task_name=task_name,
            model_override=model_override,
            deck_override=deck_override,
            no_auto_commit=no_auto_commit,
        )
    )


def list_jobs(
    *,
    collection_dir: Path,
) -> list[LlmJobListItem]:
    with _open_llm_db(collection_dir) as db:
        return db.list_jobs()


def show_job(
    *,
    collection_dir: Path,
    job_id: str,
) -> LlmJobDetail | None:
    with _open_llm_db(collection_dir) as db:
        resolved = db.resolve_job_id(job_id)
        if resolved is None:
            return None
        return db.get_job_detail(int(resolved))
