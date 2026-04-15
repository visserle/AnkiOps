"""Task planning logic for the LLM runtime."""

from __future__ import annotations

from ankiops.models import ANKIOPS_KEY_FIELD, NoteTypeConfig

from .discovery import DiscoveryItem, DiscoverySnapshot
from .task_options import format_deck_scope, format_request_defaults
from .task_types import (
    FieldAccess,
    LlmItemStatus,
    NotePayload,
    PlanFieldSurface,
    TaskConfig,
    TaskPlanResult,
    TaskRunSummary,
)


def build_task_plan_result(
    *,
    task: TaskConfig,
    note_type_configs: dict[str, NoteTypeConfig],
    snapshot: DiscoverySnapshot,
) -> TaskPlanResult:
    eligible_items = [
        item
        for item in snapshot.items
        if item.item_status is LlmItemStatus.QUEUED and item.payload is not None
    ]
    eligible_payloads = [
        item.payload for item in eligible_items if item.payload is not None
    ]
    eligible = len(eligible_items)
    skipped_no_editable_fields = sum(
        1
        for item in snapshot.items
        if item.item_status is LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS
    )
    errors = sum(
        1 for item in snapshot.items if item.item_status is LlmItemStatus.INVALID_NOTE
    )
    input_tokens_estimate = sum(
        _estimate_note_input_tokens(task, payload) for payload in eligible_payloads
    )
    max_output_tokens = task.request.max_output_tokens or 2048
    output_tokens_cap = eligible * max_output_tokens
    summary = TaskRunSummary(
        task_name=task.name,
        model=task.model,
        decks_seen=snapshot.counts.decks_seen,
        decks_matched=snapshot.counts.decks_matched,
        notes_seen=snapshot.counts.notes_seen,
        eligible=eligible,
        skipped_no_editable_fields=skipped_no_editable_fields,
        errors=errors,
        requests=eligible,
    )
    return TaskPlanResult(
        task_name=task.name,
        model=task.model,
        deck_scope=format_deck_scope(task),
        serializer_scope=format_deck_scope(task),
        system_prompt_path=(
            str(task.system_prompt_path)
            if task.system_prompt_path is not None
            else None
        ),
        prompt_path=(str(task.prompt_path) if task.prompt_path is not None else None),
        system_prompt=task.system_prompt,
        task_prompt=task.prompt,
        request_defaults=format_request_defaults(task),
        summary=summary,
        field_surface=_build_plan_field_surface(
            task=task,
            note_type_configs=note_type_configs,
            snapshot_items=snapshot.items,
        ),
        requests_estimate=eligible,
        input_tokens_estimate=input_tokens_estimate,
        output_tokens_cap=output_tokens_cap,
    )


def _estimate_tokens(text: str) -> int:
    value = text.strip()
    if not value:
        return 0
    return max(1, (len(value) + 3) // 4)


def _estimate_note_input_tokens(task: TaskConfig, payload: NotePayload) -> int:
    parts = [
        task.system_prompt,
        task.prompt,
        payload.note_key,
        payload.note_type,
    ]
    for name, value in sorted(payload.editable_fields.items()):
        parts.append(name)
        parts.append(value)
    for name, value in sorted(payload.read_only_fields.items()):
        parts.append(name)
        parts.append(value)
    return _estimate_tokens("\n".join(parts))


def _build_plan_field_surface(
    *,
    task: TaskConfig,
    note_type_configs: dict[str, NoteTypeConfig],
    snapshot_items: list[DiscoveryItem],
) -> list[PlanFieldSurface]:
    observed_note_types = {
        item.note_type
        for item in snapshot_items
        if item.note_type is not None and item.note_type_config is not None
    }
    field_surface: list[PlanFieldSurface] = []
    for note_type in sorted(observed_note_types):
        config = note_type_configs.get(note_type)
        if config is None:
            continue
        editable_fields: list[str] = []
        read_only_fields: list[str] = []
        hidden_fields: list[str] = []
        for field in config.fields:
            if field.name == ANKIOPS_KEY_FIELD.name:
                continue
            access = task.field_access(note_type, field.name)
            if access is FieldAccess.EDIT:
                editable_fields.append(field.name)
            elif access is FieldAccess.READ_ONLY:
                read_only_fields.append(field.name)
            else:
                hidden_fields.append(field.name)
        candidate_notes = sum(
            1
            for item in snapshot_items
            if item.item_status is LlmItemStatus.QUEUED and item.note_type == note_type
        )
        field_surface.append(
            PlanFieldSurface(
                note_type=note_type,
                candidate_notes=candidate_notes,
                editable_fields=editable_fields,
                read_only_fields=read_only_fields,
                hidden_fields=hidden_fields,
            )
        )
    return field_surface
