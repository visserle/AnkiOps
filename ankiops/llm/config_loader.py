"""YAML config loading for Claude task specs."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from ankiops.config import LLM_DIR
from ankiops.models import NoteTypeConfig

from .anthropic_models import format_supported_model_names, parse_model
from .llm_models import (
    DeckScope,
    ExecutionMode,
    FieldExceptionRule,
    TaskCatalog,
    TaskConfig,
    TaskExecutionOptions,
    TaskRequestOptions,
)


class LlmConfigError(ValueError):
    """Raised when a Claude task config is invalid."""


_SYSTEM_PROMPT_FILE_NAME = "system_prompt.md"
_TASKS_DIR_NAME = "tasks"


def _iter_yaml_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.yaml"))
    files.extend(sorted(directory.glob("*.yml")))
    return files


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise LlmConfigError(f"{path}: config must be a YAML mapping")
    return raw


def _require_str(mapping: dict[str, Any], key: str, path: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LlmConfigError(f"{path}: '{key}' must be a non-empty string")
    return value.strip()


def _optional_str(mapping: dict[str, Any], key: str, path: Path) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise LlmConfigError(f"{path}: '{key}' must be a non-empty string")
    return value.strip()


def _read_text_file(path: Path, *, label: str) -> str:
    if not path.exists() or not path.is_file():
        raise LlmConfigError(f"{path}: {label} file not found")
    value = path.read_text(encoding="utf-8")
    if not value.strip():
        raise LlmConfigError(f"{path}: {label} file must be non-empty")
    return value.strip()


def _resolve_relative_file(
    task_path: Path,
    raw_reference: str,
    *,
    llm_dir: Path,
    key: str,
) -> Path:
    candidate = (task_path.parent / raw_reference).resolve()
    try:
        candidate.relative_to(llm_dir.resolve())
    except ValueError as error:
        raise LlmConfigError(
            f"{task_path}: '{key}' must stay within {llm_dir}"
        ) from error
    return candidate


def _parse_str_list(value: Any, key: str, path: Path) -> list[str]:
    if not isinstance(value, list) or not value:
        raise LlmConfigError(f"{path}: '{key}' must be a non-empty list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LlmConfigError(f"{path}: '{key}' must contain only strings")
        items.append(item.strip())
    return items


def _parse_optional_str_list(value: Any, key: str, path: Path) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise LlmConfigError(f"{path}: '{key}' must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LlmConfigError(f"{path}: '{key}' must contain only strings")
        items.append(item.strip())
    return items


def _is_non_bool_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_request_options(value: Any, path: Path) -> TaskRequestOptions:
    if value is None:
        return TaskRequestOptions()
    if not isinstance(value, dict):
        raise LlmConfigError(f"{path}: request options must be a mapping")

    valid_keys = {
        "temperature",
        "max_output_tokens",
        "retries",
        "retry_backoff_seconds",
        "retry_backoff_jitter",
    }
    unknown = sorted(set(value.keys()) - valid_keys)
    if unknown:
        raise LlmConfigError(f"{path}: unknown request option(s): {', '.join(unknown)}")

    temperature = value.get("temperature")
    if temperature is not None:
        if not _is_non_bool_number(temperature):
            raise LlmConfigError(f"{path}: 'temperature' must be numeric")
        if not 0 <= float(temperature) <= 1:
            raise LlmConfigError(f"{path}: 'temperature' must be between 0 and 1")

    max_output_tokens = value.get("max_output_tokens")
    if max_output_tokens is not None and (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or max_output_tokens <= 0
    ):
        raise LlmConfigError(f"{path}: 'max_output_tokens' must be a positive integer")

    retries = value.get("retries", 2)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise LlmConfigError(f"{path}: 'retries' must be a non-negative integer")

    retry_backoff_seconds = value.get("retry_backoff_seconds", 0.5)
    if not _is_non_bool_number(retry_backoff_seconds):
        raise LlmConfigError(f"{path}: 'retry_backoff_seconds' must be numeric")
    if float(retry_backoff_seconds) <= 0:
        raise LlmConfigError(f"{path}: 'retry_backoff_seconds' must be > 0")

    retry_backoff_jitter = value.get("retry_backoff_jitter", True)
    if not isinstance(retry_backoff_jitter, bool):
        raise LlmConfigError(f"{path}: 'retry_backoff_jitter' must be a boolean")

    return TaskRequestOptions(
        temperature=float(temperature) if temperature is not None else None,
        max_output_tokens=max_output_tokens,
        retries=retries,
        retry_backoff_seconds=float(retry_backoff_seconds),
        retry_backoff_jitter=retry_backoff_jitter,
    )


def _parse_execution_options(value: Any, path: Path) -> TaskExecutionOptions:
    if value is None:
        return TaskExecutionOptions()
    if not isinstance(value, dict):
        raise LlmConfigError(f"{path}: 'execution' must be a mapping")

    valid_keys = {
        "mode",
        "concurrency",
        "fail_fast",
        "batch_poll_seconds",
    }
    unknown = sorted(set(value.keys()) - valid_keys)
    if unknown:
        raise LlmConfigError(
            f"{path}: unknown execution option(s): {', '.join(unknown)}"
        )

    raw_mode = value.get("mode")
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        raise LlmConfigError(f"{path}: 'execution.mode' must be a non-empty string")
    normalized_mode = raw_mode.strip().lower()
    try:
        mode = ExecutionMode(normalized_mode)
    except ValueError as error:
        supported = ", ".join(execution_mode.value for execution_mode in ExecutionMode)
        raise LlmConfigError(
            f"{path}: 'execution.mode' must be one of: {supported}"
        ) from error

    concurrency = value.get("concurrency", 8)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool):
        raise LlmConfigError(f"{path}: 'execution.concurrency' must be an integer")
    if concurrency <= 0:
        raise LlmConfigError(f"{path}: 'execution.concurrency' must be > 0")

    fail_fast = value.get("fail_fast", True)
    if not isinstance(fail_fast, bool):
        raise LlmConfigError(f"{path}: 'execution.fail_fast' must be a boolean")

    batch_poll_seconds = value.get("batch_poll_seconds", 15)
    if (
        not isinstance(batch_poll_seconds, int)
        or isinstance(batch_poll_seconds, bool)
        or batch_poll_seconds <= 0
    ):
        raise LlmConfigError(
            f"{path}: 'execution.batch_poll_seconds' must be a positive integer"
        )

    if mode is ExecutionMode.BATCH and "concurrency" in value:
        raise LlmConfigError(
            f"{path}: 'execution.concurrency' is only supported for online mode"
        )
    if mode is ExecutionMode.ONLINE and "batch_poll_seconds" in value:
        raise LlmConfigError(
            f"{path}: 'execution.batch_poll_seconds' is only supported for batch mode"
        )

    return TaskExecutionOptions(
        mode=mode,
        concurrency=concurrency,
        fail_fast=fail_fast,
        batch_poll_seconds=batch_poll_seconds,
    )


def _parse_deck_scope(value: Any, path: Path) -> DeckScope:
    if value is None:
        return DeckScope()
    if not isinstance(value, dict):
        raise LlmConfigError(f"{path}: 'decks' must be a mapping")

    unknown = sorted(set(value.keys()) - {"include", "exclude", "include_subdecks"})
    if unknown:
        raise LlmConfigError(f"{path}: unknown deck scope key(s): {', '.join(unknown)}")

    include_value = value.get("include", ["*"])
    exclude_value = value.get("exclude", [])
    include_subdecks = value.get("include_subdecks", True)
    if not isinstance(include_value, list) or not include_value:
        raise LlmConfigError(f"{path}: 'decks.include' must be a non-empty list")
    if not isinstance(exclude_value, list):
        raise LlmConfigError(f"{path}: 'decks.exclude' must be a list")
    if not isinstance(include_subdecks, bool):
        raise LlmConfigError(f"{path}: 'decks.include_subdecks' must be a boolean")

    include = _parse_str_list(include_value, "decks.include", path)
    exclude = _parse_optional_str_list(exclude_value, "decks.exclude", path)
    return DeckScope(
        include=include,
        exclude=exclude,
        include_subdecks=include_subdecks,
    )


def _field_names_by_note_type(
    note_type_configs: list[NoteTypeConfig],
) -> dict[str, set[str]]:
    return {
        config.name: {field.name for field in config.fields}
        for config in note_type_configs
    }


def _parse_field_exceptions(
    value: Any,
    *,
    note_type_configs: list[NoteTypeConfig],
    path: Path,
) -> list[FieldExceptionRule]:
    if value is None:
        return []
    if not isinstance(value, dict):
        raise LlmConfigError(f"{path}: 'fields' must be a mapping")

    unknown_fields = sorted(set(value.keys()) - {"exceptions"})
    if unknown_fields:
        raise LlmConfigError(
            f"{path}: unknown fields config key(s): {', '.join(unknown_fields)}"
        )

    exceptions_value = value.get("exceptions", [])
    if not isinstance(exceptions_value, list):
        raise LlmConfigError(f"{path}: 'fields.exceptions' must be a list")

    note_type_names = [config.name for config in note_type_configs]
    field_names = _field_names_by_note_type(note_type_configs)
    parsed: list[FieldExceptionRule] = []

    for index, entry in enumerate(exceptions_value):
        entry_path = Path(f"{path}#fields.exceptions[{index}]")
        if not isinstance(entry, dict):
            raise LlmConfigError(f"{entry_path}: exception must be a mapping")

        unknown = sorted(set(entry.keys()) - {"note_types", "read_only", "hidden"})
        if unknown:
            raise LlmConfigError(
                f"{entry_path}: unknown exception key(s): {', '.join(unknown)}"
            )

        note_types = (
            _parse_str_list(entry["note_types"], "note_types", entry_path)
            if "note_types" in entry
            else ["*"]
        )
        read_only = _parse_optional_str_list(
            entry.get("read_only"), "read_only", entry_path
        )
        hidden = _parse_optional_str_list(entry.get("hidden"), "hidden", entry_path)
        if not read_only and not hidden:
            raise LlmConfigError(
                f"{entry_path}: exception must declare 'read_only' or 'hidden'"
            )

        matched_types = {
            note_type_name
            for note_type_name in note_type_names
            if any(fnmatchcase(note_type_name, pattern) for pattern in note_types)
        }
        if not matched_types:
            patterns = ", ".join(note_types)
            raise LlmConfigError(
                f"{entry_path}: note type pattern(s) match no note types: {patterns}"
            )

        available_fields = set().union(*(field_names[name] for name in matched_types))
        for field_name in read_only + hidden:
            if field_name not in available_fields:
                raise LlmConfigError(
                    f"{entry_path}: field '{field_name}' does not exist on any matched "
                    "note type"
                )

        parsed.append(
            FieldExceptionRule(
                note_types=note_types,
                read_only=read_only,
                hidden=hidden,
            )
        )

    return parsed


def _parse_task(
    path: Path,
    *,
    note_type_configs: list[NoteTypeConfig],
    llm_dir: Path,
) -> TaskConfig:
    mapping = _read_yaml_mapping(path)
    name = path.stem
    model_name = _require_str(mapping, "model", path)
    model = parse_model(model_name)
    if model is None:
        supported = format_supported_model_names()
        raise LlmConfigError(f"{path}: 'model' must be one of: {supported}")
    prompt_file_ref = _require_str(mapping, "prompt_file", path)
    prompt_file_path = _resolve_relative_file(
        path,
        prompt_file_ref,
        llm_dir=llm_dir,
        key="prompt_file",
    )
    prompt = _read_text_file(prompt_file_path, label="prompt")
    system_prompt_file_ref = _optional_str(mapping, "system_prompt_file", path)
    if system_prompt_file_ref is None:
        system_prompt_path = (llm_dir / _SYSTEM_PROMPT_FILE_NAME).resolve()
    else:
        system_prompt_path = _resolve_relative_file(
            path,
            system_prompt_file_ref,
            llm_dir=llm_dir,
            key="system_prompt_file",
        )
    system_prompt = _read_text_file(system_prompt_path, label="system prompt")
    api_key_env = _optional_str(mapping, "api_key_env", path) or "ANTHROPIC_API_KEY"
    timeout_seconds = mapping.get("timeout_seconds", 60)
    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        raise LlmConfigError(f"{path}: 'timeout_seconds' must be a positive integer")
    decks = _parse_deck_scope(mapping.get("decks"), path)
    field_exceptions = _parse_field_exceptions(
        mapping.get("fields"),
        note_type_configs=note_type_configs,
        path=path,
    )
    request = _parse_request_options(mapping.get("request"), path)
    execution = _parse_execution_options(mapping.get("execution"), path)

    unknown = sorted(
        set(mapping.keys())
        - {
            "model",
            "prompt_file",
            "system_prompt_file",
            "api_key_env",
            "timeout_seconds",
            "decks",
            "fields",
            "request",
            "execution",
        }
    )
    if unknown:
        raise LlmConfigError(f"{path}: unknown task key(s): {', '.join(unknown)}")

    return TaskConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        prompt=prompt,
        system_prompt_path=system_prompt_path,
        prompt_path=prompt_file_path,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        decks=decks,
        field_exceptions=field_exceptions,
        request=request,
        execution=execution,
    )


def load_llm_task_catalog(
    collection_dir: Path,
    *,
    note_type_configs: list[NoteTypeConfig],
) -> TaskCatalog:
    llm_dir = collection_dir / LLM_DIR
    tasks_by_name: dict[str, TaskConfig] = {}
    errors: dict[str, str] = {}

    if not llm_dir.exists() or not llm_dir.is_dir():
        errors[str(llm_dir)] = (
            f"{llm_dir}: LLM config directory not found. "
            "Run 'ankiops init' to create IaC LLM configs."
        )
        return TaskCatalog(tasks_by_name=tasks_by_name, errors=errors)

    tasks_dir = llm_dir / _TASKS_DIR_NAME
    if not tasks_dir.exists() or not tasks_dir.is_dir():
        errors[str(tasks_dir)] = (
            f"{tasks_dir}: task directory not found. "
            "Expected task YAML files in llm/tasks/."
        )
        return TaskCatalog(tasks_by_name=tasks_by_name, errors=errors)

    task_files = _iter_yaml_files(tasks_dir)
    if not task_files:
        errors[str(tasks_dir)] = f"{tasks_dir}: no task YAML files found"
        return TaskCatalog(tasks_by_name=tasks_by_name, errors=errors)

    for path in task_files:
        try:
            task = _parse_task(
                path,
                note_type_configs=note_type_configs,
                llm_dir=llm_dir,
            )
            if task.name in tasks_by_name:
                raise LlmConfigError(f"{path}: duplicate task name '{task.name}'")
            tasks_by_name[task.name] = task
        except LlmConfigError as error:
            errors[str(path)] = str(error)

    return TaskCatalog(
        tasks_by_name=tasks_by_name,
        errors=errors,
    )
