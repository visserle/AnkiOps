"""YAML config loading for LLM tasks and providers."""

from __future__ import annotations

import os
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from ankiops.config import LLM_DIR, LLM_PROVIDERS_DIR, LLM_TASKS_DIR
from ankiops.models import NoteTypeConfig

from .models import (
    DeckScope,
    FieldExceptionRule,
    LlmConfigSet,
    ProviderConfig,
    ProviderType,
    TaskConfig,
    TaskRequestOptions,
)


class LlmConfigError(ValueError):
    """Raised when an LLM task or provider config is invalid."""


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


def _require_version(mapping: dict[str, Any], path: Path) -> int:
    version = mapping.get("version")
    if version != 1:
        raise LlmConfigError(f"{path}: only version 1 configs are supported")
    return 1


def _require_name_stem(name: str, path: Path) -> None:
    if path.stem != name:
        raise LlmConfigError(
            f"{path}: config name '{name}' must match file name '{path.stem}'"
        )


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


def _parse_request_options(value: Any, path: Path) -> TaskRequestOptions:
    if value is None:
        return TaskRequestOptions()
    if not isinstance(value, dict):
        raise LlmConfigError(f"{path}: request options must be a mapping")

    valid_keys = {"temperature", "max_output_tokens"}
    unknown = sorted(set(value.keys()) - valid_keys)
    if unknown:
        raise LlmConfigError(f"{path}: unknown request option(s): {', '.join(unknown)}")

    temperature = value.get("temperature")
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise LlmConfigError(f"{path}: 'temperature' must be numeric")

    max_output_tokens = value.get("max_output_tokens")
    if max_output_tokens is not None and (
        not isinstance(max_output_tokens, int) or max_output_tokens <= 0
    ):
        raise LlmConfigError(f"{path}: 'max_output_tokens' must be a positive integer")

    return TaskRequestOptions(
        temperature=float(temperature) if temperature is not None else None,
        max_output_tokens=max_output_tokens,
    )


def _parse_deck_scope(value: Any, path: Path) -> DeckScope:
    if value is None:
        return DeckScope()
    if not isinstance(value, dict):
        raise LlmConfigError(f"{path}: 'decks' must be a mapping")

    unknown = sorted(set(value.keys()) - {"include", "exclude"})
    if unknown:
        raise LlmConfigError(f"{path}: unknown deck scope key(s): {', '.join(unknown)}")

    include_value = value.get("include", ["*"])
    exclude_value = value.get("exclude", [])
    if not isinstance(include_value, list) or not include_value:
        raise LlmConfigError(f"{path}: 'decks.include' must be a non-empty list")
    if not isinstance(exclude_value, list):
        raise LlmConfigError(f"{path}: 'decks.exclude' must be a list")

    include = _parse_str_list(include_value, "decks.include", path)
    exclude = _parse_optional_str_list(exclude_value, "decks.exclude", path)
    return DeckScope(include=include, exclude=exclude)


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


def _parse_provider(path: Path) -> ProviderConfig:
    mapping = _read_yaml_mapping(path)
    version = _require_version(mapping, path)
    name = _require_str(mapping, "name", path)
    _require_name_stem(name, path)
    provider_type_raw = _require_str(mapping, "type", path)
    try:
        provider_type = ProviderType(provider_type_raw)
    except ValueError as error:
        raise LlmConfigError(
            f"{path}: unsupported provider type '{provider_type_raw}'"
        ) from error

    base_url = _require_str(mapping, "base_url", path).rstrip("/")
    model = _require_str(mapping, "model", path)
    api_key_env = _optional_str(mapping, "api_key_env", path)
    timeout_seconds = mapping.get("timeout_seconds", 60)
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise LlmConfigError(f"{path}: 'timeout_seconds' must be a positive integer")

    request_defaults = _parse_request_options(mapping.get("request_defaults"), path)

    if provider_type is ProviderType.OPENAI and not api_key_env:
        raise LlmConfigError(f"{path}: openai provider requires 'api_key_env'")
    if api_key_env and not os.environ.get(api_key_env):
        raise LlmConfigError(
            f"{path}: required environment variable '{api_key_env}' is not set"
        )

    return ProviderConfig(
        version=version,
        name=name,
        type=provider_type,
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        request_defaults=request_defaults,
    )


def _parse_task(path: Path, *, note_type_configs: list[NoteTypeConfig]) -> TaskConfig:
    mapping = _read_yaml_mapping(path)
    version = _require_version(mapping, path)
    name = _require_str(mapping, "name", path)
    _require_name_stem(name, path)
    provider = _require_str(mapping, "provider", path)
    prompt = _require_str(mapping, "prompt", path)
    decks = _parse_deck_scope(mapping.get("decks"), path)
    field_exceptions = _parse_field_exceptions(
        mapping.get("fields"),
        note_type_configs=note_type_configs,
        path=path,
    )
    request = _parse_request_options(mapping.get("request"), path)

    unknown = sorted(
        set(mapping.keys())
        - {"version", "name", "provider", "decks", "prompt", "fields", "request"}
    )
    if unknown:
        raise LlmConfigError(f"{path}: unknown task key(s): {', '.join(unknown)}")

    return TaskConfig(
        version=version,
        name=name,
        provider=provider,
        prompt=prompt,
        decks=decks,
        field_exceptions=field_exceptions,
        request=request,
    )


def load_llm_config_set(
    collection_dir: Path,
    *,
    note_type_configs: list[NoteTypeConfig],
) -> LlmConfigSet:
    llm_dir = collection_dir / LLM_DIR
    providers_dir = llm_dir / LLM_PROVIDERS_DIR
    tasks_dir = llm_dir / LLM_TASKS_DIR

    providers_by_name: dict[str, ProviderConfig] = {}
    tasks_by_name: dict[str, TaskConfig] = {}
    provider_errors: dict[str, str] = {}
    task_errors: dict[str, str] = {}

    if providers_dir.exists():
        for path in _iter_yaml_files(providers_dir):
            try:
                provider = _parse_provider(path)
                if provider.name in providers_by_name:
                    raise LlmConfigError(
                        f"{path}: duplicate provider name '{provider.name}'"
                    )
                providers_by_name[provider.name] = provider
            except LlmConfigError as error:
                provider_errors[str(path)] = str(error)

    if tasks_dir.exists():
        for path in _iter_yaml_files(tasks_dir):
            try:
                task = _parse_task(path, note_type_configs=note_type_configs)
                if task.name in tasks_by_name:
                    raise LlmConfigError(f"{path}: duplicate task name '{task.name}'")
                if task.provider not in providers_by_name:
                    raise LlmConfigError(
                        f"{path}: referenced provider '{task.provider}' "
                        "is invalid or missing"
                    )
                tasks_by_name[task.name] = task
            except LlmConfigError as error:
                task_errors[str(path)] = str(error)

    return LlmConfigSet(
        providers_by_name=providers_by_name,
        tasks_by_name=tasks_by_name,
        provider_errors=provider_errors,
        task_errors=task_errors,
    )
