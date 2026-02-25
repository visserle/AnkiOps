"""Prompt YAML loading and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import PromptConfigError
from .model_profiles import MODELS_FILE_NAME
from .types import PromptConfig


class _RawPromptConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    prompt: str
    target_fields: list[str] | None = None
    send_fields: list[str] | None = None
    note_types: list[str] | None = None
    model_profile: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)

    @field_validator("name", "model_profile")
    @classmethod
    def _normalize_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        return normalized

    @field_validator("prompt")
    @classmethod
    def _normalize_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        return normalized

    @field_validator("target_fields", "send_fields", "note_types", mode="before")
    @classmethod
    def _normalize_pattern_list(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            raw_values = [value]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            raw_values = list(value)
        else:
            raise ValueError("must be a string or list of strings")

        normalized = [item.strip() for item in raw_values if item.strip()]
        if not normalized:
            raise ValueError("must not be empty")
        return normalized


def resolve_prompt_path(prompts_dir: Path, prompt_ref: str) -> Path:
    """Resolve a prompt name/path to a YAML file path."""
    raw = _require_prompt_ref(prompt_ref)

    direct = Path(raw)
    if direct.exists() and direct.is_file():
        return _assert_is_prompt_file(direct)

    candidates = _candidate_prompt_paths(prompts_dir, raw)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return _assert_is_prompt_file(candidate)

    tried = ", ".join(str(candidate_path) for candidate_path in candidates)
    raise PromptConfigError(f"Prompt not found: '{prompt_ref}'. Tried: {tried}")


def load_prompt(prompts_dir: Path, prompt_ref: str) -> PromptConfig:
    """Load a prompt YAML file into a validated PromptConfig."""
    path = resolve_prompt_path(prompts_dir, prompt_ref)
    raw = _load_yaml_mapping(path)

    normalized = {
        "name": raw.get("name"),
        "prompt": raw.get("prompt"),
        "target_fields": raw.get("target_fields", raw.get("fields_to_edit")),
        "send_fields": raw.get("send_fields", raw.get("fields_to_send")),
        "note_types": raw.get("note_types"),
        "model_profile": raw.get("model_profile"),
        "temperature": raw.get("temperature", 0),
    }
    parsed = _validate_raw_prompt(normalized, path)

    target_fields = parsed.target_fields or ["*"]
    send_fields = parsed.send_fields or list(target_fields)
    note_types = parsed.note_types or ["*"]
    return PromptConfig(
        name=parsed.name or path.stem,
        prompt=parsed.prompt,
        target_fields=target_fields,
        send_fields=send_fields,
        note_types=note_types,
        model_profile=parsed.model_profile,
        temperature=parsed.temperature,
        source_path=path,
    )


def _require_prompt_ref(prompt_ref: str) -> str:
    if not isinstance(prompt_ref, str) or not prompt_ref.strip():
        raise PromptConfigError("Prompt name/path cannot be empty.")
    return prompt_ref.strip()


def _assert_is_prompt_file(path: Path) -> Path:
    if path.name == MODELS_FILE_NAME:
        raise PromptConfigError(f"'{MODELS_FILE_NAME}' is model config, not a prompt.")
    return path


def _candidate_prompt_paths(prompts_dir: Path, prompt_ref: str) -> list[Path]:
    candidate_in_dir = prompts_dir / prompt_ref
    if candidate_in_dir.suffix:
        return [candidate_in_dir]
    return [
        prompts_dir / f"{prompt_ref}.yaml",
        prompts_dir / f"{prompt_ref}.yml",
        prompts_dir / prompt_ref,
    ]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise PromptConfigError(f"Prompt file must contain a YAML mapping: {path}")
    return raw


def _validate_raw_prompt(raw: dict[str, Any], path: Path) -> _RawPromptConfig:
    try:
        return _RawPromptConfig.model_validate(raw)
    except ValidationError as error:
        first = error.errors()[0]
        field_path = ".".join(str(part) for part in first.get("loc", ()))
        detail = first.get("msg", "invalid value")
        if field_path:
            raise PromptConfigError(
                f"Invalid prompt '{path}' field '{field_path}': {detail}"
            ) from None
        raise PromptConfigError(f"Invalid prompt '{path}': {detail}") from None
