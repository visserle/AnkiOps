"""Prompt YAML loading and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config_utils import load_yaml_mapping, validate_config_model
from .errors import PromptConfigError
from .model_profiles import MODELS_FILE_NAME
from .paths import AIPaths
from .types import PromptConfig

_VALID_PROMPT_SUFFIXES = frozenset({".yaml", ".yml"})


class _RawPromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    prompt: str
    target_fields: list[str] | None = None
    send_fields: list[str] | None = None
    note_types: list[str] | None = None
    model_profile: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)

    @field_validator("name", "description", "model_profile")
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


def resolve_prompt_path(ai_paths: AIPaths, prompt_ref: str) -> Path:
    """Resolve a prompt name/path to a YAML file path."""
    raw = _require_prompt_ref(prompt_ref)
    candidates = _candidate_prompt_paths(ai_paths.prompts, raw)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return _assert_is_prompt_file(candidate)

    tried = ", ".join(str(candidate_path) for candidate_path in candidates)
    raise PromptConfigError(f"Prompt not found: '{prompt_ref}'. Tried: {tried}")


def load_prompt(ai_paths: AIPaths, prompt_ref: str) -> PromptConfig:
    """Load a prompt YAML file into a validated PromptConfig."""
    path = resolve_prompt_path(ai_paths, prompt_ref)
    raw = load_yaml_mapping(
        path,
        error_type=PromptConfigError,
        mapping_label="Prompt file",
    )
    parsed = validate_config_model(
        raw,
        model_type=_RawPromptConfig,
        path=path,
        error_type=PromptConfigError,
        config_label="prompt",
    )

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
    if path.suffix.lower() not in _VALID_PROMPT_SUFFIXES:
        raise PromptConfigError(f"Prompt file must use .yaml or .yml extension: {path}")
    if path.name == MODELS_FILE_NAME:
        raise PromptConfigError(f"'{MODELS_FILE_NAME}' is model config, not a prompt.")
    return path


def _candidate_prompt_paths(prompts_dir: Path, prompt_ref: str) -> list[Path]:
    ref_path = Path(prompt_ref)
    if ref_path.is_absolute():
        if ref_path.suffix:
            return [ref_path]
        return [ref_path.with_suffix(".yaml"), ref_path.with_suffix(".yml")]

    candidate_in_dir = prompts_dir / ref_path
    if ref_path.suffix:
        return [candidate_in_dir]
    return [
        candidate_in_dir.with_suffix(".yaml"),
        candidate_in_dir.with_suffix(".yml"),
    ]
