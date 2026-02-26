"""Model profile configuration loader and runtime resolver."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config_utils import load_yaml_mapping, validate_config_model
from .errors import AIConfigError
from .paths import AIPaths
from .providers import get_provider_spec, provider_ids
from .types import ModelProfile, ModelsConfig, RuntimeAIConfig

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_IN_FLIGHT = 4
_VALID_MODEL_SUFFIXES = frozenset({".yaml", ".yml"})


class _RawModelProfileFile(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: str = Field(default="ai.model.v1", alias="schema")
    id: str | None = None
    default: bool = False
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0)
    max_in_flight: int = Field(default=DEFAULT_MAX_IN_FLIGHT, gt=0)

    @field_validator("schema_name")
    @classmethod
    def _validate_schema(cls, value: str) -> str:
        normalized = value.strip()
        if normalized != "ai.model.v1":
            raise ValueError("must equal 'ai.model.v1'")
        return normalized

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        return _coerce_provider(value)

    @field_validator("id", "model", "base_url", "api_key_env")
    @classmethod
    def _normalize_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        return normalized


def models_config_path(ai_paths: AIPaths) -> Path:
    """Return path to the collection-scoped model profile directory."""
    return ai_paths.models


def load_model_configs(ai_paths: AIPaths) -> ModelsConfig:
    """Load model profiles from ai/models/*.yaml."""
    models_dir = models_config_path(ai_paths)
    if not models_dir.exists() or not models_dir.is_dir():
        raise AIConfigError(
            f"Model profile directory not found: {models_dir}. "
            "Run 'ankiops init' to eject built-in AI assets."
        )

    model_files = sorted(
        path
        for path in models_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _VALID_MODEL_SUFFIXES
    )
    if not model_files:
        raise AIConfigError(f"No model profile YAML files found in: {models_dir}")

    profiles: dict[str, ModelProfile] = {}
    default_profiles: list[str] = []
    for path in model_files:
        raw = load_yaml_mapping(
            path,
            error_type=AIConfigError,
            mapping_label="Model profile file",
        )
        parsed = validate_config_model(
            raw,
            model_type=_RawModelProfileFile,
            path=path,
            error_type=AIConfigError,
            config_label="model profile",
        )

        profile_name = parsed.id or path.stem
        if parsed.id is not None and parsed.id != path.stem:
            raise AIConfigError(
                f"Model profile id must match file name stem in '{path}' "
                f"(expected '{path.stem}', got '{parsed.id}')"
            )
        if profile_name in profiles:
            raise AIConfigError(f"Duplicate model profile id '{profile_name}'")

        profiles[profile_name] = ModelProfile(
            name=profile_name,
            provider=parsed.provider,
            model=parsed.model,
            base_url_override=parsed.base_url,
            api_key_env_override=parsed.api_key_env,
            timeout_seconds=parsed.timeout_seconds,
            max_in_flight=parsed.max_in_flight,
        )
        if parsed.default:
            default_profiles.append(profile_name)

    if len(default_profiles) > 1:
        all_defaults = ", ".join(default_profiles)
        raise AIConfigError(
            f"Multiple model profiles marked as default: {all_defaults}"
        )
    default_profile = default_profiles[0] if default_profiles else next(iter(profiles))

    return ModelsConfig(
        default_profile=default_profile,
        profiles=profiles,
        source_path=models_dir,
    )


def resolve_runtime_config(
    models_config: ModelsConfig,
    *,
    profile: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: int | None = None,
    max_in_flight: int | None = None,
    api_key: str | None = None,
) -> RuntimeAIConfig:
    """Resolve runtime config from profile defaults plus runtime overrides."""
    profile_name = (profile or models_config.default_profile).strip()
    selected = models_config.profiles.get(profile_name)
    if selected is None:
        known = ", ".join(sorted(models_config.profiles.keys()))
        raise AIConfigError(f"Unknown model profile '{profile_name}'. Known: {known}")

    selected_provider = _coerce_provider(selected.provider)
    runtime_provider = _coerce_provider(provider or selected_provider)
    provider_spec = get_provider_spec(runtime_provider)
    provider_changed = provider is not None and runtime_provider != selected_provider

    if provider_changed:
        runtime_base_url_fallback = provider_spec.default_base_url
        runtime_api_key_env_fallback = provider_spec.default_api_key_env
    else:
        runtime_base_url_fallback = (
            selected.base_url_override or provider_spec.default_base_url
        )
        runtime_api_key_env_fallback = (
            selected.api_key_env_override
            if selected.api_key_env_override is not None
            else provider_spec.default_api_key_env
        )

    runtime_model = _require_runtime_string(model, fallback=selected.model, key="model")
    runtime_base_url = _require_runtime_string(
        base_url,
        fallback=runtime_base_url_fallback,
        key="base_url",
    )
    runtime_api_key_env = _normalize_runtime_string(
        api_key_env,
        fallback=runtime_api_key_env_fallback,
        key="api_key_env",
    )
    runtime_timeout = _resolve_positive_int(
        timeout_seconds,
        fallback=selected.timeout_seconds,
        key="timeout",
    )
    runtime_max_in_flight = _resolve_positive_int(
        max_in_flight,
        fallback=selected.max_in_flight,
        key="max_in_flight",
    )
    resolved_api_key = _normalize_runtime_string(
        api_key,
        fallback=(
            os.environ.get(runtime_api_key_env)
            if runtime_api_key_env is not None
            else None
        ),
        key="api_key",
    )

    return RuntimeAIConfig(
        profile=profile_name,
        provider=runtime_provider,
        transport=provider_spec.transport,
        model=runtime_model,
        base_url=runtime_base_url,
        api_key_env=runtime_api_key_env,
        requires_api_key=provider_spec.requires_api_key,
        timeout_seconds=runtime_timeout,
        max_in_flight=runtime_max_in_flight,
        api_key=resolved_api_key,
    )


def _coerce_provider(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        allowed = ", ".join(provider_ids())
        raise AIConfigError(f"provider must be one of: {allowed}")
    get_provider_spec(normalized)
    return normalized


def provider_choices() -> tuple[str, ...]:
    """Return supported provider ids for CLI argument choices."""
    return provider_ids()


def _require_runtime_string(raw: str | None, *, fallback: str, key: str) -> str:
    value = _normalize_runtime_string(raw, fallback=fallback, key=key)
    if value is None:
        raise AIConfigError(f"{key} must be a non-empty string")
    return value


def _normalize_runtime_string(
    raw: str | None,
    *,
    fallback: str | None,
    key: str,
) -> str | None:
    value = fallback if raw is None else raw
    if value is None:
        return None
    if not isinstance(value, str):
        raise AIConfigError(f"{key} must be a string")
    normalized = value.strip()
    return normalized or None


def _resolve_positive_int(raw: int | None, *, fallback: int, key: str) -> int:
    value = fallback if raw is None else raw
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AIConfigError(f"{key} must be > 0")
    return value
