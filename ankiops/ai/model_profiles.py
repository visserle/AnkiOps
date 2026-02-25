"""Model profile configuration loader and runtime resolver."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config_utils import load_yaml_mapping, validate_config_model
from .errors import AIConfigError
from .paths import AIPaths
from .types import ModelProfile, ModelsConfig, RuntimeAIConfig

MODELS_FILE_NAME = "models.yaml"
DEFAULT_API_KEY_ENV = "ANKIOPS_AI_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_IN_FLIGHT = 4
_VALID_PROVIDERS = frozenset({"local", "remote"})


class _RawModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["local", "remote"]
    model: str
    base_url: str
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0)
    max_in_flight: int = Field(default=DEFAULT_MAX_IN_FLIGHT, gt=0)

    @field_validator("model", "base_url", "api_key_env")
    @classmethod
    def _normalize_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        return normalized


class _RawModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int | None = None
    default_profile: str | None = None
    profiles: dict[str, _RawModelProfile]

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("must be > 0")
        return value

    @field_validator("profiles", mode="before")
    @classmethod
    def _normalize_profiles(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError("must be a non-empty mapping")

        normalized: dict[str, Any] = {}
        for raw_name, raw_profile in value.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("profile names must be non-empty strings")
            profile_name = raw_name.strip()
            if profile_name in normalized:
                raise ValueError(f"duplicate profile '{profile_name}'")
            normalized[profile_name] = raw_profile
        return normalized

    @field_validator("default_profile")
    @classmethod
    def _normalize_default_profile(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        return normalized


def models_config_path(ai_paths: AIPaths) -> Path:
    """Return path to the collection-scoped model profile config."""
    return ai_paths.models


def load_model_profiles(ai_paths: AIPaths) -> ModelsConfig:
    """Load model profiles from ai/models.yaml."""
    path = models_config_path(ai_paths)
    raw = load_yaml_mapping(
        path,
        error_type=AIConfigError,
        mapping_label="Model profile config",
        missing_message=(
            f"Model profile config not found: {path}. "
            "Run 'ankiops init' to eject built-in AI assets."
        ),
    )
    parsed = validate_config_model(
        raw,
        model_type=_RawModelsConfig,
        path=path,
        error_type=AIConfigError,
        config_label="models config",
    )

    default_profile = parsed.default_profile or next(iter(parsed.profiles))
    if default_profile not in parsed.profiles:
        raise AIConfigError(
            f"default_profile '{default_profile}' was not found in profiles in '{path}'"
        )

    profiles = {
        profile_name: ModelProfile(
            name=profile_name,
            provider=profile.provider,
            model=profile.model,
            base_url=profile.base_url,
            api_key_env=profile.api_key_env,
            timeout_seconds=profile.timeout_seconds,
            max_in_flight=profile.max_in_flight,
        )
        for profile_name, profile in parsed.profiles.items()
    }
    return ModelsConfig(
        default_profile=default_profile,
        profiles=profiles,
        source_path=path,
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

    runtime_provider = _coerce_provider(provider or selected.provider)
    runtime_model = _require_runtime_string(model, fallback=selected.model, key="model")
    runtime_base_url = _require_runtime_string(
        base_url,
        fallback=selected.base_url,
        key="base_url",
    )
    runtime_api_key_env = (
        _normalize_runtime_string(
            api_key_env,
            fallback=selected.api_key_env,
            key="api_key_env",
        )
        or DEFAULT_API_KEY_ENV
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
        fallback=os.environ.get(runtime_api_key_env),
        key="api_key",
    )

    return RuntimeAIConfig(
        profile=profile_name,
        provider=runtime_provider,
        model=runtime_model,
        base_url=runtime_base_url,
        api_key_env=runtime_api_key_env,
        timeout_seconds=runtime_timeout,
        max_in_flight=runtime_max_in_flight,
        api_key=resolved_api_key,
    )


def _coerce_provider(value: str) -> str:
    normalized = value.strip()
    if normalized not in _VALID_PROVIDERS:
        allowed = ", ".join(sorted(_VALID_PROVIDERS))
        raise AIConfigError(f"provider must be one of: {allowed}")
    return normalized


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
