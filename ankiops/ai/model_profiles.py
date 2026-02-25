"""Model profile configuration loader and runtime resolver."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .types import ModelProfile, ModelsConfig, RuntimeAIConfig

VALID_PROVIDERS = {"local", "remote"}
MODELS_FILE_NAME = "models.yaml"
DEFAULT_API_KEY_ENV = "ANKIOPS_AI_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_IN_FLIGHT = 4


def models_config_path(prompts_dir: Path) -> Path:
    """Return path to the collection-scoped model profile config."""
    return prompts_dir / MODELS_FILE_NAME


def load_models_config(prompts_dir: Path) -> ModelsConfig:
    """Load model profiles from prompts/models.yaml."""
    path = models_config_path(prompts_dir)
    if not path.exists() or not path.is_file():
        raise ValueError(
            f"Model profile config not found: {path}. "
            "Run 'ankiops init' to eject built-in AI assets."
        )

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Model profile config must be a YAML mapping: {path}")

    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError(f"Key 'profiles' must be a non-empty mapping in '{path}'")

    profiles: dict[str, ModelProfile] = {}
    for profile_name, profile_raw in raw_profiles.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError(f"Profile names must be non-empty strings in '{path}'")
        profiles[profile_name] = _normalize_profile(
            profile_name.strip(),
            profile_raw,
            source_path=path,
        )

    default_profile = raw.get("default_profile")
    if default_profile is None:
        default_profile = next(iter(profiles.keys()))
    if not isinstance(default_profile, str) or not default_profile.strip():
        raise ValueError(
            f"Key 'default_profile' must be a non-empty string in '{path}'"
        )
    default_profile = default_profile.strip()
    if default_profile not in profiles:
        raise ValueError(
            f"default_profile '{default_profile}' was not found in profiles in '{path}'"
        )

    return ModelsConfig(
        default_profile=default_profile,
        profiles=profiles,
        source_path=path,
    )


def resolve_runtime_ai_config(
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
    """Resolve runtime config from model profiles plus CLI overrides."""
    profile_name = profile or models_config.default_profile
    selected = models_config.profiles.get(profile_name)
    if selected is None:
        known = ", ".join(sorted(models_config.profiles.keys()))
        raise ValueError(f"Unknown model profile '{profile_name}'. Known: {known}")

    runtime_provider = provider or selected.provider
    if runtime_provider not in VALID_PROVIDERS:
        raise ValueError("provider must be one of: local, remote")

    runtime_model = (model or selected.model).strip()
    runtime_base_url = (base_url or selected.base_url).strip()
    runtime_api_key_env = (api_key_env or selected.api_key_env).strip()
    runtime_timeout = (
        selected.timeout_seconds if timeout_seconds is None else timeout_seconds
    )
    runtime_max_in_flight = (
        selected.max_in_flight if max_in_flight is None else max_in_flight
    )

    if not runtime_model:
        raise ValueError("model must be a non-empty string")
    if not runtime_base_url:
        raise ValueError("base_url must be a non-empty string")
    if not runtime_api_key_env:
        runtime_api_key_env = DEFAULT_API_KEY_ENV
    if runtime_timeout <= 0:
        raise ValueError("timeout must be > 0")
    if runtime_max_in_flight <= 0:
        raise ValueError("max_in_flight must be > 0")

    resolved_api_key = api_key or os.environ.get(runtime_api_key_env)
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


def _normalize_profile(
    profile_name: str,
    raw: Any,
    *,
    source_path: Path,
) -> ModelProfile:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Profile '{profile_name}' must be a mapping in '{source_path}'"
        )

    provider = str(raw.get("provider", "")).strip()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Profile '{profile_name}' provider must be one of: "
            f"{', '.join(sorted(VALID_PROVIDERS))}"
        )

    model = str(raw.get("model", "")).strip()
    base_url = str(raw.get("base_url", "")).strip()
    api_key_env = str(raw.get("api_key_env", DEFAULT_API_KEY_ENV)).strip()
    timeout_seconds = _parse_positive_int(
        raw.get("timeout_seconds"),
        fallback=DEFAULT_TIMEOUT_SECONDS,
    )
    max_in_flight = _parse_positive_int(
        raw.get("max_in_flight"),
        fallback=DEFAULT_MAX_IN_FLIGHT,
    )

    if not model:
        raise ValueError(f"Profile '{profile_name}' requires a non-empty 'model'")
    if not base_url:
        raise ValueError(f"Profile '{profile_name}' requires a non-empty 'base_url'")
    if not api_key_env:
        api_key_env = DEFAULT_API_KEY_ENV

    return ModelProfile(
        name=profile_name,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_in_flight=max_in_flight,
    )


def _parse_positive_int(raw: Any, *, fallback: int) -> int:
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback
