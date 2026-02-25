"""High-level AI run preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model_profiles import load_model_profiles, resolve_runtime_config
from .paths import AIPaths
from .prompt_loader import load_prompt
from .types import PromptConfig, RuntimeAIConfig


@dataclass(frozen=True)
class AIRuntimeOverrides:
    """Runtime override values from CLI/user input."""

    profile: str | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int | None = None
    max_in_flight: int | None = None
    api_key: str | None = None


def prepare_ai_run(
    collection_dir: Path,
    prompt_ref: str,
    *,
    overrides: AIRuntimeOverrides | None = None,
) -> tuple[PromptConfig, RuntimeAIConfig]:
    """Resolve prompt config and runtime config for one AI prompt run."""
    runtime_overrides = overrides or AIRuntimeOverrides()
    ai_paths = AIPaths.from_collection_dir(collection_dir)
    prompt_config = load_prompt(ai_paths, prompt_ref)
    models_config = load_model_profiles(ai_paths)

    runtime_config = resolve_runtime_config(
        models_config,
        profile=runtime_overrides.profile or prompt_config.model_profile,
        provider=runtime_overrides.provider,
        model=runtime_overrides.model,
        base_url=runtime_overrides.base_url,
        api_key_env=runtime_overrides.api_key_env,
        timeout_seconds=runtime_overrides.timeout_seconds,
        max_in_flight=runtime_overrides.max_in_flight,
        api_key=runtime_overrides.api_key,
    )
    return prompt_config, runtime_config
