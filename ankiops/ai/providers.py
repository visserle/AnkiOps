"""Provider registry for AI runtimes."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import AIConfigError


@dataclass(frozen=True)
class ProviderSpec:
    """Canonical provider capabilities and defaults."""

    id: str
    transport: str
    default_base_url: str
    requires_api_key: bool
    default_api_key_env: str | None


_PROVIDER_SPECS = {
    "ollama": ProviderSpec(
        id="ollama",
        transport="openai_chat_completions",
        default_base_url="http://localhost:11434/v1",
        requires_api_key=False,
        default_api_key_env=None,
    ),
    "openai": ProviderSpec(
        id="openai",
        transport="openai_chat_completions",
        default_base_url="https://api.openai.com/v1",
        requires_api_key=True,
        default_api_key_env="OPENAI_API_KEY",
    ),
    "groq": ProviderSpec(
        id="groq",
        transport="openai_chat_completions",
        default_base_url="https://api.groq.com/openai/v1",
        requires_api_key=True,
        default_api_key_env="GROQ_API_KEY",
    ),
}


def provider_ids() -> tuple[str, ...]:
    """Return supported provider ids."""
    return tuple(sorted(_PROVIDER_SPECS.keys()))


def get_provider_spec(provider_id: str) -> ProviderSpec:
    """Return canonical provider spec or raise config error."""
    normalized = provider_id.strip().lower()
    provider = _PROVIDER_SPECS.get(normalized)
    if provider is None:
        allowed = ", ".join(provider_ids())
        raise AIConfigError(f"provider must be one of: {allowed}")
    return provider
