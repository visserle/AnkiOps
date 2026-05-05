"""Model transport capability definitions for runtime v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import CapabilityError


class TransportMode(Enum):
    OPENAI_RESPONSES_STRUCTURED = "openai_responses_structured"
    OPENAI_COMPAT_JSON_SCHEMA_STRICT = "openai_compat_json_schema_strict"
    ANTHROPIC_TOOL_USE_STRICT = "anthropic_tool_use_strict"
    GEMINI_NATIVE_STRUCTURED = "gemini_native_structured"


@dataclass(frozen=True)
class ModelCapabilities:
    provider: str
    model_id: str
    transport_mode: TransportMode
    supports_strict_json: bool


def resolve_model_capabilities(
    *,
    provider: str,
    model_id: str,
    transport_mode: TransportMode | None = None,
    supports_strict_json: bool | None = None,
) -> ModelCapabilities:
    normalized_provider = provider.strip().lower()
    resolved_transport = transport_mode or _default_transport(normalized_provider)
    resolved_strict_json = (
        supports_strict_json
        if supports_strict_json is not None
        else _default_strict_json_support(normalized_provider)
    )

    capabilities = ModelCapabilities(
        provider=normalized_provider,
        model_id=model_id,
        transport_mode=resolved_transport,
        supports_strict_json=resolved_strict_json,
    )
    require_strict_json_support(capabilities)
    return capabilities


def require_strict_json_support(capabilities: ModelCapabilities) -> None:
    if not capabilities.supports_strict_json:
        raise CapabilityError(
            "Model does not support strict structured JSON output"
        )


def _default_transport(provider: str) -> TransportMode:
    if provider == "openai":
        return TransportMode.OPENAI_RESPONSES_STRUCTURED
    if provider == "anthropic":
        return TransportMode.ANTHROPIC_TOOL_USE_STRICT
    if provider in {"gemini", "google", "google_ai_studio"}:
        return TransportMode.GEMINI_NATIVE_STRUCTURED
    return TransportMode.OPENAI_COMPAT_JSON_SCHEMA_STRICT


def _default_strict_json_support(provider: str) -> bool:
    return provider in {
        "openai",
        "openai-compatible",
        "openai_compatible",
        "anthropic",
        "gemini",
        "google",
        "google_ai_studio",
        "openrouter",
        "groq",
        "xai",
        "ollama",
    }
