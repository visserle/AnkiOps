from __future__ import annotations

import pytest

from ankiops.llm_v2.domain.capabilities import (
    TransportMode,
    resolve_model_capabilities,
)
from ankiops.llm_v2.domain.errors import CapabilityError


def test_resolve_openai_capabilities_defaults_to_responses_structured() -> None:
    capabilities = resolve_model_capabilities(
        provider="openai",
        model_id="gpt-5.4",
    )

    assert capabilities.provider == "openai"
    assert capabilities.transport_mode is TransportMode.OPENAI_RESPONSES_STRUCTURED
    assert capabilities.supports_strict_json is True


def test_resolve_anthropic_capabilities_defaults_to_tool_use() -> None:
    capabilities = resolve_model_capabilities(
        provider="anthropic",
        model_id="claude-sonnet-4-6",
    )

    assert capabilities.transport_mode is TransportMode.ANTHROPIC_TOOL_USE_STRICT
    assert capabilities.supports_strict_json is True


def test_resolve_gemini_capabilities_defaults_to_native_structured() -> None:
    capabilities = resolve_model_capabilities(
        provider="gemini",
        model_id="gemini-2.5-pro",
    )

    assert capabilities.transport_mode is TransportMode.GEMINI_NATIVE_STRUCTURED
    assert capabilities.supports_strict_json is True


def test_resolve_unknown_provider_requires_explicit_strict_json_support() -> None:
    with pytest.raises(CapabilityError, match="strict structured JSON"):
        resolve_model_capabilities(
            provider="legacy-provider",
            model_id="legacy-model",
        )


def test_explicit_non_strict_model_is_rejected() -> None:
    with pytest.raises(CapabilityError, match="strict structured JSON"):
        resolve_model_capabilities(
            provider="openai",
            model_id="gpt-5.4",
            supports_strict_json=False,
        )
