from __future__ import annotations

from ankiops.llm.model_registry import ModelSpec
from ankiops.llm_v2.domain.capabilities import ModelCapabilities, TransportMode
from ankiops.llm_v2.providers.adapter_factory import create_structured_adapter
from ankiops.llm_v2.providers.anthropic_tool_strict_adapter import (
    AnthropicToolStrictAdapter,
)
from ankiops.llm_v2.providers.gemini_structured_adapter import GeminiStructuredAdapter
from ankiops.llm_v2.providers.openai_compat_adapter import OpenAICompatStructuredAdapter
from ankiops.llm_v2.providers.openai_responses_adapter import (
    OpenAIResponsesStructuredAdapter,
)


def _model(provider: str) -> ModelSpec:
    return ModelSpec(
        model=f"{provider}-model",
        model_id=f"{provider}-model",
        provider=provider,
        base_url="https://api.example.com/v1",
        api_key="$EXAMPLE_API_KEY",
    )


def test_factory_returns_openai_responses_adapter() -> None:
    adapter = create_structured_adapter(
        model=_model("openai"),
        capabilities=ModelCapabilities(
            provider="openai",
            model_id="gpt-5.4",
            transport_mode=TransportMode.OPENAI_RESPONSES_STRUCTURED,
            supports_strict_schema=True,
        ),
        api_key="key",
        timeout_seconds=30,
    )

    assert isinstance(adapter, OpenAIResponsesStructuredAdapter)


def test_factory_returns_openai_compat_adapter() -> None:
    adapter = create_structured_adapter(
        model=_model("openrouter"),
        capabilities=ModelCapabilities(
            provider="openrouter",
            model_id="qwen3-32b",
            transport_mode=TransportMode.OPENAI_COMPAT_JSON_SCHEMA_STRICT,
            supports_strict_schema=True,
        ),
        api_key="key",
        timeout_seconds=30,
    )

    assert isinstance(adapter, OpenAICompatStructuredAdapter)


def test_factory_returns_anthropic_tool_adapter() -> None:
    adapter = create_structured_adapter(
        model=_model("anthropic"),
        capabilities=ModelCapabilities(
            provider="anthropic",
            model_id="claude-sonnet-4-6",
            transport_mode=TransportMode.ANTHROPIC_TOOL_USE_STRICT,
            supports_strict_schema=True,
        ),
        api_key="key",
        timeout_seconds=30,
    )

    assert isinstance(adapter, AnthropicToolStrictAdapter)


def test_factory_returns_gemini_adapter() -> None:
    adapter = create_structured_adapter(
        model=_model("gemini"),
        capabilities=ModelCapabilities(
            provider="gemini",
            model_id="gemini-2.5-pro",
            transport_mode=TransportMode.GEMINI_NATIVE_STRUCTURED,
            supports_strict_schema=True,
        ),
        api_key="key",
        timeout_seconds=30,
    )

    assert isinstance(adapter, GeminiStructuredAdapter)
