"""Supported model registry with provider routing and optional pricing."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

_TOKENS_PER_MTOK = Decimal("1000000")
_USD_CENTS_QUANTUM = Decimal("0.01")


@dataclass(frozen=True)
class CostEstimate:
    input_usd: Decimal
    output_usd: Decimal

    @property
    def total_usd(self) -> Decimal:
        return self.input_usd + self.output_usd

    def format(self) -> str:
        return format_usd_cents(self.total_usd)


@dataclass(frozen=True)
class ProviderModel:
    name: str
    api_id: str
    provider: str
    base_url: str
    api_key_env: str
    input_usd_per_mtok: Decimal | None = None
    output_usd_per_mtok: Decimal | None = None

    def estimate_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEstimate | None:
        if self.input_usd_per_mtok is None or self.output_usd_per_mtok is None:
            return None
        return CostEstimate(
            input_usd=(
                Decimal(input_tokens) * self.input_usd_per_mtok / _TOKENS_PER_MTOK
            ),
            output_usd=(
                Decimal(output_tokens) * self.output_usd_per_mtok / _TOKENS_PER_MTOK
            ),
        )

    def __str__(self) -> str:
        return self.name


CLAUDE_OPUS_4_6 = ProviderModel(
    name="claude-opus-4-6",
    api_id="claude-opus-4-6",
    provider="anthropic",
    base_url="https://api.anthropic.com/v1/",
    api_key_env="ANTHROPIC_API_KEY",
    input_usd_per_mtok=Decimal("5"),
    output_usd_per_mtok=Decimal("25"),
)
CLAUDE_SONNET_4_6 = ProviderModel(
    name="claude-sonnet-4-6",
    api_id="claude-sonnet-4-6",
    provider="anthropic",
    base_url="https://api.anthropic.com/v1/",
    api_key_env="ANTHROPIC_API_KEY",
    input_usd_per_mtok=Decimal("3"),
    output_usd_per_mtok=Decimal("15"),
)
CLAUDE_HAIKU_4_5 = ProviderModel(
    name="claude-haiku-4-5",
    api_id="claude-haiku-4-5",
    provider="anthropic",
    base_url="https://api.anthropic.com/v1/",
    api_key_env="ANTHROPIC_API_KEY",
    input_usd_per_mtok=Decimal("1"),
    output_usd_per_mtok=Decimal("5"),
)
# OpenAI pricing reference: developers.openai.com/api/docs/pricing (2026-04-12)
GPT_5_4 = ProviderModel(
    name="gpt-5.4",
    api_id="gpt-5.4",
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    input_usd_per_mtok=Decimal("2.5"),
    output_usd_per_mtok=Decimal("15"),
)
GPT_5_4_MINI = ProviderModel(
    name="gpt-5.4-mini",
    api_id="gpt-5.4-mini",
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    input_usd_per_mtok=Decimal("0.75"),
    output_usd_per_mtok=Decimal("4.5"),
)
GPT_5_4_NANO = ProviderModel(
    name="gpt-5.4-nano",
    api_id="gpt-5.4-nano",
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    input_usd_per_mtok=Decimal("0.2"),
    output_usd_per_mtok=Decimal("1.25"),
)

_MODELS = (
    CLAUDE_OPUS_4_6,
    CLAUDE_SONNET_4_6,
    CLAUDE_HAIKU_4_5,
    GPT_5_4,
    GPT_5_4_MINI,
    GPT_5_4_NANO,
)
_MODELS_BY_NAME = {model.name: model for model in _MODELS}


def parse_model(value: str) -> ProviderModel | None:
    return _MODELS_BY_NAME.get(value.strip())


def supported_model_names() -> tuple[str, ...]:
    return tuple(model.name for model in _MODELS)


def format_supported_model_names() -> str:
    return ", ".join(supported_model_names())


def format_usd_cents(amount: Decimal) -> str:
    quantized = amount.quantize(_USD_CENTS_QUANTUM, rounding=ROUND_HALF_UP)
    return f"${quantized:.2f}"
