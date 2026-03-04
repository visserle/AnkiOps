"""Anthropic model classes, current API ids, and pricing."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

_TOKENS_PER_MTOK = Decimal("1000000")
_USD_DISPLAY_QUANTUM = Decimal("0.000001")
_MIN_DISPLAY_DECIMALS = 4


@dataclass(frozen=True)
class CostEstimate:
    input_usd: Decimal
    output_usd: Decimal

    @property
    def total_usd(self) -> Decimal:
        return self.input_usd + self.output_usd

    def format(self) -> str:
        return (
            f"LLM estimated cost: {format_usd(self.total_usd)} "
            f"({format_usd(self.input_usd)} input + "
            f"{format_usd(self.output_usd)} output)"
        )


@dataclass(frozen=True)
class AnthropicModel:
    name: str
    api_id: str
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal

    def estimate_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEstimate:
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


OPUS = AnthropicModel(
    name="opus",
    api_id="claude-opus-4-6",
    input_usd_per_mtok=Decimal("5"),
    output_usd_per_mtok=Decimal("25"),
)
SONNET = AnthropicModel(
    name="sonnet",
    api_id="claude-sonnet-4-6",
    input_usd_per_mtok=Decimal("3"),
    output_usd_per_mtok=Decimal("15"),
)
HAIKU = AnthropicModel(
    name="haiku",
    api_id="claude-haiku-4-5",
    input_usd_per_mtok=Decimal("1"),
    output_usd_per_mtok=Decimal("5"),
)

_MODELS = (OPUS, SONNET, HAIKU)
_MODELS_BY_NAME = {model.name: model for model in _MODELS}


def parse_model(value: str) -> AnthropicModel | None:
    return _MODELS_BY_NAME.get(value.strip().lower())


def supported_model_names() -> tuple[str, ...]:
    return tuple(model.name for model in _MODELS)


def format_supported_model_names() -> str:
    return ", ".join(supported_model_names())


def format_usd(amount: Decimal) -> str:
    quantized = amount.quantize(_USD_DISPLAY_QUANTUM, rounding=ROUND_HALF_UP)
    whole, fraction = f"{quantized:f}".split(".")
    fraction = fraction.rstrip("0")
    if len(fraction) < _MIN_DISPLAY_DECIMALS:
        fraction = fraction.ljust(_MIN_DISPLAY_DECIMALS, "0")
    return f"${whole}.{fraction}"
