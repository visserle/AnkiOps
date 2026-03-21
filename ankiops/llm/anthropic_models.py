"""Anthropic model classes, current API ids, and pricing."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

_TOKENS_PER_MTOK = Decimal("1000000")
_USD_CENTS_QUANTUM = Decimal("0.01")
_BATCH_PRICE_MULTIPLIER = Decimal("0.5")


@dataclass(frozen=True)
class CostEstimate:
    input_usd: Decimal
    output_usd: Decimal

    @property
    def total_usd(self) -> Decimal:
        return self.input_usd + self.output_usd

    def format(self) -> str:
        return format_usd_cents(self.total_usd)

    def scale(self, multiplier: Decimal) -> "CostEstimate":
        return CostEstimate(
            input_usd=self.input_usd * multiplier,
            output_usd=self.output_usd * multiplier,
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
        batch: bool = False,
    ) -> CostEstimate:
        estimate = CostEstimate(
            input_usd=(
                Decimal(input_tokens) * self.input_usd_per_mtok / _TOKENS_PER_MTOK
            ),
            output_usd=(
                Decimal(output_tokens) * self.output_usd_per_mtok / _TOKENS_PER_MTOK
            ),
        )
        if batch:
            return estimate.scale(_BATCH_PRICE_MULTIPLIER)
        return estimate

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


def format_usd_cents(amount: Decimal) -> str:
    quantized = amount.quantize(_USD_CENTS_QUANTUM, rounding=ROUND_HALF_UP)
    return f"${quantized:.2f}"
