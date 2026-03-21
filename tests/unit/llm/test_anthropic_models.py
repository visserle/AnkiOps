from decimal import Decimal

from ankiops.llm.anthropic_models import (
    SONNET,
    format_supported_model_names,
    format_usd_cents,
    parse_model,
    supported_model_names,
)


def test_parse_model_returns_supported_model_class():
    model = parse_model("Sonnet")

    assert model == SONNET
    assert model.api_id == "claude-sonnet-4-6"


def test_supported_model_names_are_stable():
    assert supported_model_names() == ("opus", "sonnet", "haiku")
    assert format_supported_model_names() == "opus, sonnet, haiku"


def test_model_estimate_cost_uses_current_rates():
    estimate = SONNET.estimate_cost(
        input_tokens=2573,
        output_tokens=238,
    )

    assert estimate.input_usd == Decimal("0.007719")
    assert estimate.output_usd == Decimal("0.00357")
    assert estimate.total_usd == Decimal("0.011289")
    assert estimate.format() == "$0.01"


def test_parse_model_rejects_unknown_values():
    assert parse_model("claude-sonnet-4-6") is None
    assert parse_model("gpt-5") is None


def test_format_usd_cents_rounds_to_currency_cents():
    assert format_usd_cents(Decimal("0")) == "$0.00"
    assert format_usd_cents(Decimal("0.003570")) == "$0.00"
    assert format_usd_cents(Decimal("0.011289")) == "$0.01"
