from decimal import Decimal

from ankiops.llm.llm_models import TaskRunSummary
from ankiops.llm.model_registry import (
    CLAUDE_SONNET_4_6,
    format_supported_model_names,
    format_usd_cents,
    parse_model,
    supported_model_names,
)


def test_parse_model_returns_supported_model_class():
    model = parse_model("claude-sonnet-4-6")

    assert model == CLAUDE_SONNET_4_6
    assert model.api_id == "claude-sonnet-4-6"


def test_supported_model_names_are_stable():
    model_names = supported_model_names()
    claude_model_names = tuple(
        name for name in model_names if name.startswith("claude-")
    )

    assert claude_model_names == (
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    )
    rendered = format_supported_model_names()
    assert "claude-opus-4-6" in rendered
    assert "claude-sonnet-4-6" in rendered
    assert "claude-haiku-4-5" in rendered


def test_model_estimate_cost_uses_current_rates():
    estimate = CLAUDE_SONNET_4_6.estimate_cost(
        input_tokens=2573,
        output_tokens=238,
    )

    assert estimate.input_usd == Decimal("0.007719")
    assert estimate.output_usd == Decimal("0.00357")
    assert estimate.total_usd == Decimal("0.011289")
    assert estimate.format() == "$0.01"


def test_task_run_summary_format_cost_reports_priced_model():
    summary = TaskRunSummary(
        task_name="grammar",
        model=CLAUDE_SONNET_4_6,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert summary.format_cost() == "$18.00"


def test_parse_model_rejects_unknown_values():
    assert parse_model("sonnet") is None
    assert parse_model("unknown-model") is None


def test_format_usd_cents_rounds_to_currency_cents():
    assert format_usd_cents(Decimal("0")) == "$0.00"
    assert format_usd_cents(Decimal("0.003570")) == "$0.00"
    assert format_usd_cents(Decimal("0.011289")) == "$0.01"
