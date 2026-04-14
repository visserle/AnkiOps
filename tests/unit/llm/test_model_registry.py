from decimal import Decimal
from textwrap import dedent

import pytest

from ankiops.llm.model_registry import (
    ModelRegistryError,
    format_supported_models,
    format_usd_cents,
    load_model_registry,
    parse_model,
)
from ankiops.llm.task_types import TaskRunSummary


def _write_models_file(tmp_path, content: str) -> None:
    (tmp_path / "llm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "llm/models.yaml").write_text(
        dedent(content).strip() + "\n",
        encoding="utf-8",
    )


def test_parse_model_returns_model_from_registry(tmp_path):
    _write_models_file(
        tmp_path,
        """
        - model: claude-sonnet-4-6
          model_id: claude-sonnet-4-6
          provider: anthropic
          base_url: https://api.anthropic.com/v1/
          api_key: $ANTHROPIC_API_KEY
          input_usd_per_mtok: 3
          output_usd_per_mtok: 15
        """,
    )

    model = parse_model("claude-sonnet-4-6", collection_dir=tmp_path)

    assert model is not None
    assert model.model == "claude-sonnet-4-6"
    assert model.model_id == "claude-sonnet-4-6"


def test_format_supported_models_comes_from_registry(tmp_path):
    _write_models_file(
        tmp_path,
        "- model: local-a\n"
        "  model_id: local-a\n"
        "  provider: local\n"
        "  base_url: https://localhost/v1\n"
        "  api_key: $LOCAL_A_KEY\n"
        "- model: local-b\n"
        "  model_id: local-b\n"
        "  provider: local\n"
        "  base_url: https://localhost/v1\n"
        "  api_key: $LOCAL_B_KEY\n",
    )

    rendered = format_supported_models(collection_dir=tmp_path)
    assert rendered == "local-a, local-b"


def test_model_estimate_cost_uses_registry_rates(tmp_path):
    _write_models_file(
        tmp_path,
        """
        - model: claude-sonnet-4-6
          model_id: claude-sonnet-4-6
          provider: anthropic
          base_url: https://api.anthropic.com/v1/
          api_key: $ANTHROPIC_API_KEY
          input_usd_per_mtok: 3
          output_usd_per_mtok: 15
        """,
    )
    model = parse_model("claude-sonnet-4-6", collection_dir=tmp_path)
    assert model is not None

    estimate = model.estimate_cost(
        input_tokens=2573,
        output_tokens=238,
    )
    assert estimate is not None

    assert estimate.input_usd == Decimal("0.007719")
    assert estimate.output_usd == Decimal("0.00357")
    assert estimate.total_usd == Decimal("0.011289")
    assert estimate.format() == "$0.01"


def test_task_run_summary_format_cost_reports_priced_model(tmp_path):
    _write_models_file(
        tmp_path,
        """
        - model: claude-sonnet-4-6
          model_id: claude-sonnet-4-6
          provider: anthropic
          base_url: https://api.anthropic.com/v1/
          api_key: $ANTHROPIC_API_KEY
          input_usd_per_mtok: 3
          output_usd_per_mtok: 15
        """,
    )
    model = parse_model("claude-sonnet-4-6", collection_dir=tmp_path)
    assert model is not None

    summary = TaskRunSummary(
        task_name="grammar",
        model=model,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert summary.format_cost() == "$18.00"


def test_parse_model_rejects_unknown_values(tmp_path):
    _write_models_file(
        tmp_path,
        """
        - model: claude-sonnet-4-6
          model_id: claude-sonnet-4-6
          provider: anthropic
          base_url: https://api.anthropic.com/v1/
          api_key: $ANTHROPIC_API_KEY
        """,
    )

    assert parse_model("sonnet", collection_dir=tmp_path) is None
    assert parse_model("unknown-model", collection_dir=tmp_path) is None


def test_load_model_registry_rejects_missing_registry(tmp_path):
    with pytest.raises(ModelRegistryError, match="model registry file not found"):
        load_model_registry(collection_dir=tmp_path)


def test_parse_model_uses_collection_local_registry(tmp_path):
    _write_models_file(
        tmp_path,
        """
        - model: qwen3-32b
          model_id: qwen3-32b
          provider: openai-compatible
          base_url: https://api.example.com/v1
          api_key: $EXAMPLE_API_KEY
        """,
    )

    model = parse_model("qwen3-32b", collection_dir=tmp_path)

    assert model is not None
    assert model.model == "qwen3-32b"
    assert model.base_url == "https://api.example.com/v1"
    assert model.api_key == "$EXAMPLE_API_KEY"


def test_load_model_registry_rejects_invalid_registry(tmp_path):
    _write_models_file(
        tmp_path,
        """
        - model: bad
          model_id: bad
          provider: local
          base_url: https://localhost/v1
        """,
    )

    with pytest.raises(ModelRegistryError, match="api_key"):
        load_model_registry(collection_dir=tmp_path)


def test_format_usd_cents_rounds_to_currency_cents():
    assert format_usd_cents(Decimal("0")) == "$0.00"
    assert format_usd_cents(Decimal("0.003570")) == "$0.00"
    assert format_usd_cents(Decimal("0.011289")) == "$0.01"
