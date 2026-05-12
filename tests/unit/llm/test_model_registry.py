"""Tests for LLM model registry parsing."""

from __future__ import annotations

import pytest

from ankiops.llm.model_registry import ModelRegistryError, _parse_registry


def test_model_registry_requires_responses_api_url(tmp_path):
    path = tmp_path / "_models.yaml"
    path.write_text(
        """
- model: test
  model_id: test
  api_url: https://api.openai.com/v1/chat/completions
  api_key: $OPENAI_API_KEY
""",
        encoding="utf-8",
    )

    with pytest.raises(ModelRegistryError, match="Responses API URL"):
        _parse_registry(path)


def test_model_registry_accepts_responses_api_url_with_trailing_slash(tmp_path):
    path = tmp_path / "_models.yaml"
    path.write_text(
        """
- model: test
  model_id: test
  api_url: https://api.openai.com/v1/responses/
  api_key: $OPENAI_API_KEY
""",
        encoding="utf-8",
    )

    registry = _parse_registry(path)

    assert registry.parse("test").api_url == "https://api.openai.com/v1/responses"
