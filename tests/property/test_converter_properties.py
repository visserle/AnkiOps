"""Property-based tests for AnkiOps converters."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ankiops.html_to_markdown import HTMLToMarkdown
from ankiops.markdown_to_html import MarkdownToHTML


@pytest.fixture(scope="session")
def md_to_html() -> Any:
    return MarkdownToHTML()


@pytest.fixture(scope="session")
def html_to_md() -> Any:
    return HTMLToMarkdown()


def _assert_does_not_crash(fn: Any, text: str) -> None:
    try:
        fn(text)
    except Exception as exc:  # pragma: no cover - property assertion path
        pytest.fail(f"Crashed on input {text!r}: {exc}")


@given(text=st.text())
def test_markdown_to_html_never_crashes(md_to_html: Any, text: Any) -> None:
    """Markdown converter should not crash on any input string."""
    _assert_does_not_crash(md_to_html.convert, text)


@given(text=st.text())
def test_html_to_markdown_never_crashes(html_to_md: Any, text: Any) -> None:
    """HTML converter should not crash on any input string."""
    _assert_does_not_crash(html_to_md.convert, text)


# Recursive strategy to generate simple markdown-like structures
def markdown_text() -> Any:
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cc", "Cs")), min_size=1
    )


@given(text=markdown_text())
def test_roundtrip_stability(md_to_html: Any, html_to_md: Any, text: Any) -> None:
    """md -> html -> md should ideally preserve content (pseudo-stability).

    Note: Perfect roundtrip is hard because HTML is more expressive or
    normalization happens (e.g. whitespace).
    We check that the *second* roundtrip is stable.
    md1 -> html1 -> md2 -> html2 -> md3
    Expect md2 == md3
    """
    html1 = md_to_html.convert(text)
    md2 = html_to_md.convert(html1)

    html2 = md_to_html.convert(md2)
    md3 = html_to_md.convert(html2)

    # A standalone `>` is intentionally reinterpreted as a blockquote on
    # the next Markdown pass, so it is the one supported non-fixed-point case.
    assert md2 == md3 or (md2 == ">" and md3 == "")
