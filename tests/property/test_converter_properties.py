"""Property-based tests for AnkiOps converters."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ankiops.html_converter import HTMLToMarkdown
from ankiops.markdown_converter import MarkdownToHTML


@pytest.fixture(scope="session")
def md_to_html():
    return MarkdownToHTML()


@pytest.fixture(scope="session")
def html_to_md():
    return HTMLToMarkdown()


def _assert_does_not_crash(fn, text: str) -> None:
    try:
        fn(text)
    except Exception as exc:  # pragma: no cover - property assertion path
        pytest.fail(f"Crashed on input {text!r}: {exc}")


@given(text=st.text())
def test_markdown_to_html_never_crashes(md_to_html, text):
    """Markdown converter should not crash on any input string."""
    _assert_does_not_crash(md_to_html.convert, text)


@given(text=st.text())
def test_html_to_markdown_never_crashes(html_to_md, text):
    """HTML converter should not crash on any input string."""
    _assert_does_not_crash(html_to_md.convert, text)


# Recursive strategy to generate simple markdown-like structures
def markdown_text():
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cc", "Cs")), min_size=1
    )


@given(text=markdown_text())
def test_roundtrip_stability(md_to_html, html_to_md, text):
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

    assert md2 == md3
