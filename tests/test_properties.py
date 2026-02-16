"""Property-based tests for AnkiOps converters."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ankiops.html_converter import HTMLToMarkdown
from ankiops.markdown_converter import MarkdownToHTML

# Initialize converters
md_to_html = MarkdownToHTML()
html_to_md = HTMLToMarkdown()


@given(st.text())
def test_markdown_to_html_never_crashes(text):
    """Markdown converter should not crash on any input string."""
    try:
        md_to_html.convert(text)
    except Exception as e:
        pytest.fail(f"Crashed on input {repr(text)}: {e}")


@given(st.text())
def test_html_to_markdown_never_crashes(text):
    """HTML converter should not crash on any input string."""
    try:
        html_to_md.convert(text)
    except Exception as e:
        pytest.fail(f"Crashed on input {repr(text)}: {e}")


# Recursive strategy to generate simple markdown-like structures
def markdown_text():
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cc", "Cs")), min_size=1
    )


@given(markdown_text())
def test_roundtrip_stability(text):
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
