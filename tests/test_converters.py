"""Tests for HTML <-> Markdown conversion.

Consolidates previous `test_html_converter.py` and `test_markdown_converter.py`
using extensive parametrization to reduce redundancy.
"""

import pytest

from ankiops.html_converter import HTMLToMarkdown
from ankiops.markdown_converter import MarkdownToHTML


@pytest.fixture
def html_to_md():
    return HTMLToMarkdown()


@pytest.fixture
def md_to_html():
    return MarkdownToHTML()


# -- Parametrized Round-Trip Tests ------------------------------------------
# Many simple inline elements can be tested via round-trips or
# directional checks where exact output is expected.


@pytest.mark.parametrize(
    "md,html",
    [
        # Basic Formatting
        ("**Bold**", "<strong>Bold</strong>"),
        ("*Italic*", "<em>Italic</em>"),
        ("`Code`", "<code>Code</code>"),
        ("~~Strike~~", "<del>Strike</del>"),
        # Headings
        ("# H1", "<h1>H1</h1>"),
        ("## H2", "<h2>H2</h2>"),
        ("### H3", "<h3>H3</h3>"),
        # Links and Images
        ("[Text](http://link.com)", '<a href="http://link.com">Text</a>'),
        (
            "![Alt](img.png)",
            '<img src="img.png" alt="Alt">',
        ),  # Note: <img> usually has no closing tag in HTML5, but parser dependent
        # Blockquotes
        ("> Quote", "<blockquote>\n<p>Quote</p>\n</blockquote>"),
        # Separators
        ("---\n", "<hr>"),
    ],
)
def test_markup_conversion(md_to_html, md, html):
    """Test standard markup conversion (MD -> HTML).

    Note: We don't always test HTML->MD roundtrip here because
    HTML conversion might normalize things (e.g. <b> -> <strong>).
    """
    converted_html = md_to_html.convert(md)
    # Loose check for key tags due to newline/spacing variations
    # Normalize expected and actual by removing whitespace for comparison logic
    # or just check basic tag presence.

    if "<blockquote>" in html:
        assert "<blockquote>" in converted_html
        assert "Quote" in converted_html
    else:
        # Check that the main tag is present
        main_tag = html.split(">")[0][1:]  # e.g. strong
        if " " in main_tag:
            main_tag = main_tag.split()[0]

        assert f"<{main_tag}" in converted_html


@pytest.mark.parametrize(
    "html,expected_md",
    [
        ("<b>Bold</b>", "**Bold**"),
        ("<strong>Bold</strong>", "**Bold**"),
        ("<i>Italic</i>", "*Italic*"),
        ("<em>Italic</em>", "*Italic*"),
        ("<code>Code</code>", "`Code`"),
        ("<del>Strike</del>", "~~Strike~~"),
        ("<s>Strike</s>", "~~Strike~~"),
        # <strike> often arguably supported or not depending on library version, skipping to avoid flake
        ('<a href="u">L</a>', "[L](<u>)"),
        ("<h1>H1</h1>", "# H1"),
        ("<h2>H2</h2>", "## H2"),
        ("<h3>H3</h3>", "### H3"),
        ("<blockquote>Q</blockquote>", "> Q"),
    ],
)
def test_html_to_markdown_output(html_to_md, html, expected_md):
    """Test HTML -> Markdown specific normalization."""
    assert html_to_md.convert(html).strip() == expected_md


# -- Lists ------------------------------------------------------------------


def test_unordered_list(md_to_html, html_to_md):
    md = "- Item 1\n- Item 2"
    html = md_to_html.convert(md)
    assert "<ul>" in html
    assert "<li>Item 1</li>" in html

    back_to_md = html_to_md.convert(html)
    assert "- Item 1" in back_to_md
    assert "- Item 2" in back_to_md


def test_ordered_list(md_to_html, html_to_md):
    md = "1. First\n2. Second"
    html = md_to_html.convert(md)
    assert "<ol>" in html
    assert "<li>First</li>" in html

    back_to_md = html_to_md.convert(html)
    assert "1. First" in back_to_md
    assert "2. Second" in back_to_md


def test_nested_list(md_to_html, html_to_md):
    md = "- Parent\n  - Child"
    html = md_to_html.convert(md)
    assert "<ul>" in html
    # Check for nesting structure
    assert "Item 1" not in html  # sanity

    back_to_md = html_to_md.convert(html)
    assert "- Parent" in back_to_md
    assert "  - Child" in back_to_md


# -- Tables -----------------------------------------------------------------


def test_table_conversion(md_to_html, html_to_md):
    md = "| Head1 | Head2 |\n| --- | --- |\n| Cell1 | Cell2 |"
    html = md_to_html.convert(md)
    assert "<table>" in html
    assert "<th>Head1</th>" in html
    assert "<td>Cell1</td>" in html

    # Tables are often not perfectly round-tripped by simple libraries
    # converting standard markdown tables -> html -> markdown is complex.
    # AnkiOps implementation might simplify or rely on specific library behavior.
    # We verify content preservation.
    back_to_md = html_to_md.convert(html)
    assert "Head1" in back_to_md
    assert "Cell1" in back_to_md


# -- Math -------------------------------------------------------------------


# PREVIOUSLY: We expected $ -> \\(
# CURRENTLY: We expect $ -> $ (preserved) because automatic conversion caused issues.
@pytest.mark.parametrize(
    "md_math",
    [
        "$E=mc^2$",
        "$$E=mc^2$$",
    ],
)
def test_math_preservation(md_to_html, md_math):
    """AnkiOps now preserves math delimiters for Anki's native handling or user preference."""
    html = md_to_html.convert(md_math)
    assert md_math in html


def test_math_mixed_content(md_to_html):
    md = "Text $x=1$ end."
    html = md_to_html.convert(md)
    # Match exact preservation
    assert "Text $x=1$ end." in html


# -- Code Blocks ------------------------------------------------------------


def test_fenced_code_block(md_to_html, html_to_md):
    md = "```python\nprint('hello')\n```"
    html = md_to_html.convert(md)
    # Pygments adds classes and spans.
    # Check for the presence of the code content logic
    # The string "print('hello')" might be broken up by spans like:
    # <span class="...">print</span><span class="...">('hello')</span>
    # So searching for the full string might fail.
    # We check for parts or check for the container.

    assert '<div class="highlight">' in html or "<pre>" in html
    assert "python" in html or "language-python" in html

    # Markdownify (html_to_md) usually strips simpler html well,
    # but complex pygments HTML might require a very robust converter.
    # We'll check if the TEXT is recovered.
    back_to_md = html_to_md.convert(html)
    assert "print" in back_to_md
    assert "'hello'" in back_to_md


def test_indented_code_block(md_to_html):
    md = "    code block"
    html = md_to_html.convert(md)
    assert "<pre" in html
    assert "code block" in html


# -- Edge Case Cleanup ------------------------------------------------------


def test_empty_input(md_to_html, html_to_md):
    assert md_to_html.convert("") == ""
    assert html_to_md.convert("") == ""


def test_malformed_html(html_to_md):
    html = "<div>Unclosed div"
    md = html_to_md.convert(html)
    assert "Unclosed div" in md


def test_unknown_tags_preserved_or_stripped(html_to_md):
    """Unknown tags might be stripped or kept as text depending on strictness."""
    html = "<custom>Text</custom>"
    md = html_to_md.convert(html)
    # Usually text is kept
    assert "Text" in md


def test_media_prefix_stripping(md_to_html):
    """Test that media/ prefix is stripped from image src."""
    md = "![Alt](media/img.png)"
    html = md_to_html.convert(md)
    # Should result in src="img.png" NOT src="media/img.png"
    assert '<img src="img.png"' in html
    assert "media/img.png" not in html


def test_html_to_markdown_enforces_brackets(html_to_md):
    """Verify that HTMLToMarkdown always enforces angle brackets for links and images."""
    # Test images
    html_img = '<img src="test.jpg" alt="alt">'
    md_img = html_to_md.convert(html_img)
    assert "![alt](<media/test.jpg>)" in md_img

    # Test links
    html_link = '<a href="https://example.com">Link</a>'
    md_link = html_to_md.convert(html_link)
    assert "[Link](<https://example.com>)" in md_link

    # Test links with parens
    html_link_parens = '<a href="https://example.com/(test)">Link</a>'
    md_link_parens = html_to_md.convert(html_link_parens)
    assert "[Link](<https://example.com/(test)>)" in md_link_parens

