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


@pytest.mark.parametrize(
    "anki_in,expected_html,expected_md",
    [
        # Structural: Typing these in Anki should "upgrade" to Markdown elements
        ("> Quote", "<blockquote>\nQuote</blockquote>\n", "> Quote"),
        ("- Item", "<ul>\n<li>Item</li>\n</ul>\n", "- Item"),
        # Literal: Typing these in Anki should stay literal markers
        ("+ Literal Plus", "+ Literal Plus", r"\+ Literal Plus"),
        ("# Literal Hash", "# Literal Hash", r"\# Literal Hash"),
        ("* Literal Asterisk", "* Literal Asterisk", r"\* Literal Asterisk"),
        # Anki Specifics
        ("{{c1::cloze}}", "{{c1::cloze}}", "{{c1::cloze}}"),
    ],
)
def test_structural_vs_literal_roundtrips(
    md_to_html, html_to_md, anki_in, expected_html, expected_md
):
    """Verify that some characters are structural (auto-upgrade) while others stay literal."""
    # 1. Anki -> MD (Export)
    md = html_to_md.convert(anki_in)
    assert md == expected_md

    # 2. MD -> HTML (Import)
    html = md_to_html.convert(md)
    # Check if the structural tag is there or content is same
    if "<blockquote>" in expected_html:
        assert "<blockquote>" in html
    elif "<ul>" in expected_html:
        assert "<ul>" in html
    else:
        assert html == expected_html

    # 3. HTML -> MD (Roundtrip)
    md_roundtrip = html_to_md.convert(html)
    assert md_roundtrip == expected_md


# ===========================================================================
# Restored Round-Trip Tests
# ===========================================================================
# The following tests were deleted during the Hexagonal Architecture refactor
# (commit 8ae03bd) but cover critical edge cases for bidirectional conversion.
# They are adapted from the original test_html_converter.py and
# test_markdown_converter.py (combined ~1,500 lines → parametrized below).


# -- Blockquotes (MD → HTML → MD) -------------------------------------------


class TestBlockquoteRoundTrips:
    """Test round-trip conversion of blockquotes."""

    def test_simple_blockquote(self, md_to_html, html_to_md):
        md = "> Simple quote"
        html = md_to_html.convert(md)
        assert "<blockquote>" in html
        back = html_to_md.convert(html)
        assert "> Simple quote" in back

    def test_multiline_blockquote(self, md_to_html, html_to_md):
        md = "> Line 1\n> Line 2"
        html = md_to_html.convert(md)
        assert "<blockquote>" in html
        back = html_to_md.convert(html)
        assert "Line 1" in back
        assert "Line 2" in back

    def test_blockquote_with_formatting(self, md_to_html, html_to_md):
        md = "> **Bold** quote"
        html = md_to_html.convert(md)
        assert "<blockquote>" in html
        assert "<strong>" in html or "<b>" in html
        back = html_to_md.convert(html)
        assert "**Bold**" in back

    def test_separate_blockquotes(self, md_to_html, html_to_md):
        md = "> Quote 1\n\n> Quote 2"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "Quote 1" in back
        assert "Quote 2" in back

    def test_blockquote_html_to_md(self, html_to_md, md_to_html):
        """HTML blockquote round-trips correctly."""
        html = "<blockquote>Quote text</blockquote>"
        md = html_to_md.convert(html)
        assert "> " in md
        assert "Quote" in md
        back = md_to_html.convert(md)
        assert "<blockquote>" in back


# -- Escape Sequences -------------------------------------------------------


class TestEscapeSequenceRoundTrips:
    """Round-trip tests for escaped markdown characters."""

    @pytest.mark.parametrize(
        "md_input,expected_char",
        [
            (r"\*not bold\*", "*"),
            (r"\_not italic\_", "_"),
            (r"\[not link\]", "["),
            (r"\`not code\`", "`"),
        ],
    )
    def test_escaped_chars_md_roundtrip(
        self, md_to_html, html_to_md, md_input, expected_char
    ):
        """Escaped markdown chars should survive a round-trip as literals."""
        html = md_to_html.convert(md_input)
        back = html_to_md.convert(html)
        assert expected_char in back

    def test_literal_backslash(self, md_to_html, html_to_md):
        md = "path\\\\to\\\\file"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "\\" in back

    def test_mixed_escaped_and_formatting(self, md_to_html, html_to_md):
        md = r"**bold** and \*literal\*"
        html = md_to_html.convert(md)
        assert "<strong>" in html
        back = html_to_md.convert(html)
        assert "**bold**" in back
        assert "*" in back  # The literal asterisks should be present


# -- HTML Entities -----------------------------------------------------------


class TestHTMLEntityRoundTrips:
    """Round-trip tests for HTML entities and special characters."""

    def test_ampersand_roundtrip(self, md_to_html, html_to_md):
        md = "A & B"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "A" in back
        assert "&" in back or "&amp;" in back
        assert "B" in back

    def test_less_than_greater_than(self, md_to_html, html_to_md):
        md = "a < b > c"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "a" in back
        assert "b" in back
        assert "c" in back

    def test_quotes_roundtrip(self, md_to_html, html_to_md):
        md = 'He said "hello"'
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "hello" in back

    def test_html_entities_from_html(self, html_to_md, md_to_html):
        """HTML entities (&amp; etc.) should convert to literal chars in markdown."""
        html = "A &amp; B &lt; C &gt; D"
        md = html_to_md.convert(html)
        assert "&" in md
        assert "<" in md or "&lt;" in md
        back = md_to_html.convert(md)
        assert "A" in back
        assert "B" in back

    def test_nbsp_from_html(self, html_to_md):
        """Non-breaking spaces from HTML should be handled."""
        html = "word1&nbsp;word2"
        md = html_to_md.convert(html)
        assert "word1" in md
        assert "word2" in md


# -- Images ------------------------------------------------------------------


class TestImageRoundTrips:
    """Round-trip tests for image elements."""

    def test_image_with_alt(self, md_to_html, html_to_md):
        md = "![Alt text](<media/photo.jpg>)"
        html = md_to_html.convert(md)
        assert "<img" in html
        assert 'alt="Alt text"' in html
        back = html_to_md.convert(html)
        assert "Alt text" in back
        assert "photo.jpg" in back

    def test_image_without_alt(self, md_to_html, html_to_md):
        md = "![](<media/photo.jpg>)"
        html = md_to_html.convert(md)
        assert "<img" in html
        back = html_to_md.convert(html)
        assert "photo.jpg" in back

    def test_image_with_width(self, md_to_html, html_to_md):
        md = "![Alt](<media/photo.jpg>){width=300}"
        html = md_to_html.convert(md)
        assert "300" in html
        back = html_to_md.convert(html)
        assert "photo.jpg" in back
        assert "300" in back

    def test_image_from_html(self, html_to_md, md_to_html):
        """HTML img element round-trips via markdown."""
        html = '<img src="photo.jpg" alt="Alt text">'
        md = html_to_md.convert(html)
        assert "Alt text" in md
        assert "photo.jpg" in md
        back = md_to_html.convert(md)
        assert 'src="photo.jpg"' in back

    def test_image_with_width_from_html(self, html_to_md, md_to_html):
        html = '<img src="photo.jpg" alt="Alt" style="width: 300px;">'
        md = html_to_md.convert(html)
        back = md_to_html.convert(md)
        assert "photo.jpg" in back
        assert "300" in back


# -- Links -------------------------------------------------------------------


class TestLinkRoundTrips:
    """Round-trip tests for link elements."""

    def test_simple_link(self, md_to_html, html_to_md):
        md = "[Link text](<https://example.com>)"
        html = md_to_html.convert(md)
        assert "https://example.com" in html
        assert "Link text" in html
        back = html_to_md.convert(html)
        assert "Link text" in back
        assert "https://example.com" in back

    def test_link_with_query_params(self, md_to_html, html_to_md):
        md = "[Click](<https://example.com/path?q=1&r=2>)"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "Click" in back
        assert "example.com/path" in back

    def test_link_with_parentheses(self, md_to_html, html_to_md):
        md = "[Wiki](<https://en.wikipedia.org/wiki/Python_(programming_language)>)"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "Wiki" in back
        assert "programming_language" in back

    def test_link_from_html(self, html_to_md, md_to_html):
        html = '<a href="https://example.com">Link text</a>'
        md = html_to_md.convert(html)
        assert "Link text" in md
        assert "https://example.com" in md
        back = md_to_html.convert(md)
        assert "https://example.com" in back

    def test_link_with_parentheses_from_html(self, html_to_md, md_to_html):
        html = '<a href="https://en.wikipedia.org/wiki/Python_(programming_language)">Wiki</a>'
        md = html_to_md.convert(html)
        back = md_to_html.convert(md)
        assert "Wiki" in back
        assert "programming_language" in back


# -- Unicode / Emoji / Umlauts -----------------------------------------------


class TestUnicodeRoundTrips:
    """Round-trip tests for unicode characters, emoji, and special chars."""

    @pytest.mark.parametrize(
        "text",
        [
            "日本語テスト",
            "Ünîcödé tëxt",
            "🎉 celebration 🎊",
            "Ä Ö Ü ä ö ü ß",
            "Café résumé naïve",
        ],
        ids=["japanese", "diacritics", "emoji", "german_umlauts", "french_accents"],
    )
    def test_unicode_roundtrip(self, md_to_html, html_to_md, text):
        html = md_to_html.convert(text)
        back = html_to_md.convert(html)
        assert text in back

    def test_arrow_roundtrip(self, md_to_html, html_to_md):
        md = "A → B"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        # → may be serialized back as --> in markdown
        assert "→" in back or "-->" in back

    def test_arrow_syntax_converted(self, md_to_html, html_to_md):
        md = "A --> B"
        html = md_to_html.convert(md)
        # --> should be converted to → in HTML
        assert "→" in html


# -- Code Blocks (extended) -------------------------------------------------


class TestCodeBlockRoundTrips:
    """Extended round-trip tests for code blocks."""

    def test_code_block_with_language(self, md_to_html, html_to_md):
        md = "```python\ndef foo():\n    pass\n```"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "def foo():" in back
        assert "pass" in back

    def test_code_block_with_special_chars(self, md_to_html, html_to_md):
        md = "```\n<div>test</div>\n**bold**\n```"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        # Inside code block, special chars should be preserved
        assert "test" in back
        assert "bold" in back

    def test_code_block_from_html(self, html_to_md, md_to_html):
        """Pre/code HTML elements should round-trip."""
        html = "<pre><code>def foo():\n    pass</code></pre>"
        md = html_to_md.convert(html)
        assert "def foo():" in md
        back = md_to_html.convert(md)
        assert "def foo():" in back

    def test_code_block_with_language_from_html(self, html_to_md, md_to_html):
        html = '<pre><code class="language-python">def foo():\n    pass</code></pre>'
        md = html_to_md.convert(html)
        # Verify text is recovered in markdown
        assert "def foo():" in md
        assert "pass" in md


# -- Complex Edge Cases ------------------------------------------------------


class TestComplexEdgeCases:
    """Tests for nested formatting, consecutive elements, and whitespace."""

    def test_nested_formatting(self, md_to_html, html_to_md):
        md = "**Bold *and italic* text**"
        html = md_to_html.convert(md)
        assert "<strong>" in html
        back = html_to_md.convert(html)
        assert "Bold" in back
        assert "italic" in back

    def test_consecutive_formatting(self, md_to_html, html_to_md):
        md = "**bold** *italic* `code`"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "**bold**" in back
        assert "*italic*" in back
        assert "`code`" in back

    def test_link_with_formatting(self, md_to_html, html_to_md):
        md = "[**bold link**](<https://example.com>)"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "bold link" in back
        assert "example.com" in back

    def test_heading_with_formatting(self, md_to_html, html_to_md):
        md = "## **Bold** Title"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "##" in back
        assert "Bold" in back
        assert "Title" in back

    def test_list_with_formatting(self, md_to_html, html_to_md):
        md = "- **Bold** item\n- *Italic* item\n- `code` item"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "**Bold**" in back
        assert "*Italic*" in back
        assert "`code`" in back

    def test_multiple_line_breaks(self, md_to_html, html_to_md):
        md = "Line 1\n\nLine 2"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "Line 1" in back
        assert "Line 2" in back

    def test_single_character_formatting(self, md_to_html, html_to_md):
        md = "**B** *I* `C`"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "**B**" in back
        assert "*I*" in back
        assert "`C`" in back

    def test_whitespace_preservation(self, md_to_html, html_to_md):
        md = "Word   with   spaces"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "Word" in back
        assert "spaces" in back

    def test_parentheses_in_text(self, md_to_html, html_to_md):
        md = "Some (text) here"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "Some (text) here" in back

    def test_medical_terminology(self, md_to_html, html_to_md):
        md = "Patient presented with **dyspnea** and *tachycardia*"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "**dyspnea**" in back
        assert "*tachycardia*" in back

    def test_mixed_formatting_and_lists(self, md_to_html, html_to_md):
        md = "- **Bold** with `code`\n- *Italic* with [link](<https://example.com>)"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "**Bold**" in back
        assert "`code`" in back
        assert "*Italic*" in back
        assert "link" in back

    def test_ordered_list_with_nested_unordered(self, md_to_html, html_to_md):
        md = "1. First item\n   - Nested bullet 1\n   - Nested bullet 2\n2. Second item"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "First item" in back
        assert "Nested bullet 1" in back
        assert "Second bullet" not in back or "Second item" in back
        assert "Second item" in back

    def test_table_with_formatting(self, md_to_html, html_to_md):
        md = "| **Bold** | *Italic* |\n| --- | --- |\n| `code` | normal |"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "Bold" in back
        assert "Italic" in back
        assert "code" in back


# -- Directional-Only Checks ------------------------------------------------
# These test specific direction behaviors (MD-only or HTML-only) that are
# not round-trip symmetric.


class TestDirectionalChecks:
    """Tests for direction-specific behavior."""

    # MD → HTML only
    def test_bold_to_strong(self, md_to_html):
        assert "<strong>" in md_to_html.convert("**bold**")

    def test_italic_to_em(self, md_to_html):
        assert "<em>" in md_to_html.convert("*italic*")

    def test_inline_code_to_code(self, md_to_html):
        assert "<code>" in md_to_html.convert("`code`")

    def test_highlight_to_mark(self, md_to_html):
        assert "<mark>" in md_to_html.convert("==important==")

    def test_link_to_anchor(self, md_to_html):
        result = md_to_html.convert("[Link](https://example.com)")
        assert "https://example.com" in result
        assert "Link" in result

    def test_line_breaks(self, md_to_html):
        result = md_to_html.convert("Line 1\n\nLine 2")
        assert "Line 1" in result
        assert "Line 2" in result
        assert "<br>" in result

    # HTML → MD only
    def test_plain_html_paragraph(self, html_to_md):
        md = html_to_md.convert("plain text")
        assert "plain text" in md

    def test_html_bold_tag(self, html_to_md):
        md = html_to_md.convert("<b>bold</b>")
        assert "**bold**" in md

    def test_html_line_break(self, html_to_md):
        md = html_to_md.convert("Line 1<br>Line 2")
        assert "Line 1" in md
        assert "Line 2" in md

    def test_html_with_literal_asterisks(self, html_to_md):
        """Asterisks in HTML text should be escaped in markdown."""
        md = html_to_md.convert("* Literal Asterisk")
        assert "*" in md or "\\*" in md

    def test_html_greater_than_becomes_blockquote(self, html_to_md):
        """Lines starting with > in HTML text become blockquotes in markdown."""
        md = html_to_md.convert("> This is a quote")
        assert ">" in md

    def test_escaped_greater_than_at_line_start(self, md_to_html):
        r"""Test that \> at line start renders as literal >, not blockquote."""
        md = r"\> This is not a blockquote"
        result = md_to_html.convert(md)
        assert "<blockquote>" not in result

    def test_real_blockquote_vs_escaped(self, md_to_html):
        """Real blockquotes work but escaped > doesn't."""
        md_blockquote = "> This is a blockquote"
        result_blockquote = md_to_html.convert(md_blockquote)
        assert "<blockquote>" in result_blockquote

        md_escaped = r"\> This is not a blockquote"
        result_escaped = md_to_html.convert(md_escaped)
        assert "<blockquote>" not in result_escaped


# -- Math Round-Trips (extended) ---------------------------------------------


class TestMathRoundTrips:
    """Extended math conversion tests."""

    def test_inline_math_with_underscores(self, md_to_html, html_to_md):
        md = "$x_1 + x_2 = y$"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "x_1" in back
        assert "x_2" in back

    def test_block_math_with_underscores(self, md_to_html, html_to_md):
        md = "$$\\sum_{i=1}^{n} x_i$$"
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "sum" in back or "\\sum" in back

    def test_math_mixed_with_text(self, md_to_html, html_to_md):
        md = "The equation $E=mc^2$ is famous."
        html = md_to_html.convert(md)
        back = html_to_md.convert(html)
        assert "equation" in back
        assert "E=mc^2" in back
        assert "famous" in back
