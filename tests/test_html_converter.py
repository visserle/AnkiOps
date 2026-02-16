"""Tests for HTMLToMarkdown converter with round-trip validation.

All tests start with markdown, convert to HTML, then back to markdown,
and verify the round-trip produces equivalent markdown.
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




class TestRoundTripHeadings:
    """Test round-trip conversion of headings."""

    def test_h1_roundtrip(self, html_to_md, md_to_html):
        original_md = "# Main Title"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert restored_md.strip() == "# Main Title"

    def test_h2_roundtrip(self, html_to_md, md_to_html):
        original_md = "## Subtitle"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert restored_md.strip() == "## Subtitle"

    def test_h3_roundtrip(self, html_to_md, md_to_html):
        original_md = "### Section"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert restored_md.strip() == "### Section"

    def test_heading_with_formatting_roundtrip(self, html_to_md, md_to_html):
        original_md = "## **Bold** Title"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "##" in restored_md
        assert "**Bold**" in restored_md
        assert "Title" in restored_md


class TestRoundTripLists:
    """Test round-trip conversion of lists."""

    def test_unordered_list_roundtrip(self, html_to_md, md_to_html):
        original_md = "- Item 1\n- Item 2\n- Item 3"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "- Item 1" in restored_md
        assert "- Item 2" in restored_md
        assert "- Item 3" in restored_md

    def test_ordered_list_roundtrip(self, html_to_md, md_to_html):
        original_md = "1. First\n2. Second\n3. Third"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "1. First" in restored_md
        assert "2. Second" in restored_md
        assert "3. Third" in restored_md

    def test_nested_list_roundtrip(self, html_to_md, md_to_html):
        original_md = "- Parent\n  - Child 1\n  - Child 2"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "- Parent" in restored_md
        assert "- Child 1" in restored_md
        assert "- Child 2" in restored_md

    def test_list_with_formatting_roundtrip(self, html_to_md, md_to_html):
        original_md = "- **Bold** item\n- *Italic* item\n- `code` item"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "**Bold**" in restored_md
        assert "*Italic*" in restored_md
        assert "`code`" in restored_md

    def test_ordered_list_with_nested_unordered_roundtrip(self, html_to_md, md_to_html):
        original_md = (
            "1. First item\n   - Nested bullet 1\n   - Nested bullet 2\n2. Second item"
        )
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        # Check that nested items are indented (indicating they're nested under item 1)
        assert "1. First item" in restored_md
        assert "2. Second item" in restored_md
        # Nested bullets should have indentation (spaces or tabs before -)
        lines = restored_md.split("\n")
        # Find lines with nested bullets - they should have leading whitespace
        nested_lines = [line for line in lines if "Nested bullet" in line]
        assert len(nested_lines) == 2
        assert all(line.startswith((" ", "\t")) for line in nested_lines), (
            "Nested items should be indented"
        )


class TestRoundTripLinks:
    """Test round-trip conversion of links."""

    def test_simple_link_roundtrip(self, html_to_md, md_to_html):
        original_md = "[Link text](https://example.com)"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "[Link text](https://example.com)" in restored_md

    def test_link_with_special_chars_roundtrip(self, html_to_md, md_to_html):
        original_md = "[Click](https://example.com/path?q=1&r=2)"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "[Click]" in restored_md
        assert "example.com/path" in restored_md

    def test_link_with_parentheses_roundtrip(self, html_to_md, md_to_html):
        original_md = (
            "[Wiki](<https://en.wikipedia.org/wiki/Python_(programming_language)>)"
        )
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        # Link with parens should use angle brackets
        assert "[Wiki](<" in restored_md
        assert "programming_language" in restored_md


class TestRoundTripImages:
    """Test round-trip conversion of images."""

    def test_image_roundtrip(self, html_to_md, md_to_html):
        original_md = "![Alt text](<media/photo.jpg>)"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "![Alt text](<media/photo.jpg>)" in restored_md

    def test_image_without_alt_roundtrip(self, html_to_md, md_to_html):
        original_md = "![](<media/photo.jpg>)"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "![](<media/photo.jpg>)" in restored_md

    def test_image_with_width_roundtrip(self, html_to_md, md_to_html):
        original_md = "![Alt](<media/photo.jpg>){width=300}"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "![Alt](<media/photo.jpg>){width=300}" in restored_md


class TestRoundTripCodeBlocks:
    """Test round-trip conversion of code blocks."""

    def test_fenced_code_block_roundtrip(self, html_to_md, md_to_html):
        original_md = "```\ndef foo():\n    pass\n```"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "```" in restored_md
        assert "def foo():" in restored_md
        assert "pass" in restored_md

    def test_fenced_code_block_with_language_roundtrip(self, html_to_md, md_to_html):
        original_md = "```python\ndef foo():\n    pass\n```"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "```python" in restored_md
        assert "def foo():" in restored_md
        assert "pass" in restored_md

    def test_code_block_with_special_chars_roundtrip(self, html_to_md, md_to_html):
        original_md = "```\n<div>test</div>\n**bold**\n```"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "```" in restored_md
        # Inside code block, special chars should be preserved literally
        assert "<div>test</div>" in restored_md or "&lt;div&gt;" in restored_md


class TestRoundTripTables:
    """Test round-trip conversion of tables."""

    def test_simple_table_roundtrip(self, html_to_md, md_to_html):
        original_md = "| Name | Age |\n| --- | --- |\n| Alice | 30 |"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "| Name | Age |" in restored_md
        assert "| --- | --- |" in restored_md
        assert "| Alice | 30 |" in restored_md

    def test_table_with_formatting_roundtrip(self, html_to_md, md_to_html):
        original_md = "| Header |\n| --- |\n| **bold** |\n| *italic* |"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "**bold**" in restored_md
        assert "*italic*" in restored_md


class TestRoundTripBlockquotes:
    """Test round-trip conversion of blockquotes."""

    def test_simple_blockquote_roundtrip(self, html_to_md, md_to_html):
        original_md = "> Important quote"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "> Important quote" in restored_md

    def test_multiline_blockquote_roundtrip(self, html_to_md, md_to_html):
        original_md = "> Line 1\n> Line 2\n> Line 3"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "> Line 1" in restored_md or ">Line 1" in restored_md

    def test_blockquote_with_formatting_roundtrip(self, html_to_md, md_to_html):
        original_md = "> **Bold** and *italic* quote"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert ">" in restored_md
        assert "**Bold**" in restored_md
        assert "*italic*" in restored_md

    def test_separate_blockquotes_roundtrip(self, html_to_md, md_to_html):
        """Test that two separate blockquotes separated by a blank line round-trip correctly."""
        original_md = "> xxx\n\n> xxx"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert restored_md.strip() == original_md

        # Verify stability across multiple round-trips
        html2 = md_to_html.convert(restored_md)
        restored_md2 = html_to_md.convert(html2)
        html3 = md_to_html.convert(restored_md2)
        restored_md3 = html_to_md.convert(html3)
        assert restored_md == restored_md2 == restored_md3, (
            "Separate blockquotes should be stable across round-trips"
        )

    def test_blockquote_spacing_preserved_in_markdown(self, html_to_md, md_to_html):
        """Test that markdown properly formats blockquotes with spacing.

        This documents expected behavior: when HTML has adjacent blockquotes
        and text, the markdown will add blank lines for proper separation.
        This is a markdown requirement and improves readability.
        """
        original_md = "Text before\n\n> Quote\n\nText after"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)

        # Blockquote marker is preserved
        assert ">" in restored_md
        assert "Quote" in restored_md
        assert "Text before" in restored_md
        assert "Text after" in restored_md

        # Blank lines around blockquote are preserved (this is expected)
        assert "\n\n>" in restored_md or "\n>" in restored_md


class TestRoundTripEscapedCharacters:
    r"""Test round-trip conversion of escaped characters.

    Note on > and # at line start:
    These characters are intentionally NOT preserved as escaped across
    round-trips. This is a deliberate trade-off to enable better Anki editing:
    - Users can type "> quote" or "# heading" in Anki to create formatting
    - Anki stores typed > as &gt; (HTML entity), making it impossible to
      distinguish from escaped \> in markdown
    - We prioritize the common use case (creating blockquotes/headings) over
      the rare case (preserving literal > or # at line start across round-trips)
    """

    def test_escaped_asterisk_roundtrip(self, html_to_md, md_to_html):
        original_md = r"Therapeut\*in und Patient\*in"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        # Should preserve the escaped asterisks
        assert r"\*" in restored_md

    def test_escaped_underscore_roundtrip(self, html_to_md, md_to_html):
        original_md = r"snake\_case\_variable"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert r"\_" in restored_md or "_" in restored_md

    def test_escaped_brackets_roundtrip(self, html_to_md, md_to_html):
        original_md = r"\[not a link\]"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert r"\[" in restored_md or "[not a link]" in restored_md

    def test_escaped_backtick_roundtrip(self, html_to_md, md_to_html):
        original_md = r"Use \`backticks\` for code"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "backticks" in restored_md

    def test_escaped_hash_roundtrip(self, html_to_md, md_to_html):
        original_md = r"\# Not a heading"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "# Not a heading" in restored_md

    def test_escaped_greater_than_becomes_blockquote_after_roundtrip(
        self, html_to_md, md_to_html
    ):
        r"""Escaped > renders as literal initially, then becomes blockquote.

        Flow: \> in markdown -> literal > in HTML -> > in markdown
              -> <blockquote> in HTML

        IMPORTANT: This "data loss" is intentional!
        - Anki stores user-typed ">" as "&gt;" (HTML entity)
        - We can't distinguish between escaped \> and user-typed >
        - We choose to enable the common case: typing "> quote" in Anki to
          create blockquotes, at the cost of not preserving literal >
        """
        original_md = r"\> This text"
        html = md_to_html.convert(original_md)
        # First conversion: escape works, HTML has literal >
        assert "> This text" in html or "&gt; This text" in html
        assert "<blockquote>" not in html

        # Convert HTML back to markdown
        restored_md = html_to_md.convert(html)
        assert "> This text" in restored_md

        # Second conversion: now > is treated as blockquote syntax
        html2 = md_to_html.convert(restored_md)
        assert "<blockquote>" in html2

    def test_escaped_pipe_roundtrip(self, html_to_md, md_to_html):
        original_md = r"a \| b"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "a | b" in restored_md or r"a \| b" in restored_md

    def test_literal_backslash_roundtrip(self, html_to_md, md_to_html):
        original_md = r"Path: C:\\Users\\Name"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "C:" in restored_md
        assert "Users" in restored_md

    def test_mixed_escaped_and_formatting_roundtrip(self, html_to_md, md_to_html):
        original_md = r"Therapeut\*in is **important**"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert r"\*" in restored_md  # Literal asterisk
        assert "**important**" in restored_md  # Formatting asterisks


class TestRoundTripSpecialCharacters:
    """Test round-trip conversion of special characters."""

    def test_ampersand_roundtrip(self, html_to_md, md_to_html):
        original_md = "Tom & Jerry"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "Tom & Jerry" in restored_md or "Tom &amp; Jerry" in restored_md

    def test_less_than_greater_than_roundtrip(self, html_to_md, md_to_html):
        original_md = "1 < 2 and 3 > 2"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "<" in restored_md or "&lt;" in restored_md
        assert ">" in restored_md or "&gt;" in restored_md

    def test_quotes_roundtrip(self, html_to_md, md_to_html):
        original_md = "He said \"Hello\" and she replied 'Hi'"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "Hello" in restored_md
        assert "Hi" in restored_md

    def test_arrow_roundtrip(self, html_to_md, md_to_html):
        original_md = "A --> B"
        html = md_to_html.convert(original_md)
        # HTML converter changes → back to -->
        restored_md = html_to_md.convert(html)
        assert "A --> B" in restored_md

    def test_unicode_characters_roundtrip(self, html_to_md, md_to_html):
        original_md = "Café résumé naïve"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "Café" in restored_md
        assert "résumé" in restored_md
        assert "naïve" in restored_md

    def test_emoji_roundtrip(self, html_to_md, md_to_html):
        original_md = "Hello 👋 World 🌍"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "Hello" in restored_md
        assert "World" in restored_md
        # Emojis should be preserved
        assert "👋" in restored_md or "wave" in restored_md.lower()

    def test_german_umlauts_roundtrip(self, html_to_md, md_to_html):
        original_md = "Über Äpfel und Öl"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "Über" in restored_md
        assert "Äpfel" in restored_md
        assert "Öl" in restored_md

    def test_parentheses_in_text_roundtrip(self, html_to_md, md_to_html):
        original_md = "This (with parentheses) text"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "(with parentheses)" in restored_md

    def test_multiple_spaces_roundtrip(self, html_to_md, md_to_html):
        original_md = "Word1    Word2"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "Word1" in restored_md
        assert "Word2" in restored_md


class TestRoundTripComplexScenarios:
    """Test round-trip conversion of complex, real-world scenarios."""

    def test_medical_terminology_roundtrip(self, html_to_md, md_to_html):
        original_md = "==Derealisation== (S)\nUmwelt wird als unwirklich wahrgenommen"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "==Derealisation==" in restored_md
        assert "Umwelt" in restored_md

    def test_mixed_formatting_and_lists_roundtrip(self, html_to_md, md_to_html):
        original_md = """**Symptome:**
- Symptom 1
- Symptom 2
- ==Wichtig=="""
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "**Symptome:**" in restored_md
        assert "- Symptom 1" in restored_md
        assert "==Wichtig==" in restored_md

    def test_code_with_markdown_syntax_roundtrip(self, html_to_md, md_to_html):
        original_md = '```python\n# This is a comment\ntext = "**not bold**"\n```'
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "```python" in restored_md
        assert "comment" in restored_md
        # Content inside code block should be preserved
        assert "not bold" in restored_md

    def test_nested_formatting_roundtrip(self, html_to_md, md_to_html):
        original_md = "**Bold with *italic* inside**"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "**" in restored_md
        assert "*italic*" in restored_md or "italic" in restored_md

    def test_link_with_formatting_roundtrip(self, html_to_md, md_to_html):
        original_md = "[**Bold link**](https://example.com)"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "[" in restored_md
        assert "https://example.com" in restored_md
        assert "Bold link" in restored_md

    def test_multiple_line_breaks_roundtrip(self, html_to_md, md_to_html):
        original_md = "Line 1\n\nLine 2\n\nLine 3"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "Line 1" in restored_md
        assert "Line 2" in restored_md
        assert "Line 3" in restored_md


class TestRoundTripEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_character_formatting_roundtrip(self, html_to_md, md_to_html):
        original_md = "**A** *B* `C`"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "**A**" in restored_md
        assert "*B*" in restored_md
        assert "`C`" in restored_md

    def test_empty_formatting_roundtrip(self, html_to_md, md_to_html):
        # Empty formatting tags should be handled gracefully
        original_html = "Text <strong></strong> more text"
        restored_md = html_to_md.convert(original_html)
        assert "Text" in restored_md
        assert "more text" in restored_md

    def test_consecutive_formatting_roundtrip(self, html_to_md, md_to_html):
        original_md = "**Bold****More bold**"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "**" in restored_md
        assert "Bold" in restored_md

    def test_whitespace_preservation_roundtrip(self, html_to_md, md_to_html):
        original_md = "Start\n\nEnd"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        assert "Start" in restored_md
        assert "End" in restored_md

    def test_literal_html_entities_roundtrip(self, html_to_md, md_to_html):
        # Test that HTML entities are handled correctly
        original_html = "Tom &amp; Jerry &lt;tag&gt;"
        restored_md = html_to_md.convert(original_html)
        # Should decode entities
        assert "Tom & Jerry" in restored_md or "&amp;" in restored_md
        # Angle brackets may be escaped differently
        assert "tag" in restored_md


class TestRoundTripMathExpressions:
    """Test round-trip conversion of LaTeX math expressions."""

    def test_inline_math_with_underscores_roundtrip(self, html_to_md, md_to_html):
        original_md = r"The formula is \(r_{xy} = \frac{a_1}{b_2}\) here"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        # Underscores inside math should NOT be escaped
        assert r"r_{xy}" in restored_md
        assert r"a_1" in restored_md
        assert r"b_2" in restored_md
        # Should not have escaped underscores
        assert r"r\_{xy}" not in restored_md

    def test_block_math_with_underscores_roundtrip(self, html_to_md, md_to_html):
        original_md = r"\[r_{xy}=\frac{\operatorname{cov}_{(x, y)} }{s_x \cdot s_y}\]"
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        # Math delimiters should be preserved
        assert r"\[" in restored_md and r"\]" in restored_md
        # Underscores inside math should NOT be escaped
        assert "r_{xy}" in restored_md
        assert "{cov}_{(x, y)}" in restored_md  # operatorname puts cov in braces
        assert "s_x" in restored_md
        assert "s_y" in restored_md
        # Should not have escaped underscores
        assert r"\_{" not in restored_md

    def test_math_with_text_roundtrip(self, html_to_md, md_to_html):
        original_md = (
            r"A: \[r_{xy}=\frac{\operatorname{cov}_{(x, y)} }{s_x \cdot s_y}\] "
            r"--> Standardisierung der Kovarianz, nimmt Werte von \(-1\) bis \(1\) an"
        )
        html = md_to_html.convert(original_md)
        restored_md = html_to_md.convert(html)
        # Math delimiters should be preserved
        assert r"\[" in restored_md and r"\]" in restored_md
        assert r"\(" in restored_md and r"\)" in restored_md
        # All math underscores should be preserved
        assert "r_{xy}" in restored_md
        assert "{cov}_{(x, y)}" in restored_md  # operatorname puts cov in braces
        assert "s_x" in restored_md
        assert "s_y" in restored_md
        assert r"\(-1\)" in restored_md
        assert r"\(1\)" in restored_md
        # Should not have escaped underscores in math
        assert r"\_{" not in restored_md

    def test_dollar_delimiters_protected(self, html_to_md):
        """$$ and $ delimiters from HTML should not have content escaped."""
        # This tests the bug fix: if HTML contains $$ delimiters (which Anki might store),
        # the content should be protected from escaping
        html_with_dollars = r"$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$"
        markdown = html_to_md.convert(html_with_dollars)

        # Underscores should NOT be escaped
        assert r"\_{" not in markdown
        assert "_{-" in markdown  # Underscores preserved

        # Carets should NOT be escaped or turned into superscript
        assert "e^{-x^2}" in markdown  # Expression preserved correctly
        assert "e2}" not in markdown  # Bug: expression was corrupted

        # The delimiters themselves should be preserved
        assert "$$" in markdown


class TestSpecialCharacterRoundTrips:
    """Test how special characters behave during markdown->HTML->markdown round-trips.

    These tests verify whether plain text with special characters survives round-trips
    or gets interpreted as markdown syntax.
    """

    def test_asterisk_in_math_expression(self, html_to_md, md_to_html):
        """Single asterisks in expressions like '2 * 3' are preserved as literal."""
        original_md = "Calculate 2 * 3 = 6"
        html = md_to_html.convert(original_md)
        assert "2 <em>3 = 6" not in html, "Should not create emphasis"

        restored_md = html_to_md.convert(html)
        assert "2 * 3" in restored_md or r"2 \* 3" in restored_md

    def test_underscore_in_variable_names(self, html_to_md, md_to_html):
        """Underscores in snake_case variable names are preserved as literal."""
        original_md = "Use snake_case_variable here"
        html = md_to_html.convert(original_md)
        assert "<em>case</em>" not in html, "Should not create emphasis"

        restored_md = html_to_md.convert(html)
        assert (
            "snake_case_variable" in restored_md
            or r"snake\_case\_variable" in restored_md
        )

    def test_backtick_creates_inline_code(self, html_to_md, md_to_html):
        """Backticks create inline code formatting (expected markdown behavior)."""
        original_md = "Press the `backtick` key"
        html = md_to_html.convert(original_md)
        assert "<code>backtick</code>" in html

        restored_md = html_to_md.convert(html)
        assert "`backtick`" in restored_md

    def test_hash_in_middle_of_line_stays_literal(self, html_to_md, md_to_html):
        """Hash in middle of line (like issue #123) stays as literal text."""
        original_md = "Fixed issue #123 today"
        html = md_to_html.convert(original_md)
        assert "<h1>" not in html and "<h2>" not in html, "Should not create heading"

        restored_md = html_to_md.convert(html)
        assert "#123" in restored_md or r"\#123" in restored_md

    def test_hash_at_line_start_creates_heading(self, html_to_md, md_to_html):
        """Hash at line start creates heading (standard markdown behavior)."""
        original_md = "# Heading"
        html = md_to_html.convert(original_md)
        assert "<h1>" in html

        restored_md = html_to_md.convert(html)
        assert "# Heading" in restored_md

    def test_pipe_in_plain_text_stays_literal(self, html_to_md, md_to_html):
        """Pipe in plain text (not table syntax) stays as literal character."""
        original_md = "Choose option A | option B"
        html = md_to_html.convert(original_md)
        assert "<table>" not in html, "Should not create table"

        restored_md = html_to_md.convert(html)
        assert (
            "option A | option B" in restored_md
            or r"option A \| option B" in restored_md
        )

    def test_single_tilde_in_paths_stays_literal(self, html_to_md, md_to_html):
        """Single tilde in file paths stays as literal (not strikethrough)."""
        original_md = "Edit ~/.bashrc file"
        html = md_to_html.convert(original_md)
        assert "<del>" not in html and "<s>" not in html

        restored_md = html_to_md.convert(html)
        assert "~/.bashrc" in restored_md or r"\~/.bashrc" in restored_md

    def test_double_arrow_conversion(self, html_to_md, md_to_html):
        """==> is converted to Unicode ⇒ in HTML, then back to ==> in markdown."""
        original_md = "A ==> B"
        html = md_to_html.convert(original_md)
        assert "⇒" in html or "==>" in html

        restored_md = html_to_md.convert(html)
        assert "A ==> B" in restored_md

    def test_single_arrow_conversion(self, html_to_md, md_to_html):
        """---> is converted to Unicode → in HTML, then back to --> in markdown."""
        original_md = "A --> B"
        html = md_to_html.convert(original_md)
        assert "→" in html or "-->" in html

        restored_md = html_to_md.convert(html)
        assert "A --> B" in restored_md


class TestDirectHTMLConversion:
    """Test direct HTML to markdown conversion."""

    def test_plain_html_paragraph(self, html_to_md):
        html = "<p>Hello world</p>"
        assert html_to_md.convert(html) == "Hello world"

    def test_html_with_literal_asterisks(self, html_to_md):
        html = "Therapeut*in und Patient*in"
        result = html_to_md.convert(html)
        # Literal asterisks should be escaped
        assert r"\*" in result

    def test_html_with_line_starting_greater_than_becomes_blockquote(self, html_to_md):
        """HTML text starting with > is converted to blockquote markdown syntax.

        This means typing "> text" in Anki will become a blockquote after round-trip.
        """
        html = "> This text"
        result = html_to_md.convert(html)
        assert "> This text" in result

    def test_html_bold_tag(self, html_to_md):
        html = "<b>bold</b>"
        assert html_to_md.convert(html) == "**bold**"

    def test_html_line_break(self, html_to_md):
        html = "Line one<br>Line two"
        result = html_to_md.convert(html)
        assert "Line one\nLine two" == result

    def test_html_with_nbsp(self, html_to_md):
        # Non-breaking space: preserved or converted to regular space
        html = "Word1&nbsp;Word2"
        result = html_to_md.convert(html)
        assert "Word1" in result
        assert "Word2" in result
