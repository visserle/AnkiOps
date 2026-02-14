"""Tests for MarkdownToHTML converter with round-trip validation.

All tests start with HTML, convert to markdown, then back to HTML,
and verify the round-trip produces equivalent HTML.
"""

import pytest
from ankiops.html_converter import HTMLToMarkdown
from ankiops.markdown_converter import MarkdownToHTML


@pytest.fixture
def md_to_html():
    return MarkdownToHTML()


@pytest.fixture
def html_to_md():
    return HTMLToMarkdown()


class TestRoundTripBasicFormatting:
    """Test round-trip conversion of basic formatting from HTML origin."""

    def test_empty_input(self, md_to_html):
        assert md_to_html.convert("") == ""
        assert md_to_html.convert("   ") == ""

    def test_plain_text_roundtrip(self, md_to_html, html_to_md):
        original_html = "Hello world"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Hello world" in restored_html

    def test_bold_roundtrip(self, md_to_html, html_to_md):
        original_html = "This is <strong>bold</strong> text"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<strong>bold</strong>" in restored_html
        assert "This is" in restored_html

    def test_bold_b_tag_roundtrip(self, md_to_html, html_to_md):
        original_html = "This is <b>bold</b> text"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "bold" in restored_html
        # Should convert to strong tag
        assert "<strong>" in restored_html or "<b>" in restored_html

    def test_italic_roundtrip(self, md_to_html, html_to_md):
        original_html = "This is <em>italic</em> text"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<em>italic</em>" in restored_html

    def test_italic_i_tag_roundtrip(self, md_to_html, html_to_md):
        original_html = "This is <i>italic</i> text"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "italic" in restored_html
        assert "<em>" in restored_html or "<i>" in restored_html

    def test_code_roundtrip(self, md_to_html, html_to_md):
        original_html = "Use the <code>print()</code> function"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<code>print()</code>" in restored_html

    def test_mark_roundtrip(self, md_to_html, html_to_md):
        original_html = "This is <mark>important</mark> text"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<mark>important</mark>" in restored_html

    def test_underline_roundtrip(self, md_to_html, html_to_md):
        original_html = "This is <u>underlined</u> text"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<u>underlined</u>" in restored_html

    def test_nested_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = "<strong><em>bold and italic</em></strong>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "bold and italic" in restored_html
        # Should preserve both formatting
        assert "<strong>" in restored_html or "<b>" in restored_html
        assert "<em>" in restored_html or "<i>" in restored_html


class TestRoundTripHeadings:
    """Test round-trip conversion of headings."""

    def test_h1_roundtrip(self, md_to_html, html_to_md):
        original_html = "<h1>Main Title</h1>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<h1>Main Title</h1>" in restored_html

    def test_h2_roundtrip(self, md_to_html, html_to_md):
        original_html = "<h2>Subtitle</h2>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<h2>Subtitle</h2>" in restored_html

    def test_h3_roundtrip(self, md_to_html, html_to_md):
        original_html = "<h3>Section</h3>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<h3>Section</h3>" in restored_html

    def test_heading_with_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = "<h2><strong>Bold</strong> Title</h2>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<h2>" in restored_html
        assert "Bold" in restored_html
        assert "Title" in restored_html
        assert "<strong>" in restored_html or "<b>" in restored_html


class TestRoundTripLists:
    """Test round-trip conversion of lists."""

    def test_unordered_list_roundtrip(self, md_to_html, html_to_md):
        original_html = "<ul><li>Item 1</li><li>Item 2</li><li>Item 3</li></ul>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<ul>" in restored_html
        assert "<li>" in restored_html
        assert "Item 1" in restored_html
        assert "Item 2" in restored_html
        assert "Item 3" in restored_html

    def test_ordered_list_roundtrip(self, md_to_html, html_to_md):
        original_html = "<ol><li>First</li><li>Second</li><li>Third</li></ol>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<ol>" in restored_html
        assert "First" in restored_html
        assert "Second" in restored_html
        assert "Third" in restored_html

    def test_nested_list_roundtrip(self, md_to_html, html_to_md):
        original_html = (
            "<ul><li>Parent<ul><li>Child 1</li><li>Child 2</li></ul></li></ul>"
        )
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<ul>" in restored_html
        assert "Parent" in restored_html
        assert "Child 1" in restored_html
        assert "Child 2" in restored_html

    def test_list_with_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = (
            "<ul><li><strong>Bold</strong></li>"
            "<li><em>Italic</em></li>"
            "<li><code>code</code></li></ul>"
        )
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<strong>" in restored_html or "<b>" in restored_html
        assert "<em>" in restored_html or "<i>" in restored_html
        assert "<code>" in restored_html

    def test_ordered_list_with_nested_unordered_roundtrip(self, md_to_html, html_to_md):
        original_html = (
            "<ol><li>First item<ul><li>Nested bullet 1</li><li>Nested bullet 2</li></ul></li>"
            "<li>Second item</li></ol>"
        )
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        # Verify the nesting structure is preserved: <ul> should be inside <ol><li>
        assert "<ol>" in restored_html
        assert "<ul>" in restored_html
        # Check that ul comes after the first li but before the closing of ol
        # This ensures proper nesting
        ol_start = restored_html.find("<ol>")
        ul_start = restored_html.find("<ul>")
        ol_end = restored_html.find("</ol>")
        assert ol_start < ul_start < ol_end, "UL should be nested inside OL"
        # Verify content is in the right structure
        assert "First item" in restored_html
        assert "Nested bullet 1" in restored_html
        assert "Nested bullet 2" in restored_html
        assert "Second item" in restored_html


class TestRoundTripLinks:
    """Test round-trip conversion of links."""

    def test_simple_link_roundtrip(self, md_to_html, html_to_md):
        original_html = '<a href="https://example.com">Link text</a>'
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "https://example.com" in restored_html
        assert "Link text" in restored_html

    def test_link_with_query_params_roundtrip(self, md_to_html, html_to_md):
        original_html = '<a href="https://example.com/path?q=1&r=2">Click</a>'
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "example.com/path" in restored_html
        assert "Click" in restored_html

    def test_link_with_parentheses_roundtrip(self, md_to_html, html_to_md):
        original_html = (
            '<a href="https://en.wikipedia.org/wiki/Python_'
            '(programming_language)">Wiki</a>'
        )
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "programming_language" in restored_html
        assert "Wiki" in restored_html


class TestRoundTripImages:
    """Test round-trip conversion of images."""

    def test_image_roundtrip(self, md_to_html, html_to_md):
        original_html = '<img src="photo.jpg" alt="Alt text">'
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert 'src="photo.jpg"' in restored_html
        assert 'alt="Alt text"' in restored_html

    def test_image_without_alt_roundtrip(self, md_to_html, html_to_md):
        original_html = '<img src="photo.jpg" alt="">'
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert 'src="photo.jpg"' in restored_html

    def test_image_with_width_roundtrip(self, md_to_html, html_to_md):
        original_html = '<img src="photo.jpg" alt="Alt" style="width: 300px;">'
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert 'src="photo.jpg"' in restored_html
        assert "width: 300px" in restored_html


class TestRoundTripCodeBlocks:
    """Test round-trip conversion of code blocks."""

    def test_pre_code_roundtrip(self, md_to_html, html_to_md):
        original_html = "<pre><code>def foo():\n    pass</code></pre>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<pre><code>" in restored_html
        assert "def foo():" in restored_html
        assert "pass" in restored_html

    def test_pre_code_with_language_roundtrip(self, md_to_html, html_to_md):
        original_html = (
            '<pre><code class="language-python">def foo():\n    pass</code></pre>'
        )
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<pre><code" in restored_html
        assert 'class="language-python"' in restored_html
        # Content is preserved (may have syntax highlighting spans)
        assert "def" in restored_html
        assert "foo" in restored_html
        assert "pass" in restored_html

    def test_code_block_with_special_chars_roundtrip(self, md_to_html, html_to_md):
        original_html = "<pre><code>&lt;div&gt;test&lt;/div&gt;</code></pre>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "&lt;div&gt;" in restored_html or "<div>" in restored_html


class TestRoundTripTables:
    """Test round-trip conversion of tables."""

    def test_simple_table_roundtrip(self, md_to_html, html_to_md):
        original_html = """
        <table>
            <thead><tr><th>Name</th><th>Age</th></tr></thead>
            <tbody><tr><td>Alice</td><td>30</td></tr></tbody>
        </table>
        """
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<table>" in restored_html
        assert "<th>Name</th>" in restored_html
        assert "<th>Age</th>" in restored_html
        assert "<td>Alice</td>" in restored_html
        assert "<td>30</td>" in restored_html

    def test_table_with_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = """
        <table>
            <thead><tr><th>Header</th></tr></thead>
            <tbody>
                <tr><td><strong>bold</strong></td></tr>
                <tr><td><em>italic</em></td></tr>
            </tbody>
        </table>
        """
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<strong>" in restored_html or "<b>" in restored_html
        assert "<em>" in restored_html or "<i>" in restored_html


class TestRoundTripBlockquotes:
    """Test round-trip conversion of blockquotes."""

    def test_simple_blockquote_roundtrip(self, md_to_html, html_to_md):
        original_html = "<blockquote>Important quote</blockquote>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<blockquote>" in restored_html
        assert "Important quote" in restored_html

    def test_blockquote_with_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = (
            "<blockquote><strong>Bold</strong> and <em>italic</em></blockquote>"
        )
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<blockquote>" in restored_html
        assert "Bold" in restored_html
        assert "italic" in restored_html

    def test_multiline_blockquote_roundtrip(self, md_to_html, html_to_md):
        original_html = "<blockquote>Line 1<br>Line 2<br>Line 3</blockquote>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<blockquote>" in restored_html
        assert "Line 1" in restored_html
        assert "Line 2" in restored_html

    def test_blockquote_spacing_added_for_readability(self, md_to_html, html_to_md):
        """Test that spacing is added around blockquotes for better readability.

        This is expected behavior: markdown requires blank lines around blockquotes,
        so round-tripping will add <br> tags for visual separation. This improves
        readability in Anki cards without changing the semantic meaning.
        """
        original_html = "Text before<blockquote>Quote</blockquote>Text after"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)

        # Blockquote content is preserved
        assert "<blockquote>" in restored_html
        assert "Quote" in restored_html
        assert "Text before" in restored_html
        assert "Text after" in restored_html

        # Spacing is added for readability (this is expected)
        assert (
            "<br><br><blockquote>" in restored_html
            or "<br><blockquote>" in restored_html
        )

    def test_separate_blockquotes_roundtrip(self, md_to_html, html_to_md):
        """Test that two separate blockquotes separated by a blank line round-trip correctly."""
        original_html = (
            "<blockquote>xxx</blockquote><br><br><blockquote>xxx</blockquote>"
        )
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)

        # Both blockquotes are preserved as separate elements
        assert restored_html.count("<blockquote>") == 2
        assert restored_html.count("xxx") == 2

        # Verify stability across multiple round-trips
        md2 = html_to_md.convert(restored_html)
        html2 = md_to_html.convert(md2)
        md3 = html_to_md.convert(html2)
        assert md == md2 == md3, (
            "Separate blockquotes should be stable across round-trips"
        )

    def test_blockquote_roundtrip_stability(self, md_to_html, html_to_md):
        """Test that blockquotes don't oscillate between round trips.

        Regression test: previously, blockquotes would alternate between
        having single and double newlines before them on successive round trips.
        This test ensures the conversions are stable.
        """
        original_md = "Q: Question?\n\n> Answer quote"

        # Round trip 1
        html1 = md_to_html.convert(original_md)
        md1 = html_to_md.convert(html1)

        # Round trip 2
        html2 = md_to_html.convert(md1)
        md2 = html_to_md.convert(html2)

        # Round trip 3
        html3 = md_to_html.convert(md2)
        md3 = html_to_md.convert(html3)

        # All should be identical (stable)
        assert md1 == md2 == md3, "Blockquote round trips should be stable"
        assert ">" in md1
        assert "Question?" in md1
        assert "Answer quote" in md1


class TestRoundTripSpecialCharacters:
    """Test round-trip conversion of special characters."""

    def test_html_entities_ampersand_roundtrip(self, md_to_html, html_to_md):
        original_html = "Tom &amp; Jerry"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Tom" in restored_html
        assert "Jerry" in restored_html
        assert "&amp;" in restored_html or "&" in restored_html

    def test_html_entities_lt_gt_roundtrip(self, md_to_html, html_to_md):
        original_html = "1 &lt; 2 and 3 &gt; 2"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        # Should preserve the comparison
        assert "1" in restored_html
        assert "2" in restored_html
        assert "3" in restored_html

    def test_html_entities_quotes_roundtrip(self, md_to_html, html_to_md):
        original_html = "He said &quot;Hello&quot;"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Hello" in restored_html

    def test_unicode_characters_roundtrip(self, md_to_html, html_to_md):
        original_html = "Café résumé naïve"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Café" in restored_html
        assert "résumé" in restored_html
        assert "naïve" in restored_html

    def test_german_umlauts_roundtrip(self, md_to_html, html_to_md):
        original_html = "Über Äpfel und Öl"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Über" in restored_html
        assert "Äpfel" in restored_html
        assert "Öl" in restored_html

    def test_emoji_roundtrip(self, md_to_html, html_to_md):
        original_html = "Hello 👋 World 🌍"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Hello" in restored_html
        assert "World" in restored_html
        # Emojis should be preserved
        assert "👋" in restored_html or "wave" in restored_html.lower()

    def test_non_breaking_space_roundtrip(self, md_to_html, html_to_md):
        original_html = "Word1&nbsp;Word2&nbsp;Word3"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Word1" in restored_html
        assert "Word2" in restored_html
        assert "Word3" in restored_html

    def test_multiple_spaces_roundtrip(self, md_to_html, html_to_md):
        original_html = "Word1    Word2"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Word1" in restored_html
        assert "Word2" in restored_html

    def test_arrow_character_roundtrip(self, md_to_html, html_to_md):
        # The converters transform --> to → and back
        original_html = "A → B"
        md = html_to_md.convert(original_html)
        # Should convert back to -->
        assert "-->" in md
        restored_html = md_to_html.convert(md)
        # Should convert back to →
        assert "→" in restored_html


class TestRoundTripEscapedCharacters:
    """Test round-trip conversion of literal special characters in HTML.

    Note on > and # at line start:
    These characters become markdown syntax after HTML round-trip (not
    escaped). This is intentional - see TestRoundTripEscapedCharacters in
    test_html_converter.py for detailed explanation of this trade-off.
    """

    def test_literal_asterisks_roundtrip(self, md_to_html, html_to_md):
        # HTML with literal asterisks (not emphasis)
        original_html = "Therapeut*in und Patient*in"
        md = html_to_md.convert(original_html)
        # Should escape them in markdown
        assert r"\*" in md
        restored_html = md_to_html.convert(md)
        # Should restore as literal asterisks
        assert "*" in restored_html
        assert "Therapeut" in restored_html

    def test_literal_underscores_roundtrip(self, md_to_html, html_to_md):
        original_html = "snake_case_variable"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "snake_case_variable" in restored_html

    def test_literal_brackets_roundtrip(self, md_to_html, html_to_md):
        original_html = "[not a link]"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "[not a link]" in restored_html

    def test_literal_parentheses_roundtrip(self, md_to_html, html_to_md):
        original_html = "Text (with parentheses) here"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "(with parentheses)" in restored_html

    def test_literal_backticks_roundtrip(self, md_to_html, html_to_md):
        original_html = "Use `backticks` for code"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "backticks" in restored_html

    def test_html_text_with_hash_becomes_heading_after_roundtrip(
        self, md_to_html, html_to_md
    ):
        """HTML text starting with # becomes a heading after round-trip.

        Flow: "# Title" in HTML -> "# Title" in markdown -> <h1> in HTML

        This enables Anki editing: type "# Title" as plain text, and it
        becomes a proper heading after the next sync.
        """
        original_html = "# This text"
        md = html_to_md.convert(original_html)
        assert "# This text" in md

        restored_html = md_to_html.convert(md)
        assert "<h1>This text</h1>" in restored_html

    def test_html_text_with_greater_than_becomes_blockquote_after_roundtrip(
        self, md_to_html, html_to_md
    ):
        """HTML text starting with > becomes a blockquote after round-trip.

        Flow: "> quote" in HTML -> "> quote" in markdown
              -> <blockquote> in HTML

        This enables Anki editing: type "> quote" as plain text, and it
        becomes a proper blockquote after the next sync.
        """
        original_html = "> This text"
        md = html_to_md.convert(original_html)
        assert "> This text" in md

        restored_html = md_to_html.convert(md)
        assert "<blockquote>" in restored_html
        assert "This text" in restored_html

    def test_multiline_greater_than_becomes_blockquote(self, md_to_html, html_to_md):
        """Multiple lines starting with > become a blockquote after round-trip."""
        original_html = "> Line 1<br>> Line 2<br>> Line 3"
        md = html_to_md.convert(original_html)
        assert "> Line 1" in md
        assert "> Line 2" in md
        assert "> Line 3" in md

        restored_html = md_to_html.convert(md)
        assert "<blockquote>" in restored_html

    def test_mixed_literal_and_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = "Therapeut*in is <strong>important</strong>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        # Literal asterisk should be preserved
        assert "*" in restored_html
        # Bold formatting should be preserved
        assert "<strong>" in restored_html or "<b>" in restored_html


class TestRoundTripComplexScenarios:
    """Test round-trip conversion of complex, real-world scenarios."""

    def test_medical_terminology_roundtrip(self, md_to_html, html_to_md):
        original_html = (
            "<mark>Derealisation</mark> (S)<br>Umwelt wird als unwirklich wahrgenommen"
        )
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<mark>Derealisation</mark>" in restored_html
        assert "Umwelt" in restored_html

    def test_mixed_formatting_and_lists_roundtrip(self, md_to_html, html_to_md):
        original_html = """<strong>Symptome:</strong><br>
<ul>
<li>Symptom 1</li>
<li>Symptom 2</li>
<li><mark>Wichtig</mark></li>
</ul>"""
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "<strong>Symptome" in restored_html or "<b>Symptome" in restored_html
        assert "<ul>" in restored_html
        assert "Symptom 1" in restored_html
        assert "<mark>Wichtig</mark>" in restored_html

    def test_nested_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = "<strong>Bold with <em>italic</em> inside</strong>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Bold" in restored_html
        assert "italic" in restored_html

    def test_link_with_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = '<a href="https://example.com"><strong>Bold link</strong></a>'
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "https://example.com" in restored_html
        assert "Bold link" in restored_html

    def test_multiple_breaks_roundtrip(self, md_to_html, html_to_md):
        original_html = "Line 1<br><br>Line 2<br><br>Line 3"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Line 1" in restored_html
        assert "Line 2" in restored_html
        assert "Line 3" in restored_html


class TestRoundTripEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_character_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = "<strong>A</strong> <em>B</em> <code>C</code>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "A" in restored_html
        assert "B" in restored_html
        assert "C" in restored_html

    def test_empty_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = "Text <strong></strong> more text"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Text" in restored_html
        assert "more text" in restored_html

    def test_consecutive_formatting_roundtrip(self, md_to_html, html_to_md):
        original_html = "<strong>Bold</strong><strong>More</strong>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "Bold" in restored_html
        assert "More" in restored_html

    def test_whitespace_in_tags_roundtrip(self, md_to_html, html_to_md):
        original_html = "<strong> spaces </strong>"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        assert "spaces" in restored_html


class TestRoundTripMathExpressions:
    """Test round-trip conversion of LaTeX math expressions."""

    def test_inline_math_with_underscores_roundtrip(self, md_to_html, html_to_md):
        original_html = r"The formula is \(r_{xy} = \frac{a_1}{b_2}\) here"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        # Underscores inside math should be preserved
        assert "r_{xy}" in restored_html
        assert "a_1" in restored_html
        assert "b_2" in restored_html

    def test_block_math_with_underscores_roundtrip(self, md_to_html, html_to_md):
        # Block math with backslash delimiters
        original_html = r"\[r_{xy}=\frac{\operatorname{cov}_{(x, y)} }{s_x \cdot s_y}\]"
        md = html_to_md.convert(original_html)
        restored_html = md_to_html.convert(md)
        # Math delimiters should be preserved
        assert r"\[" in restored_html and r"\]" in restored_html
        # Underscores inside math should be preserved
        assert "r_{xy}" in restored_html
        assert "s_x" in restored_html
        assert "s_y" in restored_html

    def test_dollar_delimiters_converted_to_backslash(self, md_to_html):
        """$$ and $ delimiters should be converted to \\[...\\] and \\(...\\)."""
        # Block math with $$ should be converted to \[...\]
        markdown_with_dollars = r"$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$"
        html = md_to_html.convert(markdown_with_dollars)
        # Should use backslash delimiters, not dollar signs
        assert r"\[" in html and r"\]" in html
        assert "$$" not in html
        # Content should be preserved correctly
        assert r"\int_{-\infty}^{\infty}" in html
        assert "e^{-x^2}" in html
        assert r"\sqrt{\pi}" in html

        # Inline math with $ should be converted to \(...\)
        markdown_inline = r"The value is $x_i^2$ here"
        html = md_to_html.convert(markdown_inline)
        assert r"\(" in html and r"\)" in html
        assert "x_i^2" in html


class TestSpecialCharacterConversion:
    """Test how special characters in markdown are converted to HTML."""

    def test_asterisk_in_math_expression(self, md_to_html):
        """Single asterisk in expressions like '2 * 3' stays as literal."""
        md = "Calculate 2 * 3 = 6"
        result = md_to_html.convert(md)
        assert "2 * 3" in result or "2 \\* 3" in result
        assert "<em>3" not in result, "Should not create emphasis"

    def test_underscore_in_snake_case(self, md_to_html):
        """Underscores in snake_case variable names stay as literal."""
        md = "Use snake_case_variable here"
        result = md_to_html.convert(md)
        assert "snake_case_variable" in result
        assert "<em>case</em>" not in result, "Should not create emphasis"

    def test_hash_in_middle_of_line(self, md_to_html):
        """Hash in middle of line (like #123) stays as literal, not heading."""
        md = "Fixed issue #123 today"
        result = md_to_html.convert(md)
        assert "#123" in result
        assert "<h1>" not in result and "<h2>" not in result

    def test_pipe_in_plain_text(self, md_to_html):
        """Pipe in plain text doesn't create table."""
        md = "Choose option A | option B"
        result = md_to_html.convert(md)
        assert "option A" in result and "option B" in result
        assert "<table>" not in result, "Should not create table"

    def test_single_tilde_in_paths(self, md_to_html):
        """Single tilde in file paths stays as literal, not strikethrough."""
        md = "Edit ~/.bashrc file"
        result = md_to_html.convert(md)
        assert "~/.bashrc" in result
        assert "<del>" not in result and "<s>" not in result

    def test_arrow_syntax_converted_to_unicode(self, md_to_html):
        """Arrow syntax (-->, ==>) is converted to Unicode arrow symbols."""
        md_single = "A --> B"
        result_single = md_to_html.convert(md_single)
        assert "→" in result_single  # U+2192

        md_double = "A ==> B"
        result_double = md_to_html.convert(md_double)
        assert "⇒" in result_double  # U+21D2


class TestDirectMarkdownConversion:
    """Test direct markdown to HTML conversion."""

    def test_plain_markdown(self, md_to_html):
        md = "Hello world"
        result = md_to_html.convert(md)
        assert "Hello world" in result

    def test_bold_markdown(self, md_to_html):
        md = "**bold**"
        result = md_to_html.convert(md)
        assert "<strong>bold</strong>" in result

    def test_italic_markdown(self, md_to_html):
        md = "*italic*"
        result = md_to_html.convert(md)
        assert "<em>italic</em>" in result

    def test_inline_code_markdown(self, md_to_html):
        md = "`code`"
        result = md_to_html.convert(md)
        assert "<code>code</code>" in result

    def test_highlight_markdown(self, md_to_html):
        md = "==important=="
        result = md_to_html.convert(md)
        assert "<mark>important</mark>" in result

    def test_link_markdown(self, md_to_html):
        md = "[Link](https://example.com)"
        result = md_to_html.convert(md)
        assert "https://example.com" in result
        assert "Link" in result

    def test_escaped_markdown(self, md_to_html):
        md = r"\*not bold\*"
        result = md_to_html.convert(md)
        # Should render literal asterisks
        assert "*not bold*" in result

    def test_arrow_conversion(self, md_to_html):
        md = "A --> B"
        result = md_to_html.convert(md)
        # Should convert --> to →
        assert "→" in result

    def test_line_breaks(self, md_to_html):
        md = "Line 1\n\nLine 2"
        result = md_to_html.convert(md)
        assert "Line 1" in result
        assert "Line 2" in result
        # Should have break tags
        assert "<br>" in result

    def test_escaped_greater_than_at_line_start(self, md_to_html):
        r"""Test that \> at line start renders as literal >, not blockquote."""
        md = r"\> This is not a blockquote"
        result = md_to_html.convert(md)
        # Should render as literal >
        assert (
            "> This is not a blockquote" in result
            or "&gt; This is not a blockquote" in result
        )
        # Should NOT create a blockquote
        assert "<blockquote>" not in result

    def test_escaped_greater_than_multiline(self, md_to_html):
        r"""Test that multiple \> at line starts render as literals."""
        md = r"\> Line 1" + "\n\n" + r"\> Line 2"
        result = md_to_html.convert(md)
        # Should render as literal >
        assert "> Line 1" in result or "&gt; Line 1" in result
        assert "> Line 2" in result or "&gt; Line 2" in result
        # Should NOT create blockquotes
        assert "<blockquote>" not in result

    def test_real_blockquote_vs_escaped(self, md_to_html):
        """Test that real blockquotes work but escaped > doesn't."""
        # Real blockquote (with space after >)
        md_blockquote = "> This is a blockquote"
        result_blockquote = md_to_html.convert(md_blockquote)
        assert "<blockquote>" in result_blockquote

        # Escaped > (should be literal)
        md_escaped = r"\> This is not a blockquote"
        result_escaped = md_to_html.convert(md_escaped)
        assert "<blockquote>" not in result_escaped
        assert ">" in result_escaped or "&gt;" in result_escaped
