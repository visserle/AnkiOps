"""Behavior tests for the AnkiOps Markdown/HTML dialect."""

from __future__ import annotations

import pytest

from ankiops.html_to_markdown import HTMLToMarkdown
from ankiops.markdown_to_html import MarkdownToHTML


@pytest.fixture
def html_to_md():
    return HTMLToMarkdown()


@pytest.fixture
def md_to_html():
    return MarkdownToHTML()


@pytest.mark.parametrize(
    ("markdown", "expected_html"),
    [
        ("# Title\n\nText", "<h1>Title</h1>Text"),
        ("Line 1\n\nLine 2", "Line 1<br><br>Line 2"),
        (
            "- Item 1\n- Item 2\n\nAfter",
            "<ul>\n<li>Item 1</li>\n<li>Item 2</li>\n</ul>\n<br><br>After",
        ),
        ("A --> B", "A \u2192 B"),
        ("A ==> B", "A \u21d2 B"),
        ("A =/= B", "A \u2260 B"),
    ],
)
def test_markdown_to_html_uses_anki_block_model(md_to_html, markdown, expected_html):
    assert md_to_html.convert(markdown) == expected_html


def test_media_references_use_local_anki_filenames_and_roundtrip_widths(
    md_to_html, html_to_md
):
    markdown = "![Alt](<media/photo.jpg>){width=300}"

    html = md_to_html.convert(markdown)
    assert html == '<img src="photo.jpg" alt="Alt" style="width: 300px;">'
    assert html_to_md.convert(html) == markdown


def test_external_image_urls_roundtrip_without_a_local_media_prefix(
    md_to_html, html_to_md
):
    html = (
        '<img src="http://ww3.haverford.edu/psychology/ble/continuous_ios/'
        'gfx/IOS-original.gif" alt="">'
    )

    markdown = html_to_md.convert(html)

    assert markdown == (
        "![](<http://ww3.haverford.edu/psychology/ble/continuous_ios/"
        "gfx/IOS-original.gif>)"
    )
    assert md_to_html.convert(markdown) == html


def test_empty_image_placeholder_roundtrips_without_becoming_local_media(
    md_to_html, html_to_md
):
    markdown = "![](<>){width=799}"

    html = md_to_html.convert(markdown)

    assert html == '<img src="" alt="" style="width: 799px;">'
    assert html_to_md.convert(html) == markdown


def test_links_with_parentheses_are_wrapped_for_stable_roundtrips(
    md_to_html, html_to_md
):
    markdown = "[Wiki](https://example.com/Python_(programming_language))"

    html = md_to_html.convert(markdown)
    assert (
        html == '<a href="https://example.com/Python_(programming_language)">Wiki</a>'
    )
    assert (
        html_to_md.convert(html)
        == "[Wiki](<https://example.com/Python_(programming_language)>)"
    )


@pytest.mark.parametrize(
    ("html_text", "expected_markdown"),
    [
        ("+ Literal Plus", r"\+ Literal Plus"),
        ("# Literal Hash", r"\# Literal Hash"),
        ("* Literal Asterisk", r"\* Literal Asterisk"),
    ],
)
def test_html_text_that_looks_like_markdown_stays_literal(
    md_to_html, html_to_md, html_text, expected_markdown
):
    markdown = html_to_md.convert(html_text)

    assert markdown == expected_markdown
    assert md_to_html.convert(markdown) == html_text


def test_blockquote_citation_links_keep_quote_continuation(html_to_md):
    html = (
        '<blockquote><p>called. <a href="https://en.wikipedia.org/wiki/Pygmalion'
        '#cite_note-13">[13]</a><br>aus Pygmalion von George Bernard Shaw'
        "</p></blockquote>"
    )

    assert html_to_md.convert(html) == (
        "> called. [[13]](<https://en.wikipedia.org/wiki/Pygmalion#cite_note-13>)\n"
        "> aus Pygmalion von George Bernard Shaw"
    )


def test_math_block_end_is_not_rewritten_as_link(html_to_md):
    html = (
        r"<div>\["
        "\n"
        r"\beta=b \frac{s_x}{s_y}"
        "\n"
        r"\](bei mehreren Pradiktoren gilt dies nur, wenn sie unkorreliert sind)"
        "</div>"
    )

    markdown = html_to_md.convert(html)

    assert (
        r"\](bei mehreren Pradiktoren gilt dies nur, wenn sie unkorreliert sind)"
        in markdown
    )
    assert (
        r"\](<bei mehreren Pradiktoren gilt dies nur, wenn sie unkorreliert sind>)"
        not in markdown
    )


@pytest.mark.parametrize(
    "markdown",
    [
        (
            r"\[\sigma^2=\frac{1}{N} "
            r"\sum_{i=1}^N\left(x_i-\mu\right)^2\]"
        ),
        (
            r"\\[\sigma^2=\frac{1}{N} "
            r"\sum_{i=1}^N\left(x_i-\mu\right)^2\]"
        ),
        (
            r"\\[\sigma^2=\frac{1}{N} "
            r"\sum_{i=1}^N\left(x_i-\mu\right)^2\\]"
        ),
    ],
)
def test_display_math_delimiters_are_canonicalized_without_shortening(
    md_to_html, html_to_md, markdown
):
    expected_math = (
        r"\[\sigma^2=\frac{1}{N} "
        r"\sum_{i=1}^N\left(x_i-\mu\right)^2\]"
    )

    html = md_to_html.convert(markdown)
    assert "<sup>" not in html
    assert expected_math in html
    assert html_to_md.convert(f"<div>{markdown}</div>") == expected_math


@pytest.mark.parametrize(
    "markdown",
    [
        r"\[\]\]",
        r"\[0\]\]",
        r"\[0\]\\]",
    ],
)
def test_escaped_bracket_suffix_roundtrip_stays_stable(
    md_to_html, html_to_md, markdown
):

    first = html_to_md.convert(md_to_html.convert(markdown))
    second = html_to_md.convert(md_to_html.convert(first))

    assert first == second


def test_cloze_syntax_is_left_for_anki(md_to_html, html_to_md):
    markdown = "{{c1::hidden}}"

    assert md_to_html.convert(markdown) == markdown
    assert html_to_md.convert(markdown) == markdown


def test_underlines_remain_explicit_html(md_to_html, html_to_md):
    markdown = html_to_md.convert("<u>underlined</u>")

    assert markdown == "<u>underlined</u>"
    assert md_to_html.convert(markdown) == "<u>underlined</u>"


def test_html_tables_export_in_compact_markdown(html_to_md):
    html = (
        "<table><thead><tr><th>Short</th><th>Long heading</th></tr></thead>"
        "<tbody><tr><td>a</td><td>bbb</td></tr></tbody></table>"
    )

    assert html_to_md.convert(html) == (
        "| Short | Long heading |\n| --- | --- |\n| a | bbb |"
    )


def test_html_table_escaped_pipes_stay_in_their_cells(html_to_md):
    html = (
        "<table><tr><th>A</th><th>B</th></tr>"
        "<tr><td>a | b</td><td><code>c | d</code></td></tr></table>"
    )

    assert html_to_md.convert(html) == "| A | B |\n| --- | --- |\n| a \\| b | `c | d` |"


def test_python_code_blocks_keep_language_and_text(md_to_html, html_to_md):
    markdown = "```python\nprint('hello')\n```"

    html = md_to_html.convert(markdown)
    assert '<pre><code class="language-python">' in html
    assert "print" in html

    back_to_markdown = html_to_md.convert(html)
    assert "print" in back_to_markdown
    assert "'hello'" in back_to_markdown


def test_bullet_html_with_line_breaks_and_images_is_exported_readably(html_to_md):
    html = (
        "<div>Gipfel:</div><div><div class='centeredbox'><div class='leftalign'>"
        "<span class='font_size_80'>\u2022 Anorexie 16-17 Jahre</span><br>"
        "<span class='font_size_80'>\u2022 Bulimie 18-19 Jahre</span><br>"
        "</div><div class='leftalign'>"
        "<img src='paste-a.png' style='width: 971.502px;'></div>"
        "</div></div>"
    )

    markdown = html_to_md.convert(html)

    assert "\u2022 Anorexie 16-17 Jahre" in markdown
    assert "\n\u2022 Bulimie 18-19 Jahre" in markdown
    assert "![](<media/paste-a.png>){width=971}" in markdown
