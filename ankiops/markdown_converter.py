"""Markdown to HTML converter for Anki import."""

import re

import mistune
from mistune.plugins.formatting import mark, strikethrough, subscript, superscript
from mistune.plugins.table import table
from mistune.util import escape as escape_text
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

from ankiops.config import LOCAL_MEDIA_DIR

_IMG_WIDTH_RE = re.compile(r'(<img src="[^"]*" alt="[^"]*")>\{width=(\d+)\}')
# Rewrite [text](url_with_(parens)) to [text](<url>) so mistune doesn't
# misparse balanced parentheses in link destinations
_LINK_WITH_PARENS_RE = re.compile(
    r"\[([^\]]*)\]\("
    r"([^)<>]*\([^)]*\)[^)<>]*)"
    r"\)"
)
_PYGMENTS_FORMATTER = HtmlFormatter(nowrap=True)
_INLINE_MATH_PAREN_PATTERN = r"\\\((?P<ipm_text>[\s\S]+?)\\\)"
_INLINE_MATH_BRACKET_PATTERN = r"\\\[(?P<ibm_text>[\s\S]+?)\\\]"
_BLOCK_MATH_PATTERN = r"^\\\[(?P<bm_text>[\s\S]+?)\\\][ \t]*$"


def _math_plugin(md):
    """Preserve LaTeX math delimiters \\(...\\) and \\[...\\] through parsing."""

    def _parse_token(token_type, group_name):
        def parser(_, m, state):
            state.append_token({"type": token_type, "raw": m.group(group_name)})
            return m.end()

        return parser

    def _parse_block(_, m, state):
        state.append_token({"type": "block_math", "raw": m.group("bm_text")})
        return m.end() + 1

    md.inline.register(
        "inline_math_paren",
        _INLINE_MATH_PAREN_PATTERN,
        _parse_token("inline_math_paren", "ipm_text"),
        before="escape",
    )
    md.inline.register(
        "inline_math_bracket",
        _INLINE_MATH_BRACKET_PATTERN,
        _parse_token("inline_math_bracket", "ibm_text"),
        before="escape",
    )
    md.block.register("block_math", _BLOCK_MATH_PATTERN, _parse_block, before="list")
    if md.renderer and md.renderer.NAME == "html":
        md.renderer.register("inline_math_paren", lambda _, t: "\\(" + t + "\\)")
        md.renderer.register("inline_math_bracket", lambda _, t: "\\[" + t + "\\]")
        md.renderer.register("block_math", lambda _, t: "\\[" + t + "\\]")


class AnkiRenderer(mistune.HTMLRenderer):
    """Custom mistune renderer producing Anki-compatible HTML.

    Only overrides methods where Anki's HTML model differs from standard:
    - No <p> wrapping (Anki uses <br> between blocks).
    - media/ prefix stripping on images
    - --> and ==> arrow replacements
    - Syntax highlighting via Pygments
    """

    def __init__(self):
        super().__init__(escape=False, allow_harmful_protocols=True)

    def __call__(self, tokens, state):
        parts = list(self.iter_tokens(tokens, state))
        return self._join_blocks(parts)

    def _join_blocks(self, parts):
        """Join block-level HTML parts with <br> separators (Anki style)."""
        output = []
        for part in parts:
            if part == "":
                output.append("<br>")
            elif part:
                if output:
                    output.append("<br>")
                output.append(part)
        html = "".join(output)
        html = re.sub(r"(<br>){3,}", "<br><br>", html)
        return html

    def text(self, text):
        return text

    def softbreak(self):
        return "<br>"

    def paragraph(self, text):
        return text

    def image(self, text, url, title=None):
        if url.startswith(f"{LOCAL_MEDIA_DIR}/"):
            url = url[len(LOCAL_MEDIA_DIR) + 1 :]
        return '<img src="' + url + '" alt="' + text + '">'

    def block_code(self, code, info=None):
        code = code.rstrip("\n")
        if not code.strip():
            return ""

        if info:
            lang = info.strip().split(None, 1)[0]
            try:
                lexer = get_lexer_by_name(lang)
                highlighted = highlight(code, lexer, _PYGMENTS_FORMATTER)
                return (
                    f'<pre><code class="language-{lang}">'
                    + highlighted
                    + "</code></pre>"
                )
            except ClassNotFound:
                pass
        return "<pre><code>" + escape_text(code) + "</code></pre>"


class MarkdownToHTML:
    """Convert Markdown back to HTML for Anki."""

    def __init__(self):
        self._md = mistune.create_markdown(
            renderer=AnkiRenderer(),
            plugins=[
                mark,
                table,
                strikethrough,
                superscript,
                subscript,
                _math_plugin,
            ],
        )

    def convert(self, markdown: str) -> str:
        """Convert markdown string to HTML."""
        if not markdown or not markdown.strip():
            return ""

        # Arrow replacements: Replace before parsing to avoid conflicts
        # with markdown syntax (== is used for highlighting)
        markdown = (
            markdown.replace("==>", "\u21d2")
            .replace("-->", "\u2192")
            .replace("=/=", "\u2260")
        )

        markdown = _LINK_WITH_PARENS_RE.sub(r"[\1](<\2>)", markdown)
        html: str = self._md(markdown)  # type: ignore[assignment]
        html = _IMG_WIDTH_RE.sub(r'\1 style="width: \2px;">', html)

        # Fix blank line preservation after lists
        # Mistune loses blank lines after lists - add extra <br> to compensate
        # Pattern: </ol> or </ul> followed by newline then <br>, make it <br><br>
        html = re.sub(r"(</(?:ol|ul)>)\n(<br>)", r"\1\n<br>\2", html)

        # Unescape characters that were explicitly escaped in HTMLToMarkdown
        # to ensure roundtrip stability for our dialect-specific markers.
        # Mistune unescapes most things but can be conservative.
        html = html.replace("\\+", "+").replace("\\#", "#").replace("\\*", "*")

        return html
