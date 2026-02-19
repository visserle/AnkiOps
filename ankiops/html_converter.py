"""HTML to Markdown converter."""

import re

from bs4 import BeautifulSoup
from html_to_markdown import ConversionOptions, convert_with_visitor

from ankiops.config import LOCAL_MEDIA_DIR

# Use Unicode placeholders (zero-width joiners + unique pattern)
_MD_SPECIAL_CHARS = {
    "*": "\u200dMDESCASTERISK\u200d",
    "\\": "\u200dMDESCBACKSLASH\u200d",
    "+": "\u200dMDESPLUS\u200d",
    "#": "\u200dMDESCHASH\u200d",
}


# Tags where content is already protected (don't escape inside these)
_PROTECTED_TAGS = {
    "code",
    "pre",
    "em",
    "strong",
    "mark",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}


def _protect_literal_chars(html: str) -> str:
    """Replace special markdown chars in plain text with placeholders."""
    soup = BeautifulSoup(html, "html.parser")

    for text_node in soup.find_all(string=True):
        # Skip if inside a tag that's already handled by markdown syntax
        if text_node.parent.name in _PROTECTED_TAGS:
            continue

        text = str(text_node)

        # Don't escape inside LaTeX math expressions
        # Math delimiters: \(...\) for inline, \[...\] for block, dollar signs not
        # protected
        if r"\(" in text or r"\[" in text:
            # Check if this looks like it contains math
            # Pattern for inline math: \(...\)
            # Pattern for block math: \[...\]
            # Only match explicit math delimiters with backslashes
            math_pattern = r"(\\\(.*?\\\)|\\\[.*?\\\])"

            # Split by math regions to preserve them
            parts = []
            last_end = 0
            for match in re.finditer(math_pattern, text, re.DOTALL):
                # Add text before math (with escaping)
                before = text[last_end : match.start()]
                for char, placeholder in _MD_SPECIAL_CHARS.items():
                    before = before.replace(char, placeholder)
                parts.append(before)
                # Add math part (without escaping)
                parts.append(match.group(0))
                last_end = match.end()

            # Add remaining text after last math (with escaping)
            after = text[last_end:]
            for char, placeholder in _MD_SPECIAL_CHARS.items():
                after = after.replace(char, placeholder)
            parts.append(after)

            text = "".join(parts)
        else:
            # No math, escape normally
            for char, placeholder in _MD_SPECIAL_CHARS.items():
                text = text.replace(char, placeholder)

        text_node.replace_with(text)

    return str(soup)


def _restore_escaped_chars(md: str) -> str:
    """Restore placeholders as escaped markdown characters."""
    for char, placeholder in _MD_SPECIAL_CHARS.items():
        md = md.replace(placeholder, "\\" + char)
    return md


class _AnkiVisitor:
    """Visitor for Anki-specific HTML elements."""

    def visit_image(self, node, src, alt, title):
        style = node["attributes"].get("style", "")
        width_match = re.search(r"width:\s*([\d.]+)px", style)
        width_attr = (
            f"{{width={int(float(width_match.group(1)))}}}" if width_match else ""
        )
        return {
            "type": "custom",
            "output": f"![{alt}](<{LOCAL_MEDIA_DIR}/{src}>){width_attr}",
        }

    def visit_underline(self, node, text):
        content = text.strip()
        if content:
            return {"type": "custom", "output": f"<u>{content}</u>"}
        return {"type": "skip"}

    def visit_link(self, node, href, text, title):
        # Always use angle brackets for robustness
        return {"type": "custom", "output": f"[{text}](<{href}>)"}

    def visit_element_end(self, node, text):
        if node["tag_name"] == "br":
            return {"type": "custom", "output": "\n"}
        return {"type": "continue"}


class HTMLToMarkdown:
    """Convert HTML to clean Markdown."""

    _OPTIONS = ConversionOptions(
        heading_style="atx",
        bullets="-",
        list_indent_width=3,
        highlight_style="double-equal",
        autolinks=False,
        extract_metadata=False,
    )

    def convert(self, html: str) -> str:
        """Convert HTML to Markdown."""
        if not html or not html.strip():
            return ""

        # Normalize multiple <br> before blockquotes to ensure stable round-trips
        # The html-to-markdown library treats <br><br><blockquote> incorrectly,
        # producing only \n> instead of \n\n>. We normalize to single <br>.
        html = re.sub(r"(<br>\s*)+(<blockquote>)", r"<br>\2", html, flags=re.IGNORECASE)

        # Protect literal characters before conversion
        html = _protect_literal_chars(html)

        md = convert_with_visitor(html, self._OPTIONS, visitor=_AnkiVisitor())

        # Restore as escaped characters
        md = _restore_escaped_chars(md)

        # Arrow replacements (convert Unicode arrows back to ASCII)
        md = (
            md.replace("\u2192", "-->")
            .replace("\u21d2", "==>")
            .replace("\u2260", "=/=")
        )

        def _normalize_code_blocks(match):
            content = match.group(1)
            if not content.strip():
                return ""
            return "```" + content.rstrip() + "\n```"

        # Normalize code blocks: remove empty ones, strictly rstrip content of others
        md = re.sub(r"(?s)```(.*?)```", _normalize_code_blocks, md)

        # Collapse excessive newlines
        md = re.sub(r"\n{3,}", "\n\n", md)

        return md.strip()
