"""HTML to Markdown converter."""

import re
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from html_to_markdown import ConversionOptions, convert

from ankiops.config import LOCAL_MEDIA_DIR

# Use Unicode placeholders (zero-width joiners + unique pattern)
_MD_SPECIAL_CHARS = {
    "*": "\u200dMDESCASTERISK\u200d",
    "+": "\u200dMDESPLUS\u200d",
    "#": "\u200dMDESCHASH\u200d",
    "\\": "\u200dMDESCBACKSLASH\u200d",
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

_HTML_TAG_RE = re.compile(
    r"(?is)(<!--.*?-->|</?[a-z][a-z0-9-]*(?:\s[^<>]*?)?>|<!doctype\s+html[^>]*>)"
)
_HTML_ENTITY_RE = re.compile(
    r"&(?:[a-zA-Z][a-zA-Z0-9]+|#\d+|#x[0-9A-Fa-f]+);"
)
# Preserve only well-formed LaTeX delimiters. Reject sequences that contain an
# unescaped closing delimiter inside, otherwise literals like \[\]\] are
# misclassified as math and accumulate backslashes across roundtrips.
_MATH_PATTERN = re.compile(
    r"(\\\((?:(?!\\\))[\s\S])+?\\\)|\\\[(?:(?!\\\])[\s\S])+?\\\])",
    re.DOTALL,
)
_ESCAPED_RIGHT_BRACKET_RUN_RE = re.compile(r"(\\{2,}\])")


def _looks_like_html(text: str) -> bool:
    """Heuristic check for HTML fragments/entities."""
    return bool(_HTML_TAG_RE.search(text) or _HTML_ENTITY_RE.search(text))


def _escape_special_chars(text: str) -> str:
    """Escape markdown special chars while preserving explicit LaTeX math blocks."""

    def _escape_segment(segment: str) -> str:
        preserved_runs: dict[str, str] = {}

        def _preserve(match: re.Match[str]) -> str:
            token = f"\u200dMDESCRBRACKET{len(preserved_runs)}\u200d"
            preserved_runs[token] = match.group(1)
            return token

        # HTMLToMarkdown may already have emitted a stable escaped \] sequence on
        # a previous pass. Preserve those runs so repeated roundtrips do not
        # multiply backslashes after a preserved \[...\] math segment.
        segment = _ESCAPED_RIGHT_BRACKET_RUN_RE.sub(_preserve, segment)
        if "\\" in segment:
            segment = segment.replace("\\", _MD_SPECIAL_CHARS["\\"])
        for char, placeholder in _MD_SPECIAL_CHARS.items():
            if char != "\\":
                segment = segment.replace(char, placeholder)
        for token, original in preserved_runs.items():
            segment = segment.replace(token, original)
        return segment

    # Skip escaping inside explicit LaTeX delimiters.
    if r"\(" not in text and r"\[" not in text:
        return _escape_segment(text)

    parts = []
    last_end = 0
    for match in _MATH_PATTERN.finditer(text):
        parts.append(_escape_segment(text[last_end : match.start()]))
        parts.append(match.group(0))
        last_end = match.end()
    parts.append(_escape_segment(text[last_end:]))
    return "".join(parts)


def _protect_literal_chars(html: str) -> str:
    """Replace special markdown chars in plain text with placeholders."""
    if not _looks_like_html(html):
        return _escape_special_chars(html)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
        soup = BeautifulSoup(html, "html.parser")

    for text_node in soup.find_all(string=True):
        # Skip if inside a tag that's already handled by markdown syntax
        parent = text_node.parent
        if parent is not None and parent.name in _PROTECTED_TAGS:
            continue

        text_node.replace_with(_escape_special_chars(str(text_node)))

    return str(soup)


def _restore_escaped_chars(md: str) -> str:
    """Restore placeholders as escaped markdown characters."""
    for char, placeholder in _MD_SPECIAL_CHARS.items():
        # Restore as escaped characters (e.g. \* or \\)
        # This ensures that when passed back to MarkdownToHTML,
        # they are treated literally.
        md = md.replace(placeholder, "\\" + char)
    return md


def _prepare_custom_tag_placeholders(html: str) -> tuple[str, dict[str, str]]:
    """Replace HTML tags that need Anki-specific markdown with stable placeholders."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
        soup = BeautifulSoup(html, "html.parser")

    replacements: dict[str, str] = {}
    counter = 0

    def _token() -> str:
        nonlocal counter
        token = f"ANKIOPSTOKEN{counter}END"
        counter += 1
        return token

    for image in soup.find_all("img"):
        src = image.get("src", "")
        alt = image.get("alt", "")
        style = image.get("style", "")
        width_match = re.search(r"width:\s*([\d.]+)px", style)
        width_attr = (
            f"{{width={int(float(width_match.group(1)))}}}" if width_match else ""
        )
        token = _token()
        replacements[token] = f"![{alt}](<{LOCAL_MEDIA_DIR}/{src}>){width_attr}"
        image.replace_with(token)

    for underline in soup.find_all("u"):
        content = underline.get_text().strip()
        token = _token()
        replacements[token] = f"<u>{content}</u>" if content else ""
        underline.replace_with(token)

    for br in soup.find_all("br"):
        # Keep blockquote <br> tags intact so html_to_markdown can emit
        # proper continuation quote markers ("> " on each line).
        if br.find_parent("blockquote") is not None:
            continue
        token = _token()
        replacements[token] = "\n"
        br.replace_with(token)

    return str(soup), replacements


def _restore_custom_tag_placeholders(md: str, replacements: dict[str, str]) -> str:
    """Restore placeholder tokens back to custom markdown fragments."""
    for token, output in replacements.items():
        md = md.replace(token, output)
    return md


def _parse_link_destination(
    md: str,
    start: int,
) -> tuple[int, str, bool] | None:
    """Parse markdown link destination starting right after '('.

    Returns a tuple of:
    - end index (exclusive) right after ')'
    - destination text without surrounding angle brackets
    - whether destination was already angle-bracket wrapped
    """
    if start >= len(md):
        return None

    # Already wrapped destination: [text](<url>)
    if md[start] == "<":
        i = start + 1
        backslash_run = 0
        while i < len(md):
            char = md[i]
            is_escaped = (backslash_run % 2) == 1
            if char == "\n":
                return None
            if char == ">" and not is_escaped:
                if i + 1 < len(md) and md[i + 1] == ")":
                    return (i + 2, md[start + 1 : i], True)
                return None
            if char == "\\":
                backslash_run += 1
            else:
                backslash_run = 0
            i += 1
        return None

    # Plain destination: [text](url(with(parens)))
    paren_depth = 0
    i = start
    backslash_run = 0
    while i < len(md):
        char = md[i]
        is_escaped = (backslash_run % 2) == 1
        if char == "\n":
            return None
        if char == "(" and not is_escaped:
            paren_depth += 1
        elif char == ")" and not is_escaped:
            if paren_depth == 0:
                return (i + 1, md[start:i], False)
            paren_depth -= 1
        if char == "\\":
            backslash_run += 1
        else:
            backslash_run = 0
        i += 1
    return None


def _enforce_link_angle_brackets(md: str) -> str:
    """Ensure markdown links always use angle-bracket destinations."""
    if "[" not in md or "](" not in md:
        return md

    parts: list[str] = []
    label_stack: list[int] = []
    cursor = 0
    i = 0
    changed = False
    backslash_run = 0

    while i < len(md):
        char = md[i]
        is_escaped = (backslash_run % 2) == 1

        if char == "\n":
            label_stack.clear()
        elif char == "[" and not is_escaped:
            label_stack.append(i)
        elif char == "]" and not is_escaped and label_stack:
            label_start = label_stack.pop()
            if i + 1 < len(md) and md[i + 1] == "(":
                parsed = _parse_link_destination(md, i + 2)
                if parsed is not None:
                    link_end, destination, is_wrapped = parsed
                    parts.append(md[cursor:label_start])
                    if is_wrapped:
                        parts.append(md[label_start:link_end])
                    else:
                        parts.extend((md[label_start : i + 2], "<", destination, ">)"))
                        changed = True
                    cursor = link_end
                    i = link_end
                    label_stack.clear()
                    backslash_run = 0
                    continue

        if char == "\\":
            backslash_run += 1
        else:
            backslash_run = 0
        i += 1

    if not changed:
        return md

    parts.append(md[cursor:])
    return "".join(parts)


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

        is_html_input = _looks_like_html(html)

        # Normalize multiple <br> before blockquotes to ensure stable round-trips
        # The html-to-markdown library treats <br><br><blockquote> incorrectly,
        # producing only \n> instead of \n\n>. We normalize to single <br>.
        if is_html_input:
            html = re.sub(
                r"(<br>\s*)+(<blockquote>)", r"<br>\2", html, flags=re.IGNORECASE
            )

        # Protect literal characters before conversion
        html = _protect_literal_chars(html)

        if is_html_input:
            html, replacements = _prepare_custom_tag_placeholders(html)
            md = convert(html, self._OPTIONS)
            md = _restore_custom_tag_placeholders(md, replacements)
            md = _enforce_link_angle_brackets(md)
        else:
            md = html

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
