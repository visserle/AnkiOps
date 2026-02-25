"""Tests for ClozeHideAll active cloze index detection.

The AnkiOpsClozeHideAll templates use JavaScript to detect which cloze
index is currently active, then render all clozes as hidden except the
active one. This module reimplements both Anki's cloze rendering and the
JS detection algorithm in Python to verify correctness — especially for
edge cases like duplicate cloze content.

How Anki renders {{cloze:Field}}:
- Front: active cloze → <span class="cloze">[...]</span> (or [hint]),
         inactive clozes → plain content (markers stripped)
- Back:  active cloze → <span class="cloze">content</span>,
         inactive clozes → plain content (markers stripped)
"""

import re

import pytest


# ---------------------------------------------------------------------------
# Anki cloze renderer (mimics Anki's {{cloze:Field}} substitution)
# ---------------------------------------------------------------------------


def anki_render_cloze(raw: str, active_idx: int, side: str) -> str:
    """Mimic Anki's {{cloze:Field}} rendering.

    Args:
        raw: The raw field content with cloze markers, e.g. "{{c1::dog}} chases {{c2::cat}}"
        active_idx: The cloze index being tested (1-based)
        side: "front" or "back"

    Returns:
        The HTML that Anki would produce for {{cloze:Field}}.
    """

    def replace_cloze(m: re.Match) -> str:
        idx = int(m.group(1))
        inner = m.group(2)
        # inner may contain a hint after "::"
        parts = inner.split("::", 1)
        content = parts[0]
        hint = parts[1] if len(parts) > 1 else ""

        if idx == active_idx:
            if side == "front":
                display = f"[{hint}]" if hint else "[...]"
                return f'<span class="cloze">{display}</span>'
            else:
                return f'<span class="cloze">{content}</span>'
        else:
            # Inactive clozes: markers stripped, plain content shown
            return content

    return re.sub(r"\{\{c(\d+)::([\s\S]*?)\}\}", replace_cloze, raw)


# ---------------------------------------------------------------------------
# Detection algorithms (Python ports of the JS in the templates)
# ---------------------------------------------------------------------------


def detect_active_idx_front(raw: str, rendered: str) -> str:
    """Port of the front-template detection algorithm.

    Strategy: for each candidate index, simulate Anki's front-side rendering.
    The one whose text-only output matches the actual rendered text is active.
    """
    indices = list(dict.fromkeys(re.findall(r"\{\{c(\d+)::", raw)))
    active_idx = indices[0] if indices else "1"

    def text_only(html: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", "", html)).strip()

    rendered_text = text_only(rendered)

    for test_idx in indices:

        def replace_fn(m: re.Match, _idx=test_idx) -> str:
            idx = m.group(1)
            inner = m.group(2)
            parts = inner.split("::", 1)
            content = parts[0]
            hint = parts[1] if len(parts) > 1 else ""
            if idx == _idx:
                return f"[{hint}]" if hint else "[...]"
            else:
                return content

        simulated = re.sub(r"\{\{c(\d+)::([\s\S]*?)\}\}", replace_fn, raw)
        if text_only(simulated) == rendered_text:
            active_idx = test_idx
            break

    return active_idx


def detect_active_idx_back(raw: str, rendered: str) -> str:
    """Port of the back-template detection algorithm.

    Strategy: for each candidate index, simulate Anki's back-side rendering
    (active cloze wrapped in <span class="cloze">, inactive as plain text).
    Compare normalized HTML structure to find the match.
    """
    indices = list(dict.fromkeys(re.findall(r"\{\{c(\d+)::", raw)))
    active_idx = indices[0] if indices else "1"

    def normalize_html(html: str) -> str:
        return re.sub(r"\s+", " ", html).replace("'", '"').strip()

    rendered_norm = normalize_html(rendered)

    for test_idx in indices:

        def replace_fn(m: re.Match, _idx=test_idx) -> str:
            idx = m.group(1)
            inner = m.group(2)
            content = inner.split("::", 1)[0]
            if idx == _idx:
                return f'<span class="cloze">{content}</span>'
            return content

        simulated = re.sub(r"\{\{c(\d+)::([\s\S]*?)\}\}", replace_fn, raw)
        if normalize_html(simulated) == rendered_norm:
            active_idx = test_idx
            break

    return active_idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Each test case: (description, raw_cloze, list_of_active_indices_to_test)
CASES = [
    (
        "simple two clozes",
        "{{c1::dog}} chases {{c2::cat}}",
        [1, 2],
    ),
    (
        "duplicate content (the bug case)",
        "{{c1::the}} and {{c2::the}}",
        [1, 2],
    ),
    (
        "with hint",
        "{{c1::Paris::capital of France}} is nice",
        [1],
    ),
    (
        "single cloze",
        "{{c1::only}}",
        [1],
    ),
    (
        "three clozes",
        "{{c1::A}} {{c2::B}} {{c3::C}}",
        [1, 2, 3],
    ),
    (
        "three identical (worst case)",
        "{{c1::X}} {{c2::X}} {{c3::X}}",
        [1, 2, 3],
    ),
    (
        "nested hint colons",
        "{{c1::a::b::c}} text",
        [1],
    ),
    (
        "multiple clozes with same index",
        "{{c1::dog}} and {{c1::cat}} are animals",
        [1],
    ),
    (
        "same index mixed with different index",
        "{{c1::dog}} and {{c1::cat}} chase {{c2::mouse}}",
        [1, 2],
    ),
    (
        "duplicate content across shared index",
        "{{c1::the}} {{c2::quick}} {{c1::the}}",
        [1, 2],
    ),
    (
        "three indices two shared",
        "{{c1::A}} {{c2::B}} {{c1::C}} {{c3::D}}",
        [1, 2, 3],
    ),
]


def _case_ids():
    """Generate pytest IDs from case descriptions."""
    return [case[0] for case in CASES]


def _expand_cases():
    """Expand cases into (raw, active_idx) pairs for parametrize."""
    expanded = []
    for desc, raw, indices in CASES:
        for idx in indices:
            expanded.append(pytest.param(raw, idx, id=f"{desc} [c{idx}]"))
    return expanded


class TestFrontDetection:
    """Test that the front-side detection algorithm correctly identifies the active cloze."""

    @pytest.mark.parametrize("raw, active_idx", _expand_cases())
    def test_detect_active_index(self, raw, active_idx):
        rendered = anki_render_cloze(raw, active_idx, "front")
        detected = detect_active_idx_front(raw, rendered)
        assert detected == str(active_idx), (
            f"Front: expected c{active_idx}, detected c{detected}\n"
            f"  raw:      {raw}\n"
            f"  rendered: {rendered}"
        )


class TestBackDetection:
    """Test that the back-side detection algorithm correctly identifies the active cloze."""

    @pytest.mark.parametrize("raw, active_idx", _expand_cases())
    def test_detect_active_index(self, raw, active_idx):
        rendered = anki_render_cloze(raw, active_idx, "back")
        detected = detect_active_idx_back(raw, rendered)
        assert detected == str(active_idx), (
            f"Back: expected c{active_idx}, detected c{detected}\n"
            f"  raw:      {raw}\n"
            f"  rendered: {rendered}"
        )


class TestAnkiRenderer:
    """Sanity-check the Anki renderer itself to ensure it mimics Anki faithfully."""

    def test_front_active_becomes_blank(self):
        result = anki_render_cloze("{{c1::dog}} chases {{c2::cat}}", 1, "front")
        assert '<span class="cloze">[...]</span>' in result
        assert "cat" in result  # inactive shown as plain text
        assert "{{" not in result  # markers stripped

    def test_front_hint_shown(self):
        result = anki_render_cloze("{{c1::Paris::capital}}", 1, "front")
        assert "[capital]" in result

    def test_back_active_revealed(self):
        result = anki_render_cloze("{{c1::dog}} chases {{c2::cat}}", 1, "back")
        assert '<span class="cloze">dog</span>' in result
        assert "cat" in result  # inactive shown as plain text
        assert "{{" not in result

    def test_multiple_same_index_all_active(self):
        """When c1 is active, ALL c1 clozes should be revealed."""
        result = anki_render_cloze(
            "{{c1::dog}} and {{c1::cat}} are animals", 1, "back"
        )
        assert '<span class="cloze">dog</span>' in result
        assert '<span class="cloze">cat</span>' in result

    def test_multiple_same_index_front_all_blanked(self):
        """When c1 is active on front, ALL c1 clozes should be blanked."""
        result = anki_render_cloze(
            "{{c1::dog}} and {{c1::cat}} are animals", 1, "front"
        )
        assert result.count('[...]') == 2
