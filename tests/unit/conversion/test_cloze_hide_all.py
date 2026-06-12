"""Static contracts for the ClozeHideAll Anki templates."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def template_dir() -> Path:
    return Path(__file__).parents[3] / "ankiops/note_types/AnkiOpsClozeHideAll"


@pytest.mark.parametrize(
    ("filename", "raw_field_tag"),
    [
        ("Front.template.anki", "{{Text}}"),
        ("Back.template.anki", "{{edit:Text}}"),
    ],
)
def test_templates_keep_one_rendered_cloze_probe_and_one_raw_source(
    template_dir, filename, raw_field_tag
):
    template = (template_dir / filename).read_text(encoding="utf-8")

    assert template.count("{{cloze:Text}}") == 1
    assert template.count(raw_field_tag) == 1
    assert 'id="cloze_rendered"' in template
    assert 'id="hidden_raw"' in template
    assert 'id="content_display"' in template


def test_back_template_keeps_show_all_toggle(template_dir):
    template = (template_dir / "Back.template.anki").read_text(encoding="utf-8")

    assert 'id="cloze_toggle"' in template
    assert "Show All" in template
    assert "Hide All" in template


@pytest.mark.parametrize("filename", ["Front.template.anki", "Back.template.anki"])
def test_templates_prefer_anki_cloze_ordinal_before_body_class_fallback(
    template_dir, filename
):
    template = (template_dir / filename).read_text(encoding="utf-8")

    ordinal_pos = template.index("activeOrdinalFromRendered")
    signature_pos = template.index("var renderedSig = normalizeSignature(rendered)")
    body_class_pos = template.index("document.body.className")

    assert ordinal_pos < signature_pos
    assert ordinal_pos < body_class_pos


@pytest.mark.parametrize("filename", ["Front.template.anki", "Back.template.anki"])
def test_templates_use_anki_ordinal_only_for_active_known_clozes(
    template_dir, filename
):
    template = (template_dir / filename).read_text(encoding="utf-8")

    assert 'querySelectorAll(".cloze[data-ordinal]")' in template
    assert '!hasClass(el, "cloze-inactive")' in template
    assert "indices.indexOf(ordinal) !== -1" in template


@pytest.mark.parametrize(
    ("filename", "simulation"),
    [
        ("Front.template.anki", "simulateFront"),
        ("Back.template.anki", "simulateBack"),
    ],
)
def test_templates_preserve_active_cloze_boundaries_for_duplicate_content_detection(
    template_dir, filename, simulation
):
    template = (template_dir / filename).read_text(encoding="utf-8")

    rendered_signature = "var renderedSig = normalizeSignature(rendered)"
    simulated_signature = f"normalizeSignature({simulation}(raw, idx))"

    assert 'hasClass(node, "cloze") && !hasClass(node, "cloze-inactive")' in template
    assert 'return "[[CLOZE]]" + out + "[[/CLOZE]]";' in template
    assert template.index(rendered_signature) < template.index(simulated_signature)
