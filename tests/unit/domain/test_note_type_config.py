"""Validation and inference tests for note-type configuration rules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from ankiops.fs import FileSystemAdapter
from ankiops.models import (
    ANKIOPS_KEY_FIELD,
    Field,
    NoteTypeConfig,
)


def _write_basic_template_pair(note_type_dir: Path) -> None:
    (note_type_dir / "Front.template.anki").write_text("{{Term}}", encoding="utf-8")
    (note_type_dir / "Back.template.anki").write_text(
        "{{FrontSide}}<hr id=answer>{{Definition}}",
        encoding="utf-8",
    )


def test_label_reuse_same_field_and_identifying_is_allowed():
    """Label reuse is allowed globally when field and identifying are identical."""
    config_a = NoteTypeConfig(
        name="TypeA",
        fields=[Field("Question", "Q:", identifying=True), ANKIOPS_KEY_FIELD],
    )
    config_b = NoteTypeConfig(
        name="TypeB",
        fields=[
            Field("Question", "Q:", identifying=True),
            Field("Context", "CTX:", identifying=True),
        ],
    )
    NoteTypeConfig.validate_configs([config_a, config_b])


def test_label_reuse_conflicting_identifying_fails():
    config_a = NoteTypeConfig(
        name="TypeA",
        fields=[Field("Extra", "E:", identifying=True)],
    )
    config_b = NoteTypeConfig(
        name="TypeB",
        fields=[Field("Extra", "E:", identifying=False)],
    )
    with pytest.raises(ValueError, match="conflicting identifying flag"):
        NoteTypeConfig.validate_configs([config_a, config_b])


def test_register_conflict_with_reserved():
    """Ensure we can register a custom type with AnkiOps Key if it's identical."""
    config = NoteTypeConfig(
        name="ConflictType",
        fields=[ANKIOPS_KEY_FIELD],  # Exact same field properties
    )
    # Should not raise
    NoteTypeConfig.validate_configs([config])


def test_mandatory_set_collision():
    """Ensure two note types cannot have identical identifying fields."""
    config1 = NoteTypeConfig(
        name="AnkiOpsType1",
        fields=[
            Field("Prop", "P:", identifying=True),
            Field("Val", "V:", identifying=True),
        ],
    )
    config2 = NoteTypeConfig(
        name="AnkiOpsType2",
        fields=[
            Field("Prop", "P:", identifying=True),
            Field("Val", "V:", identifying=True),
        ],
    )

    with pytest.raises(ValueError, match="identical identifying fields"):
        NoteTypeConfig.validate_configs([config1, config2])


def test_load_custom_from_dir():
    """Runtime loading uses local note_types only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        note_types = Path(tmpdir)

        (note_types / "MyCustomType").mkdir(parents=True, exist_ok=True)
        yaml_path = note_types / "MyCustomType" / "note_type.yaml"

        data = {
            "fields": [
                {"name": "Term", "label": "TM:", "identifying": True},
                {"name": "Definition", "label": "D:", "identifying": True},
            ],
            "styling": "AnkiOpsStyling.css",
        }
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")
        (note_types / "MyCustomType" / "AnkiOpsStyling.css").write_text(
            ".custom { color: red; }", encoding="utf-8"
        )
        _write_basic_template_pair(note_types / "MyCustomType")

        fs = FileSystemAdapter()
        configs = fs.load_note_type_configs(note_types)

        assert len(configs) == 1
        config = configs[0]

        assert sorted(
            [
                field_config.name
                for field_config in config.fields
                if field_config.identifying
            ]
        ) == [
            "Definition",
            "Term",
        ]


def test_note_type_load_missing_dir_fails(tmp_path):
    fs = FileSystemAdapter()
    with pytest.raises(ValueError, match="directory not found"):
        fs.load_note_type_configs(tmp_path / "missing_note_types")


def test_note_type_load_empty_dir_fails(tmp_path):
    note_types = tmp_path / "note_types"
    note_types.mkdir(parents=True, exist_ok=True)

    fs = FileSystemAdapter()
    with pytest.raises(ValueError, match="No note type definitions found"):
        fs.load_note_type_configs(note_types)


def test_note_type_subdir_missing_note_type_yaml_fails(tmp_path):
    note_types = tmp_path / "note_types"
    missing_config = note_types / "BadType"
    missing_config.mkdir(parents=True, exist_ok=True)

    fs = FileSystemAdapter()
    with pytest.raises(ValueError, match="is missing note_type.yaml"):
        fs.load_note_type_configs(note_types)


def test_require_styling_key():
    """Test that missing mandatory styling in per-folder YAML raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        note_types = Path(tmpdir)
        subdir = note_types / "BadType"
        subdir.mkdir()
        yaml_path = subdir / "note_type.yaml"
        # Missing 'styling' key
        data = {"fields": [{"name": "XField", "label": "XF:", "identifying": True}]}
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")

        fs = FileSystemAdapter()
        with pytest.raises(ValueError, match="missing mandatory 'styling' key"):
            fs.load_note_type_configs(note_types)


def test_require_styling_key_for_ankiops_name_too():
    """No namespace-based exception: AnkiOps* names still require styling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        note_types = Path(tmpdir)
        subdir = note_types / "AnkiOpsPretend"
        subdir.mkdir()
        yaml_path = subdir / "note_type.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "fields": [
                        {
                            "name": "Question",
                            "label": "QX:",
                            "identifying": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        fs = FileSystemAdapter()
        with pytest.raises(ValueError, match="missing mandatory 'styling' key"):
            fs.load_note_type_configs(note_types)


def test_reject_name_key_in_note_type_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        note_types = Path(tmpdir)
        subdir = note_types / "BadType"
        subdir.mkdir()
        yaml_path = subdir / "note_type.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "name": "DifferentName",
                    "styling": "AnkiOpsStyling.css",
                    "fields": [
                        {"name": "XField", "label": "XF:", "identifying": True}
                    ],
                }
            ),
            encoding="utf-8",
        )
        (subdir / "AnkiOpsStyling.css").write_text(".x{}", encoding="utf-8")

        fs = FileSystemAdapter()
        with pytest.raises(ValueError, match="must not define 'name'"):
            fs.load_note_type_configs(note_types)


def test_missing_styling_file_fails(tmp_path):
    note_types = tmp_path / "note_types"
    subdir = note_types / "BrokenType"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "note_type.yaml").write_text(
        yaml.dump(
            {
                "fields": [{"name": "Term", "label": "TM:", "identifying": True}],
                "styling": "Missing.css",
            }
        ),
        encoding="utf-8",
    )
    _write_basic_template_pair(subdir)

    fs = FileSystemAdapter()
    with pytest.raises(ValueError, match="references missing styling file"):
        fs.load_note_type_configs(note_types)


def test_missing_template_file_fails(tmp_path):
    note_types = tmp_path / "note_types"
    subdir = note_types / "BrokenType"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "note_type.yaml").write_text(
        yaml.dump(
            {
                "fields": [{"name": "Term", "label": "TM:", "identifying": True}],
                "styling": "Style.css",
                "templates": [
                    {
                        "name": "Card 1",
                        "front": "Front.template.anki",
                        "back": "Back.template.anki",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (subdir / "Style.css").write_text(".card{}", encoding="utf-8")
    (subdir / "Front.template.anki").write_text("{{Term}}", encoding="utf-8")

    fs = FileSystemAdapter()
    with pytest.raises(ValueError, match="references missing template file"):
        fs.load_note_type_configs(note_types)


def test_note_type_load_uses_cache_when_unchanged(tmp_path, monkeypatch):
    note_types = tmp_path / "note_types"
    (note_types / "MyCustomType").mkdir(parents=True, exist_ok=True)
    (note_types / "MyCustomType" / "note_type.yaml").write_text(
        yaml.dump(
            {
                "fields": [
                    {"name": "Term", "label": "TM:", "identifying": True},
                    {"name": "Definition", "label": "D:", "identifying": True},
                ],
                "styling": "AnkiOpsStyling.css",
            }
        ),
        encoding="utf-8",
    )
    (note_types / "MyCustomType" / "AnkiOpsStyling.css").write_text(
        ".custom { color: red; }", encoding="utf-8"
    )
    _write_basic_template_pair(note_types / "MyCustomType")

    calls = {"count": 0}
    original = yaml.safe_load

    def _counting_safe_load(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(yaml, "safe_load", _counting_safe_load)

    fs = FileSystemAdapter()
    fs.load_note_type_configs(note_types)
    first_call_count = calls["count"]
    assert first_call_count > 0

    fs.load_note_type_configs(note_types)
    assert calls["count"] == first_call_count


def test_note_type_load_cache_invalidates_on_file_change(tmp_path, monkeypatch):
    note_types = tmp_path / "note_types"
    (note_types / "MyCustomType").mkdir(parents=True, exist_ok=True)
    yaml_path = note_types / "MyCustomType" / "note_type.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "fields": [
                    {"name": "Term", "label": "TM:", "identifying": True},
                    {"name": "Definition", "label": "D:", "identifying": True},
                ],
                "styling": "AnkiOpsStyling.css",
            }
        ),
        encoding="utf-8",
    )
    css_path = note_types / "MyCustomType" / "AnkiOpsStyling.css"
    css_path.write_text(".custom { color: red; }", encoding="utf-8")
    _write_basic_template_pair(note_types / "MyCustomType")

    calls = {"count": 0}
    original = yaml.safe_load

    def _counting_safe_load(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(yaml, "safe_load", _counting_safe_load)

    fs = FileSystemAdapter()
    fs.load_note_type_configs(note_types)
    first_call_count = calls["count"]

    css_path.write_text(".custom { color: blue; }", encoding="utf-8")
    fs.load_note_type_configs(note_types)
    assert calls["count"] > first_call_count


def test_label_collision_different_field():
    """A label cannot map to two different field names globally."""
    config1 = NoteTypeConfig(
        name="Type1",
        fields=[Field("FieldA", "X:", identifying=True)],
    )
    config2 = NoteTypeConfig(
        name="Type2",
        fields=[Field("FieldB", "X:", identifying=True)],
    )
    with pytest.raises(ValueError, match="maps to both"):
        NoteTypeConfig.validate_configs([config1, config2])


def test_label_sharing_same_field_and_identifying_is_valid():
    """Sharing a label is valid when both field name and identifying match."""
    config1 = NoteTypeConfig(
        name="Type1",
        fields=[Field("Description", "D:", identifying=True)],
    )
    config2 = NoteTypeConfig(
        name="Type2",
        fields=[
            Field("Description", "D:", identifying=True),
            Field("Other", "O:", identifying=True),
        ],
    )
    NoteTypeConfig.validate_configs([config1, config2])


def test_subset_inference():
    """Register a type that is a superset of another, verify inference."""
    base_domain = NoteTypeConfig(
        name="BaseType",
        fields=[
            Field("BaseQ", "BQ:", identifying=True),
            Field("BaseA", "BA:", identifying=True),
        ],
    )
    super_domain = NoteTypeConfig(
        name="SupersetType",
        fields=[
            Field("BaseQ", "SQ:", identifying=True),
            Field("BaseA", "SA:", identifying=True),
            Field("Context", "C:", identifying=True),
        ],
    )
    NoteTypeConfig.validate_configs([base_domain, super_domain])

    fs = FileSystemAdapter()
    fs.set_configs([base_domain, super_domain])

    # Fields for Superset
    fields_super = {"BaseQ": "q", "BaseA": "c", "Context": "c"}
    assert fs._infer_note_type(fields_super) == "SupersetType"

    # Fields for Base (subset)
    fields_base = {"BaseQ": "q", "BaseA": "a"}
    assert fs._infer_note_type(fields_base) == "BaseType"


def test_reserved_ankiops_key():
    """Ensure reserved AnkiOps Key name cannot be used."""
    config = NoteTypeConfig(
        name="BadName",
        fields=[Field(ANKIOPS_KEY_FIELD.name, "X:", identifying=False)],
    )
    with pytest.raises(ValueError, match="uses reserved field name"):
        NoteTypeConfig.validate_configs([config])


def test_duplicate_field_names_within_note_type_fail():
    config = NoteTypeConfig(
        name="DuplicateNames",
        fields=[
            Field("Question", "Q1:", identifying=True),
            Field("Question", "Q2:", identifying=False),
        ],
    )
    with pytest.raises(ValueError, match="duplicate field name"):
        NoteTypeConfig.validate_configs([config])


def test_duplicate_field_labels_within_note_type_fail():
    config = NoteTypeConfig(
        name="DuplicatePrefixes",
        fields=[
            Field("Question", "Q:", identifying=True),
            Field("Answer", "Q:", identifying=True),
        ],
    )
    with pytest.raises(ValueError, match="duplicate field label"):
        NoteTypeConfig.validate_configs([config])


# --- Inference Tests ---


def test_inference_qa_vs_choice():
    """Verify {Q, A} matches QA (tighter) over Choice."""
    qa = NoteTypeConfig(
        "QA",
        fields=[
            Field("Question", "Q:", identifying=True),
            Field("Answer", "A:", identifying=True),
        ],
    )
    choice = NoteTypeConfig(
        "Choice",
        fields=[
            Field("Question", "QQ:", identifying=True),
            Field("Answer", "AA:", identifying=True),
            Field("Choice 1", "C1:", identifying=True),
        ],
        is_choice=True,
    )
    NoteTypeConfig.validate_configs([qa, choice])
    fs = FileSystemAdapter()
    fs.set_configs([qa, choice])

    # {Q, A} -> QA (size 2) vs Choice (size > 2) -> QA wins
    qa_fields = {"Question": "val", "Answer": "val"}
    assert fs._infer_note_type(qa_fields) == "QA"

    # {Q, A, C1} -> Choice vs QA (rejection) -> Choice wins
    choice_fields = {"Question": "val", "Answer": "val", "Choice 1": "val"}
    assert fs._infer_note_type(choice_fields) == "Choice"


def test_inference_choice_resilience():
    """Verify Choice note is detected even if some choices are missing."""
    choice = NoteTypeConfig(
        "Choice",
        fields=[
            Field("Question", "Q:", identifying=True),
            Field("Answer", "A:", identifying=True),
            Field("Choice 1", "C1:", identifying=True),
            Field("Choice 2", "C2:", identifying=True),
            Field("Choice 5", "C5:", identifying=True),
            Field("Choice 6", "C6:", identifying=True),
        ],
        is_choice=True,
    )
    NoteTypeConfig.validate_configs([choice])
    fs = FileSystemAdapter()
    fs.set_configs([choice])

    # {Q, A, C5, C6} -> Choice
    note_fields = {"Question": "v", "Answer": "v", "Choice 5": "v", "Choice 6": "v"}
    assert fs._infer_note_type(note_fields) == "Choice"


def test_inference_with_non_identifying_fields():
    """Verify non-identifying fields don't interfere with inference."""
    qa = NoteTypeConfig(
        "QA",
        fields=[
            Field("Question", "Q:", identifying=True),
            Field("Answer", "A:", identifying=True),
            Field("Extra", "E:", identifying=False),
        ],
    )
    NoteTypeConfig.validate_configs([qa])
    fs = FileSystemAdapter()
    fs.set_configs([qa])

    # {Q, A, Extra} -> matches QA
    note_fields = {"Question": "v", "Answer": "v", "Extra": "ex"}
    assert fs._infer_note_type(note_fields) == "QA"


def test_inference_unknown_field_fails():
    """Verify that unknown fields cause inference failure."""
    qa = NoteTypeConfig(
        "QA",
        fields=[
            Field("Question", "Q:", identifying=True),
            Field("Answer", "A:", identifying=True),
        ],
    )
    NoteTypeConfig.validate_configs([qa])
    fs = FileSystemAdapter()
    fs.set_configs([qa])

    # {Q, A, Random} -> Fails
    note_fields = {"Question": "v", "Answer": "v", "Random": "r"}
    with pytest.raises(ValueError, match="Cannot determine note type"):
        fs._infer_note_type(note_fields)


def test_choice_validation_requires_choice_field_name():
    config = NoteTypeConfig(
        name="BadChoice",
        is_choice=True,
        fields=[
            Field("MyQuestion", "XQ:", identifying=True),
            Field("MyAnswer", "XA:", identifying=True),
        ],
    )
    with pytest.raises(ValueError, match="containing the word 'choice' were found"):
        NoteTypeConfig.validate_configs([config])


def test_choice_validation_requires_identifying_non_choice_field():
    config = NoteTypeConfig(
        name="BadChoice",
        is_choice=True,
        fields=[
            Field("Question", "QX:", identifying=False),
            Field("Choice 1", "CX1:", identifying=True),
        ],
    )
    with pytest.raises(ValueError, match="no identifying non-choice field"):
        NoteTypeConfig.validate_configs([config])


def test_choice_validation_requires_identifying_choice_field():
    config = NoteTypeConfig(
        name="BadChoice",
        is_choice=True,
        fields=[
            Field("Question", "QX:", identifying=True),
            Field("Choice 1", "CX1:", identifying=False),
        ],
    )
    with pytest.raises(ValueError, match="no identifying choice field"):
        NoteTypeConfig.validate_configs([config])


def test_choice_validation_passes_with_identifying_base_and_choice():
    config = NoteTypeConfig(
        name="GoodChoice",
        is_choice=True,
        fields=[
            Field("Question", "XQ:", identifying=True),
            Field("Choice 1", "XC1:", identifying=True),
        ],
    )
    NoteTypeConfig.validate_configs([config])
