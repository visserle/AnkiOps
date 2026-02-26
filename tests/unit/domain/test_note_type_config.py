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


def test_register_custom_conflict_with_builtin():
    """Ensure custom types cannot use built-in prefixes."""
    builtin = NoteTypeConfig(
        name="AnkiOpsQA",
        fields=[Field("Question", "Q:", identifying=True), ANKIOPS_KEY_FIELD],
    )
    custom = NoteTypeConfig(
        name="BadCustom",
        fields=[
            Field("Question", "Q:", identifying=True),
            Field("Another", "X:", identifying=True),
        ],
    )
    with pytest.raises(ValueError, match="uses reserved built-in prefix"):
        NoteTypeConfig.validate_configs([builtin, custom])


def test_register_custom_conflict_with_builtin_optional_prefix():
    """Ensure custom types cannot use built-in optional prefixes either."""
    builtin = NoteTypeConfig(
        name="AnkiOpsQA",
        fields=[Field("Extra", "E:", identifying=False), ANKIOPS_KEY_FIELD],
    )
    custom = NoteTypeConfig(
        name="BadCustom",
        fields=[Field("Custom Extra", "E:", identifying=True)],
    )
    with pytest.raises(ValueError, match="uses reserved built-in prefix"):
        NoteTypeConfig.validate_configs([builtin, custom])


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
    """Test loading custom configurations from a directory using FileSystemAdapter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        note_types = Path(tmpdir)

        (note_types / "MyCustomType").mkdir(parents=True, exist_ok=True)
        yaml_path = note_types / "MyCustomType" / "note_type.yaml"

        data = {
            "fields": [
                {"name": "Term", "prefix": "TM:", "identifying": True},
                {"name": "Definition", "prefix": "D:", "identifying": True},
            ],
            "styling": "AnkiOpsStyling.css",
        }
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")
        (note_types / "MyCustomType" / "AnkiOpsStyling.css").write_text(
            ".custom { color: red; }", encoding="utf-8"
        )

        fs = FileSystemAdapter()
        configs = fs.load_note_type_configs(note_types)

        # 6 built-in types + 1 custom type = 7
        assert len(configs) == 7
        config = next(
            config_item for config_item in configs if config_item.name == "MyCustomType"
        )

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


def test_require_styling_key():
    """Test that missing mandatory styling in per-folder YAML raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        note_types = Path(tmpdir)
        subdir = note_types / "BadType"
        subdir.mkdir()
        yaml_path = subdir / "note_type.yaml"
        # Missing 'styling' key
        data = {"fields": [{"name": "XField", "prefix": "XF:", "identifying": True}]}
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")

        fs = FileSystemAdapter()
        with pytest.raises(ValueError, match="missing mandatory 'styling' key"):
            fs.load_note_type_configs(note_types)


def test_note_type_load_uses_cache_when_unchanged(tmp_path, monkeypatch):
    note_types = tmp_path / "note_types"
    (note_types / "MyCustomType").mkdir(parents=True, exist_ok=True)
    (note_types / "MyCustomType" / "note_type.yaml").write_text(
        yaml.dump(
            {
                "fields": [
                    {"name": "Term", "prefix": "TM:", "identifying": True},
                    {"name": "Definition", "prefix": "D:", "identifying": True},
                ],
                "styling": "AnkiOpsStyling.css",
            }
        ),
        encoding="utf-8",
    )
    (note_types / "MyCustomType" / "AnkiOpsStyling.css").write_text(
        ".custom { color: red; }", encoding="utf-8"
    )

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
                    {"name": "Term", "prefix": "TM:", "identifying": True},
                    {"name": "Definition", "prefix": "D:", "identifying": True},
                ],
                "styling": "AnkiOpsStyling.css",
            }
        ),
        encoding="utf-8",
    )
    css_path = note_types / "MyCustomType" / "AnkiOpsStyling.css"
    css_path.write_text(".custom { color: red; }", encoding="utf-8")

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


def test_prefix_collision_different_field():
    """Ensure custom note types cannot share the same prefix."""
    config1 = NoteTypeConfig(
        name="Type1",
        fields=[Field("FieldA", "X:", identifying=True)],
    )
    config2 = NoteTypeConfig(
        name="Type2",
        fields=[Field("FieldB", "X:", identifying=True)],
    )
    with pytest.raises(ValueError, match="reuses custom prefix"):
        NoteTypeConfig.validate_configs([config1, config2])


def test_prefix_sharing_custom_field_fails():
    """Ensure two custom note types cannot share a prefix even with same field name."""
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
    with pytest.raises(ValueError, match="reuses custom prefix"):
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


def test_duplicate_field_prefixes_within_note_type_fail():
    config = NoteTypeConfig(
        name="DuplicatePrefixes",
        fields=[
            Field("Question", "Q:", identifying=True),
            Field("Answer", "Q:", identifying=True),
        ],
    )
    with pytest.raises(ValueError, match="duplicate field prefix"):
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


def test_choice_validation():
    """Ensure choice note types must have at least one field with 'choice' in name."""
    # 1. is_choice=True, no choice field -> Fail
    config1 = NoteTypeConfig(
        name="BadChoice",
        is_choice=True,
        fields=[
            Field("MyQuestion", "XQ:", identifying=True),
            Field("MyAnswer", "XA:", identifying=True),
        ],
    )
    with pytest.raises(ValueError, match="containing the word 'choice' were found"):
        NoteTypeConfig.validate_configs([config1])

    # 2. is_choice=True, has choice field -> Pass
    config2 = NoteTypeConfig(
        name="GoodChoice",
        is_choice=True,
        fields=[
            Field("MyQuestion", "XQ:", identifying=True),
            Field("Choice 1", "XC1:", identifying=True),
        ],
    )
    NoteTypeConfig.validate_configs([config2])

    # 3. is_choice=False, no choice field -> Pass
    config3 = NoteTypeConfig(
        name="NormalQA",
        is_choice=False,
        fields=[
            Field("MyQuestion", "XQ:", identifying=True),
            Field("MyAnswer", "XA:", identifying=True),
        ],
    )
    NoteTypeConfig.validate_configs([config3])
