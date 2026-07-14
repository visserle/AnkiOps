"""Validation and inference tests for note-type configuration rules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from ankiops.note_types import (
    ANKIOPS_KEY_FIELD,
    NoteField,
    NoteType,
)
from tests.support.deck_files import DeckFileHarness


def _write_basic_template_pair(note_type_dir: Path) -> None:
    (note_type_dir / "Front.template.anki").write_text("{{Term}}", encoding="utf-8")
    (note_type_dir / "Back.template.anki").write_text(
        "{{FrontSide}}<hr id=answer>{{Definition}}",
        encoding="utf-8",
    )


def test_label_reuse_same_field_and_identifying_is_allowed():
    """Label reuse is allowed globally when field and identifying are identical."""
    config_a = NoteType(
        name="TypeA",
        fields=[NoteField("Question", "Q:", identifying=True), ANKIOPS_KEY_FIELD],
    )
    config_b = NoteType(
        name="TypeB",
        fields=[
            NoteField("Question", "Q:", identifying=True),
            NoteField("Context", "CTX:", identifying=True),
        ],
    )
    NoteType.validate_configs([config_a, config_b])


def test_label_reuse_conflicting_identifying_fails():
    config_a = NoteType(
        name="TypeA",
        fields=[NoteField("Extra", "E:", identifying=True)],
    )
    config_b = NoteType(
        name="TypeB",
        fields=[NoteField("Extra", "E:", identifying=False)],
    )
    with pytest.raises(ValueError, match="conflicting identifying flag"):
        NoteType.validate_configs([config_a, config_b])


def test_register_conflict_with_reserved():
    """Ensure we can register a custom type with AnkiOps Key if it's identical."""
    config = NoteType(
        name="ConflictType",
        fields=[ANKIOPS_KEY_FIELD],  # Exact same field properties
    )
    # Should not raise
    NoteType.validate_configs([config])


def test_mandatory_set_collision():
    """Ensure two note types cannot have identical identifying fields."""
    config1 = NoteType(
        name="AnkiOpsType1",
        fields=[
            NoteField("Prop", "P:", identifying=True),
            NoteField("Val", "V:", identifying=True),
        ],
    )
    config2 = NoteType(
        name="AnkiOpsType2",
        fields=[
            NoteField("Prop", "P:", identifying=True),
            NoteField("Val", "V:", identifying=True),
        ],
    )

    with pytest.raises(ValueError, match="identical identifying fields"):
        NoteType.validate_configs([config1, config2])


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

        fs = DeckFileHarness()
        configs = fs.load_note_types(note_types)

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
        assert config.source_files == frozenset(
            {
                Path("MyCustomType/note_type.yaml"),
                Path("MyCustomType/AnkiOpsStyling.css"),
                Path("MyCustomType/Front.template.anki"),
                Path("MyCustomType/Back.template.anki"),
            }
        )


def test_note_type_load_missing_dir_fails(tmp_path):
    fs = DeckFileHarness()
    with pytest.raises(ValueError, match="directory not found"):
        fs.load_note_types(tmp_path / "missing_note_types")


def test_note_type_load_empty_dir_fails(tmp_path):
    note_types = tmp_path / "note_types"
    note_types.mkdir(parents=True, exist_ok=True)

    fs = DeckFileHarness()
    with pytest.raises(ValueError, match="No note type definitions found"):
        fs.load_note_types(note_types)


def test_note_type_subdir_missing_note_type_yaml_fails(tmp_path):
    note_types = tmp_path / "note_types"
    missing_config = note_types / "BadType"
    missing_config.mkdir(parents=True, exist_ok=True)

    fs = DeckFileHarness()
    with pytest.raises(ValueError, match="is missing note_type.yaml"):
        fs.load_note_types(note_types)


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

        fs = DeckFileHarness()
        with pytest.raises(ValueError, match="missing mandatory 'styling' key"):
            fs.load_note_types(note_types)


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

        fs = DeckFileHarness()
        with pytest.raises(ValueError, match="missing mandatory 'styling' key"):
            fs.load_note_types(note_types)


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
                    "fields": [{"name": "XField", "label": "XF:", "identifying": True}],
                }
            ),
            encoding="utf-8",
        )
        (subdir / "AnkiOpsStyling.css").write_text(".x{}", encoding="utf-8")

        fs = DeckFileHarness()
        with pytest.raises(ValueError, match="must not define 'name'"):
            fs.load_note_types(note_types)


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

    fs = DeckFileHarness()
    with pytest.raises(ValueError, match="references missing styling file"):
        fs.load_note_types(note_types)


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

    fs = DeckFileHarness()
    with pytest.raises(ValueError, match="references missing template file"):
        fs.load_note_types(note_types)


def test_template_requires_both_sides(tmp_path):
    note_types = tmp_path / "note_types"
    subdir = note_types / "BrokenType"
    subdir.mkdir(parents=True)
    (subdir / "note_type.yaml").write_text(
        yaml.dump(
            {
                "fields": [{"name": "Term", "label": "TM:", "identifying": True}],
                "styling": "Style.css",
                "templates": [{"name": "Card 1", "front": "Front.template.anki"}],
            }
        ),
        encoding="utf-8",
    )
    (subdir / "Style.css").write_text(".card{}", encoding="utf-8")
    (subdir / "Front.template.anki").write_text("{{Term}}", encoding="utf-8")

    with pytest.raises(ValueError, match="template with invalid 'back'"):
        DeckFileHarness().load_note_types(note_types)


def test_label_collision_different_field():
    """A label cannot map to two different field names globally."""
    config1 = NoteType(
        name="Type1",
        fields=[NoteField("FieldA", "X:", identifying=True)],
    )
    config2 = NoteType(
        name="Type2",
        fields=[NoteField("FieldB", "X:", identifying=True)],
    )
    with pytest.raises(ValueError, match="maps to both"):
        NoteType.validate_configs([config1, config2])


def test_label_sharing_same_field_and_identifying_is_valid():
    """Sharing a label is valid when both field name and identifying match."""
    config1 = NoteType(
        name="Type1",
        fields=[NoteField("Description", "D:", identifying=True)],
    )
    config2 = NoteType(
        name="Type2",
        fields=[
            NoteField("Description", "D:", identifying=True),
            NoteField("Other", "O:", identifying=True),
        ],
    )
    NoteType.validate_configs([config1, config2])


def test_subset_inference():
    """Register a type that is a superset of another, verify inference."""
    base_domain = NoteType(
        name="BaseType",
        fields=[
            NoteField("BaseQ", "BQ:", identifying=True),
            NoteField("BaseA", "BA:", identifying=True),
        ],
    )
    super_domain = NoteType(
        name="SupersetType",
        fields=[
            NoteField("BaseQ", "SQ:", identifying=True),
            NoteField("BaseA", "SA:", identifying=True),
            NoteField("Context", "C:", identifying=True),
        ],
    )
    NoteType.validate_configs([base_domain, super_domain])

    fs = DeckFileHarness()
    fs.set_note_types([base_domain, super_domain])

    # Fields for Superset
    fields_super = {"BaseQ": "q", "BaseA": "c", "Context": "c"}
    assert fs._infer_note_type(fields_super) == "SupersetType"

    # Fields for Base (subset)
    fields_base = {"BaseQ": "q", "BaseA": "a"}
    assert fs._infer_note_type(fields_base) == "BaseType"


def test_reserved_ankiops_key():
    """Ensure reserved AnkiOps Key name cannot be used."""
    config = NoteType(
        name="BadName",
        fields=[NoteField(ANKIOPS_KEY_FIELD.name, "X:", identifying=False)],
    )
    with pytest.raises(ValueError, match="uses reserved field name"):
        NoteType.validate_configs([config])


def test_duplicate_field_names_within_note_type_fail():
    config = NoteType(
        name="DuplicateNames",
        fields=[
            NoteField("Question", "Q1:", identifying=True),
            NoteField("Question", "Q2:", identifying=False),
        ],
    )
    with pytest.raises(ValueError, match="duplicate field name"):
        NoteType.validate_configs([config])


def test_duplicate_field_labels_within_note_type_fail():
    config = NoteType(
        name="DuplicateLabels",
        fields=[
            NoteField("Question", "Q:", identifying=True),
            NoteField("Answer", "Q:", identifying=True),
        ],
    )
    with pytest.raises(ValueError, match="duplicate field label"):
        NoteType.validate_configs([config])


# --- Inference Tests ---


def test_inference_qa_vs_choice():
    """Verify {Q, A} matches QA (tighter) over Choice."""
    qa = NoteType(
        "QA",
        fields=[
            NoteField("Question", "Q:", identifying=True),
            NoteField("Answer", "A:", identifying=True),
        ],
    )
    choice = NoteType(
        "Choice",
        fields=[
            NoteField("Question", "QQ:", identifying=True),
            NoteField("Answer", "AA:", identifying=True),
            NoteField("Choice 1", "C1:", identifying=True),
        ],
        is_choice=True,
    )
    NoteType.validate_configs([qa, choice])
    fs = DeckFileHarness()
    fs.set_note_types([qa, choice])

    # {Q, A} -> QA (size 2) vs Choice (size > 2) -> QA wins
    qa_fields = {"Question": "val", "Answer": "val"}
    assert fs._infer_note_type(qa_fields) == "QA"

    # {Q, A, C1} -> Choice vs QA (rejection) -> Choice wins
    choice_fields = {"Question": "val", "Answer": "val", "Choice 1": "val"}
    assert fs._infer_note_type(choice_fields) == "Choice"


def test_inference_choice_resilience():
    """Verify Choice note is detected even if some choices are missing."""
    choice = NoteType(
        "Choice",
        fields=[
            NoteField("Question", "Q:", identifying=True),
            NoteField("Answer", "A:", identifying=True),
            NoteField("Choice 1", "C1:", identifying=True),
            NoteField("Choice 2", "C2:", identifying=True),
            NoteField("Choice 5", "C5:", identifying=True),
            NoteField("Choice 6", "C6:", identifying=True),
        ],
        is_choice=True,
    )
    NoteType.validate_configs([choice])
    fs = DeckFileHarness()
    fs.set_note_types([choice])

    # {Q, A, C5, C6} -> Choice
    note_fields = {"Question": "v", "Answer": "v", "Choice 5": "v", "Choice 6": "v"}
    assert fs._infer_note_type(note_fields) == "Choice"


def test_inference_with_non_identifying_fields():
    """Verify non-identifying fields don't interfere with inference."""
    qa = NoteType(
        "QA",
        fields=[
            NoteField("Question", "Q:", identifying=True),
            NoteField("Answer", "A:", identifying=True),
            NoteField("Extra", "E:", identifying=False),
        ],
    )
    NoteType.validate_configs([qa])
    fs = DeckFileHarness()
    fs.set_note_types([qa])

    # {Q, A, Extra} -> matches QA
    note_fields = {"Question": "v", "Answer": "v", "Extra": "ex"}
    assert fs._infer_note_type(note_fields) == "QA"


def test_inference_unknown_field_fails():
    """Verify that unknown fields cause inference failure."""
    qa = NoteType(
        "QA",
        fields=[
            NoteField("Question", "Q:", identifying=True),
            NoteField("Answer", "A:", identifying=True),
        ],
    )
    NoteType.validate_configs([qa])
    fs = DeckFileHarness()
    fs.set_note_types([qa])

    # {Q, A, Random} -> Fails
    note_fields = {"Question": "v", "Answer": "v", "Random": "r"}
    with pytest.raises(ValueError, match="Cannot determine note type"):
        fs._infer_note_type(note_fields)


def test_choice_validation_requires_choice_field_name():
    config = NoteType(
        name="BadChoice",
        is_choice=True,
        fields=[
            NoteField("MyQuestion", "XQ:", identifying=True),
            NoteField("MyAnswer", "XA:", identifying=True),
        ],
    )
    with pytest.raises(ValueError, match="containing the word 'choice' were found"):
        NoteType.validate_configs([config])


def test_choice_validation_requires_identifying_non_choice_field():
    config = NoteType(
        name="BadChoice",
        is_choice=True,
        fields=[
            NoteField("Question", "QX:", identifying=False),
            NoteField("Choice 1", "CX1:", identifying=True),
        ],
    )
    with pytest.raises(ValueError, match="no identifying non-choice field"):
        NoteType.validate_configs([config])


def test_choice_validation_requires_identifying_choice_field():
    config = NoteType(
        name="BadChoice",
        is_choice=True,
        fields=[
            NoteField("Question", "QX:", identifying=True),
            NoteField("Choice 1", "CX1:", identifying=False),
        ],
    )
    with pytest.raises(ValueError, match="no identifying choice field"):
        NoteType.validate_configs([config])


def test_choice_validation_passes_with_identifying_base_and_choice():
    config = NoteType(
        name="GoodChoice",
        is_choice=True,
        fields=[
            NoteField("Question", "XQ:", identifying=True),
            NoteField("Choice 1", "XC1:", identifying=True),
        ],
    )
    NoteType.validate_configs([config])
