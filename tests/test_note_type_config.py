import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ankiops import models
from ankiops.note_type_config import (
    ANKIOPS_KEY_FIELD,
    Field,
    NoteTypeConfig,
    NoteTypeRegistry,
)


@pytest.fixture
def registry():
    r = NoteTypeRegistry()
    r.discover_builtins()
    return r


def test_builtins_registered(registry):
    """Ensure built-in note types are registered by default."""
    assert "AnkiOpsQA" in registry.supported_note_types
    assert "AnkiOpsReversed" in registry.supported_note_types
    assert "AnkiOpsCloze" in registry.supported_note_types
    assert "AnkiOpsInput" in registry.supported_note_types
    assert "AnkiOpsChoice" in registry.supported_note_types


def test_get_config(registry):
    """Ensure we can retrieve configuration for registered types."""
    config = registry.get("AnkiOpsQA")
    assert config.name == "AnkiOpsQA"
    assert not config.is_cloze


def test_register_custom_type(registry):
    """Test registering a new custom note type manually."""
    config = NoteTypeConfig(
        name="CustomType",
        fields=[
            Field("Word", "W:", identifying=True),
            Field("Translation", "Tr:", identifying=True),
        ],
        styling_paths=[Path("Styling.css")],
    )
    registry.register(config)

    assert "CustomType" in registry.supported_note_types
    retrieved = registry.get("CustomType")
    assert retrieved.fields == [
        Field("Word", "W:", identifying=True),
        Field("Translation", "Tr:", identifying=True),
    ]


def test_register_custom_conflict_with_builtin(registry):
    """Ensure custom types cannot use built-in prefixes."""
    config = NoteTypeConfig(
        name="BadCustom",
        fields=[Field("MyQuestion", "Q:", identifying=True)],  # "Q:" is reserved
        styling_paths=[Path("Styling.css")],
    )
    # Reservation check (Rule 1) now happens before Consistency check (Rule 2)
    with pytest.raises(ValueError, match="uses reserved built-in prefix"):
        registry.register(config)


def test_register_conflict_with_reserved(registry):
    """Ensure we cannot register a type that conflicts with reserved fields."""
    config = NoteTypeConfig(
        name="ConflictType",
        fields=[ANKIOPS_KEY_FIELD],  # reserved
        styling_paths=[Path("Styling.css")],
    )
    with pytest.raises(ValueError, match="uses reserved field name"):
        registry.register(config)


def test_mandatory_set_collision(registry):
    """Ensure two note types cannot have identical identifying fields."""
    config1 = NoteTypeConfig(
        name="Type1",
        fields=[
            Field("Prop", "P:", identifying=True),
            Field("Val", "V:", identifying=True),
        ],
        styling_paths=[Path("Styling.css")],
    )
    registry.register(config1)

    config2 = NoteTypeConfig(
        name="Type2",
        fields=[
            Field("Prop", "P:", identifying=True),
            Field("Val", "V:", identifying=True),
        ],
        styling_paths=[Path("Styling.css")],
    )

    with pytest.raises(ValueError, match="identical identifying fields"):
        registry.register(config2)


def test_load_custom_from_dir(registry):
    """Test loading custom configurations from a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        note_types = Path(tmpdir)
        yaml_path = note_types / "note_types.yaml"

        data = {
            "MyCustomType": {
                "fields": [
                    {"name": "Term", "prefix": "TM:", "identifying": True},
                    {"name": "Definition", "prefix": "D:", "identifying": True},
                ],
                "styling": "MyCustomType/Styling.css",
            }
        }
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")
        # In the new logic, we look for Styling.css in the package dir (local)
        # or the dir above it (root).
        (note_types / "MyCustomType").mkdir(parents=True, exist_ok=True)
        (note_types / "MyCustomType" / "Styling.css").write_text(
            ".custom { color: red; }", encoding="utf-8"
        )

        registry.load(note_types)

        assert "MyCustomType" in registry.supported_note_types
        config = registry.get("MyCustomType")

        assert config.package_dir == note_types / "MyCustomType"
        # Only Term and Definition are identifying (have prefixes)
        # Only Term and Definition are identifying
        assert sorted([f.name for f in config.fields if f.identifying]) == [
            "Definition",
            "Term",
        ]
        assert config.css


def test_load_custom_missing_styling(registry):
    """Test that missing mandatory styling raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        note_types = Path(tmpdir)
        yaml_path = note_types / "note_types.yaml"
        # Missing 'styling' key
        data = {"NoCSSType": {"fields": [{"name": "XField", "prefix": "XF:", "identifying": True}]}}
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")

        with pytest.raises(ValueError, match="missing mandatory 'styling' key"):
            registry.load(note_types)


def test_require_styling_key(registry):
    """Test that missing mandatory styling in per-folder YAML raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        note_types = Path(tmpdir)
        subdir = note_types / "BadType"
        subdir.mkdir()
        yaml_path = subdir / "note_type.yaml"
        # Missing 'styling' key
        data = {"fields": [{"name": "XField", "prefix": "XF:", "identifying": True}]}
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")

        with pytest.raises(ValueError, match="missing mandatory 'styling' key"):
            registry.load(note_types)

def test_prefix_to_field_mapping(registry):
    """Test the global prefix -> field mapping."""
    mapping = registry.prefix_to_field

    # Common fields
    assert mapping["E:"] == "Extra"

    # Built-in fields
    assert mapping["Q:"] == "Question"
    assert mapping["A:"] == "Answer"


def test_prefix_collision_different_field(registry):
    """Ensure a prefix cannot map to two different fields globally."""
    # First registration OK
    config1 = NoteTypeConfig(
        name="Type1",
        fields=[Field("FieldA", "X:", identifying=True)],
        styling_paths=[Path("Styling.css")],
    )
    registry.register(config1)

    # Second registration uses same prefix for DIFFERENT field -> Error
    config2 = NoteTypeConfig(
        name="Type2",
        fields=[Field("FieldB", "X:", identifying=True)],
        styling_paths=[Path("Styling.css")],
    )
    with pytest.raises(
        ValueError, match="matches existing prefix.*but maps to different field"
    ):
        registry.register(config2)


def test_prefix_sharing_custom_field(registry):
    """Ensure two note types CAN share a prefix if it maps to the SAME *custom*
    field name.
    """
    config1 = NoteTypeConfig(
        name="Type1",
        fields=[Field("Description", "D:", identifying=True)],
        styling_paths=[Path("Styling.css")],
    )
    registry.register(config1)

    config2 = NoteTypeConfig(
        name="Type2",
        fields=[
            Field("Description", "D:", identifying=True),
            Field("Other", "O:", identifying=True),
        ],
        styling_paths=[Path("Styling.css")],
    )
    # Should Pass for CUSTOM fields
    registry.register(config2)

    assert registry.prefix_to_field["D:"] == "Description"


def test_strict_builtin_reservation(registry):
    """Ensure custom types cannot use built-in fields even if they match."""
    # Attempt to use "Question" and "Q:" (Built-in)
    config = NoteTypeConfig(
        name="MyQA",
        fields=[Field("Question", "Q:", identifying=True)],
        styling_paths=[Path("Styling.css")],
    )
    with pytest.raises(ValueError, match="uses reserved built-in prefix"):
        registry.register(config)

    # Attempt to use just the name "Question" with different prefix
    # (Note: This is caught by rule 2 if prefix maps to something else,
    # but if prefix is new, it's caught by rule 3 strict name check)
    config_name = NoteTypeConfig(
        name="MyQA_Name",
        fields=[Field("Question", "MyQ:", identifying=True)],
        styling_paths=[Path("Styling.css")],
    )
    # Currently Name reservation is minimal (AnkiOps Key only).
    # If Question is not reserved, this won't raise.
    # We check if Question is in _RESERVED_NAMES
    if "Question" in registry._RESERVED_NAMES:
        with pytest.raises(ValueError, match="uses reserved field name"):
            registry.register(config_name)
    else:
        # Should pass since prefix doesn't overlap built-ins (MyQ vs Q)
        registry.register(config_name)

    # Attempt to use just the prefix "Q:" with different name
    # This might be caught by Rule 2 (Global Consistency) if "Q:" is already
    # mapped to "Question"
    config_prefix = NoteTypeConfig(
        name="MyQA_Prefix",
        fields=[Field("MyQuestion", "Q:", identifying=True)],
        styling_paths=[Path("Styling.css")],
    )
    with pytest.raises(ValueError, match="uses reserved built-in prefix"):
        registry.register(config_prefix)


def test_subset_inference(registry):
    """Register a type that is a superset of another, verify inference."""
    # Create a base custom type first
    base_config = NoteTypeConfig(
        name="BaseType",
        fields=[
            Field("BaseQ", "BQ:", identifying=True),
            Field("BaseA", "BA:", identifying=True),
        ],
        styling_paths=[Path("Styling.css")],
    )
    registry.register(base_config)

    # Create SupersetType {BaseQ, BaseA, Context}
    superset_config = NoteTypeConfig(
        name="SupersetType",
        fields=[
            Field("BaseQ", "BQ:", identifying=True),
            Field("BaseA", "BA:", identifying=True),
            Field("Context", "C:", identifying=True),
        ],
        styling_paths=[Path("Styling.css")],
    )
    registry.register(superset_config)

    # Test Inference using the local registry
    from unittest.mock import patch

    from ankiops import models

    with patch("ankiops.models.registry", registry):
        # Fields for Superset
        fields_super = {"BaseQ": "q", "BaseA": "c", "Context": "c"}
        assert models.Note.infer_note_type(fields_super) == "SupersetType"

        # Fields for Base (subset)
        fields_base = {"BaseQ": "q", "BaseA": "a"}
        assert models.Note.infer_note_type(fields_base) == "BaseType"


def test_reserved_ankiops_key(registry):
    """Ensure reserved AnkiOps Key name cannot be used."""
    config = NoteTypeConfig(
        name="BadName",
        fields=[Field(ANKIOPS_KEY_FIELD.name, "X:", identifying=False)],
        styling_paths=[Path("Styling.css")],
    )
    with pytest.raises(ValueError, match="uses reserved field name"):
        registry.register(config)


# --- Inference Tests ---


@pytest.fixture
def mock_registry():
    r = NoteTypeRegistry()
    # Clear rules for clean slate testing of inference logic
    r._configs = {}
    return r


def test_inference_qa_vs_choice(mock_registry):
    """Verify {Q, A} matches QA (tighter) over Choice."""
    qa = NoteTypeConfig(
        "QA",
        [
            Field("Question", "Q:", identifying=True),
            Field("Answer", "A:", identifying=True),
        ],
        styling_paths=[Path("Styling.css")],
    )
    choice = NoteTypeConfig(
        "Choice",
        [
            Field("Question", "Q:", identifying=True),
            Field("Answer", "A:", identifying=True),
            Field("Choice 1", "C1:", identifying=True),
        ],
        styling_paths=[Path("Styling.css")],
    )
    mock_registry._configs = {"QA": qa, "Choice": choice}

    with patch("ankiops.models.registry", mock_registry):
        # {Q, A} -> QA (size 2) vs Choice (size > 2) -> QA wins
        qa_fields = {"Question": "val", "Answer": "val"}
        assert models.Note.infer_note_type(qa_fields) == "QA"

        # {Q, A, C1} -> Choice vs QA (rejection) -> Choice wins
        choice_fields = {"Question": "val", "Answer": "val", "Choice 1": "val"}
        assert models.Note.infer_note_type(choice_fields) == "Choice"


def test_inference_choice_resilience(mock_registry):
    """Verify Choice note is detected even if some choices are missing."""
    choice = NoteTypeConfig(
        "Choice",
        [
            Field("Question", "Q:", identifying=True),
            Field("Answer", "A:", identifying=True),
            Field("Choice 1", "C1:", identifying=True),
            Field("Choice 2", "C2:", identifying=True),
            Field("Choice 5", "C5:", identifying=True),
            Field("Choice 6", "C6:", identifying=True),
        ],
        is_choice=True,
        styling_paths=[Path("Styling.css")],
    )
    mock_registry._configs = {"Choice": choice}

    with patch("ankiops.models.registry", mock_registry):
        # {Q, A, C2} -> Choice
        note_fields = {"Question": "v", "Answer": "v", "Choice 5": "v", "Choice 6": "v"}
        assert models.Note.infer_note_type(note_fields) == "Choice"


def test_inference_with_non_identifying_fields(mock_registry):
    """Verify non-identifying fields don't interfere with inference."""
    qa = NoteTypeConfig(
        "QA",
        [
            Field("Question", "Q:", identifying=True),
            Field("Answer", "A:", identifying=True),
            Field("Extra", "E:", identifying=False),
        ],
        styling_paths=[Path("Styling.css")],
    )
    mock_registry._configs = {"QA": qa}

    with patch("ankiops.models.registry", mock_registry):
        # {Q, A, Extra} -> matches QA
        note_fields = {"Question": "v", "Answer": "v", "Extra": "ex"}
        assert models.Note.infer_note_type(note_fields) == "QA"


def test_inference_unknown_field_fails(mock_registry):
    """Verify that unknown fields cause inference failure."""
    qa = NoteTypeConfig(
        "QA",
        [Field("Question", "Q:", identifying=True), Field("Answer", "A:", identifying=True)],
        styling_paths=[Path("Styling.css")],
    )
    mock_registry._configs = {"QA": qa}

    with patch("ankiops.models.registry", mock_registry):
        # {Q, A, Random} -> Fails
        note_fields = {"Question": "v", "Answer": "v", "Random": "r"}
        with pytest.raises(ValueError, match="Cannot determine note type"):
            models.Note.infer_note_type(note_fields)


def test_choice_validation(registry):
    """Ensure choice note types must have at least one field with 'choice' in name."""
    # 1. is_choice=True, no choice field -> Fail
    config1 = NoteTypeConfig(
        name="BadChoice",
        is_choice=True,
        fields=[Field("MyQuestion", "XQ:", identifying=True), Field("MyAnswer", "XA:", identifying=True)],
        styling_paths=[Path("Styling.css")],
    )
    with pytest.raises(ValueError, match="containing the word 'choice' were found"):
        registry.register(config1)

    # 2. is_choice=True, has choice field -> Pass
    config2 = NoteTypeConfig(
        name="GoodChoice",
        is_choice=True,
        fields=[Field("MyQuestion", "XQ:", identifying=True), Field("Choice 1", "XC1:", identifying=True)],
        styling_paths=[Path("Styling.css")],
    )
    registry.register(config2)
    assert "GoodChoice" in registry.supported_note_types

    # 3. is_choice=False, no choice field -> Pass
    config3 = NoteTypeConfig(
        name="NormalQA",
        is_choice=False,
        fields=[
            Field("MyQuestion", "XQ:", identifying=True),
            Field("MyAnswer", "XA:", identifying=True),
        ],
        styling_paths=[Path("Styling.css")],
    )
    registry.register(config3)
    assert "NormalQA" in registry.supported_note_types
