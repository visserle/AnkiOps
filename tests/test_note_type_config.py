
import tempfile
from pathlib import Path

import pytest
import yaml
from unittest.mock import patch

from ankiops import models
from ankiops.note_type_config import (
    NoteTypeRegistry, 
    NoteTypeConfig, 
    Field, 
    IDENTIFYING_FIELDS, 
    COMMON_FIELDS
)


@pytest.fixture
def registry():
    return NoteTypeRegistry()


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
    assert not config.custom
    assert not config.is_cloze


def test_register_custom_type(registry):
    """Test registering a new custom note type manually."""
    config = NoteTypeConfig(
        name="CustomType",
        fields=[Field("Word", "W:"), Field("Translation", "Tr:")],
        custom=True,
    )
    registry.register(config)
    
    assert "CustomType" in registry.supported_note_types
    retrieved = registry.get("CustomType")
    assert retrieved.custom
    assert retrieved.fields == [Field("Word", "W:"), Field("Translation", "Tr:")]


def test_register_custom_conflict_with_builtin(registry):
    """Ensure custom types cannot use built-in prefixes."""
    config = NoteTypeConfig(
        name="BadCustom",
        fields=[Field("MyQuestion", "Q:")],  # "Q:" is reserved
        custom=True,
    )
    # Reservation check (Rule 1) now happens before Consistency check (Rule 2)
    with pytest.raises(ValueError, match="uses reserved built-in/common prefix"):
        registry.register(config)


def test_register_conflict_with_common(registry):
    """Ensure we cannot register a type that conflicts with common fields."""
    config = NoteTypeConfig(
        name="ConflictType",
        fields=[COMMON_FIELDS[0]],  # "Extra" / "E:"
    )
    with pytest.raises(ValueError, match="uses reserved common prefix"):
        registry.register(config)

    config_name = NoteTypeConfig(
        name="ConflictName",
        fields=[Field(COMMON_FIELDS[0].name, "X:")],  # "Extra" name, different prefix
    )
    with pytest.raises(ValueError, match="uses reserved common field name"):
        registry.register(config_name)


def test_register_conflict_with_existing(registry):
    """Ensure we cannot register a type that has identical identifying fields to an existing one."""
    config = NoteTypeConfig(
        name="CopyCat",
        fields=IDENTIFYING_FIELDS["AnkiOpsQA"],
    )
    with pytest.raises(ValueError, match="makes inference ambiguous"):
        registry.register(config)

    # Subset should be allowed
    subset_config = NoteTypeConfig(
        name="SubsetType",
        fields=[IDENTIFYING_FIELDS["AnkiOpsQA"][0]], # Just Question
    )
    registry.register(subset_config)
    assert "SubsetType" in registry.supported_note_types


def test_load_custom_from_dir(registry):
    """Test loading custom configurations from a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        card_templates = Path(tmpdir)
        yaml_path = card_templates / "note_types.yaml"
        
        data = {
            "MyCustomType": {
                "fields": [
                    {"name": "Term", "prefix": "TM:"},
                    {"name": "Definition", "prefix": "D:"}
                ],
                "reversed": True
            }
        }
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")
        
        registry.load_custom(card_templates)
        
        assert "MyCustomType" in registry.supported_note_types
        config = registry.get("MyCustomType")

        assert config.custom
        assert config.templates_dir == card_templates
        assert config.fields == [Field("Term", "TM:"), Field("Definition", "D:")]


def test_load_custom_css(registry):
    """Test detecting custom CSS file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        card_templates = Path(tmpdir)
        (card_templates / "note_types.yaml").touch() # Empty config is fine
        (card_templates / "Styling.css").touch()
        
        registry.load_custom(card_templates)
        
        assert registry.custom_css_path == card_templates / "Styling.css"


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
        fields=[Field("FieldA", "X:")],
        custom=True,
    )
    registry.register(config1)

    # Second registration uses same prefix for DIFFERENT field -> Error
    config2 = NoteTypeConfig(
        name="Type2",
        fields=[Field("FieldB", "X:")],
        custom=True,
    )
    with pytest.raises(ValueError, match="matches existing prefix.*but maps to different field"):
        registry.register(config2)


def test_prefix_sharing_custom_field(registry):
    """Ensure two note types CAN share a prefix if it maps to the SAME *custom* field name."""
    config1 = NoteTypeConfig(
        name="Type1",
        fields=[Field("Description", "D:")],
        custom=True,
    )
    registry.register(config1)

    config2 = NoteTypeConfig(
        name="Type2",
        fields=[Field("Description", "D:"), Field("Other", "O:")],
        custom=True,
    )
    # Should Pass for CUSTOM fields
    registry.register(config2)
    
    assert registry.prefix_to_field["D:"] == "Description"


def test_strict_builtin_reservation(registry):
    """Ensure custom types cannot use built-in fields even if they match."""
    # Attempt to use "Question" and "Q:" (Built-in)
    config = NoteTypeConfig(
        name="MyQA",
        fields=[Field("Question", "Q:")], 
        custom=True,
    )
    with pytest.raises(ValueError, match="uses reserved built-in/common"):
        registry.register(config)

    # Attempt to use just the name "Question" with different prefix
    # (Note: This is caught by rule 2 if prefix maps to something else, 
    # but if prefix is new, it's caught by rule 3 strict name check)
    config_name = NoteTypeConfig(
        name="MyQA_Name",
        fields=[Field("Question", "MyQ:")],
        custom=True,
    )
    with pytest.raises(ValueError, match="uses reserved built-in/common field name"):
        registry.register(config_name)
    
    # Attempt to use just the prefix "Q:" with different name
    # This might be caught by Rule 2 (Global Consistency) if "Q:" is already mapped to "Question"
    config_prefix = NoteTypeConfig(
        name="MyQA_Prefix",
        fields=[Field("MyQuestion", "Q:")],
        custom=True,
    )
    with pytest.raises(ValueError, match="matches existing prefix|uses reserved built-in/common prefix"):
        registry.register(config_prefix)


def test_mandatory_set_collision(registry):
    """Ensure two note types cannot have identical identifying fields."""
    config1 = NoteTypeConfig(
        name="Type1",
        fields=[Field("Prop", "P:"), Field("Val", "V:")],
        custom=True,
    )
    registry.register(config1)
    
    config2 = NoteTypeConfig(
        name="Type2",
        fields=[Field("Prop", "P:"), Field("Val", "V:")],
        custom=True,
    )
    
    with pytest.raises(ValueError, match="identical identifying fields"):
        registry.register(config2)


def test_subset_inference(registry):
    """Register a type that is a superset of another, verify inference."""
    # Create a base custom type first
    base_config = NoteTypeConfig(
        name="BaseType",
        fields=[Field("BaseQ", "BQ:"), Field("BaseA", "BA:")],
        custom=True,
    )
    registry.register(base_config)
    
    # Create SupersetType {BaseQ, BaseA, Context}
    superset_config = NoteTypeConfig(
        name="SupersetType",
        fields=[Field("BaseQ", "BQ:"), Field("BaseA", "BA:"), Field("Context", "C:")],
        custom=True,
    )
    registry.register(superset_config)
    
    # Test Inference using the local registry
    from ankiops import models
    from unittest.mock import patch
    
    with patch("ankiops.models.registry", registry):
        # Fields for Superset
        fields_super = {"BaseQ": "q", "BaseA": "a", "Context": "c"}
        assert models.Note.infer_note_type(fields_super) == "SupersetType"
        
        # Fields for Base (subset)
        fields_base = {"BaseQ": "q", "BaseA": "a"}
        assert models.Note.infer_note_type(fields_base) == "BaseType"


def test_reserved_common_names(registry):
    """Ensure reserved common field names cannot be used."""
    config = NoteTypeConfig(
        name="BadName",
        fields=[Field(COMMON_FIELDS[0].name, "X:")], # Name collision with Common Field (Extra)
        custom=True,
    )
    with pytest.raises(ValueError, match="uses reserved built-in/common field name"):
        registry.register(config)


def test_reserved_common_prefixes(registry):
    """Ensure reserved common field prefixes cannot be used."""
    config = NoteTypeConfig(
        name="BadPrefix",
        fields=[Field("Xfield", COMMON_FIELDS[0].prefix)], # Prefix collision with Common Field (E:)
        custom=True,
    )
    with pytest.raises(ValueError, match="uses reserved built-in/common prefix"):
        registry.register(config)


def test_reserved_builtin_prefixes_for_custom(registry):
    """Ensure custom types cannot highjack built-in prefixes for different fields."""
    # AnkiOpsQA uses Q: and A:
    config = NoteTypeConfig(
        name="Hijack",
        fields=[Field("MyField", "Q:")], # Q: is taken by Question
        custom=True,
    )
    # Rule 1 (Reservation) catches this before Rule 2 (Consistency) because Q: is reserved
    with pytest.raises(ValueError, match="uses reserved built-in/common prefix"):
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
    qa = NoteTypeConfig("QA", IDENTIFYING_FIELDS["AnkiOpsQA"])
    choice = NoteTypeConfig("Choice", IDENTIFYING_FIELDS["AnkiOpsChoice"])
    mock_registry._configs = {"QA": qa, "Choice": choice}
    
    with patch("ankiops.models.registry", mock_registry):
        # {Q, A} -> QA (size 2) vs Choice (size > 2) -> QA wins
        qa_fields = {f.name: "val" for f in IDENTIFYING_FIELDS["AnkiOpsQA"]}
        assert models.Note.infer_note_type(qa_fields) == "QA"
        
        # {Q, A, C1} -> Choice vs QA (rejection) -> Choice wins
        # Construct fields: QA fields + Choice 1
        choice_fields = qa_fields.copy()
        choice_1 = next(f for f in IDENTIFYING_FIELDS["AnkiOpsChoice"] if f.name == "Choice 1")
        choice_fields[choice_1.name] = "val"
        assert models.Note.infer_note_type(choice_fields) == "Choice"

def test_inference_choice_resilience(mock_registry):
    """Verify Choice note is detected even if some choices are missing."""
    choice = NoteTypeConfig("Choice", IDENTIFYING_FIELDS["AnkiOpsChoice"])
    mock_registry._configs = {"Choice": choice}
    
    with patch("ankiops.models.registry", mock_registry):
        # {Q, A, C2} -> Choice
        qa_fields = {f.name: "val" for f in IDENTIFYING_FIELDS["AnkiOpsQA"]}
        choice_2 = next(f for f in IDENTIFYING_FIELDS["AnkiOpsChoice"] if f.name == "Choice 2")
        note_fields = qa_fields | {choice_2.name: "val"}
        
        assert models.Note.infer_note_type(note_fields) == "Choice"

def test_inference_with_common_fields(mock_registry):
    """Verify common fields don't interfere with inference."""
    qa = NoteTypeConfig("QA", IDENTIFYING_FIELDS["AnkiOpsQA"])
    mock_registry._configs = {"QA": qa}
    
    with patch("ankiops.models.registry", mock_registry):
        # {Q, A, Extra} -> matches QA
        qa_fields = {f.name: "val" for f in IDENTIFYING_FIELDS["AnkiOpsQA"]}
        note_fields = qa_fields | {COMMON_FIELDS[0].name: "ex"} # Extra
        assert models.Note.infer_note_type(note_fields) == "QA"

def test_inference_unknown_field_fails(mock_registry):
    """Verify that unknown fields cause inference failure."""
    qa = NoteTypeConfig("QA", IDENTIFYING_FIELDS["AnkiOpsQA"])
    mock_registry._configs = {"QA": qa}
    
    with patch("ankiops.models.registry", mock_registry):
        # {Q, A, Random} -> Fails
        qa_fields = {f.name: "val" for f in IDENTIFYING_FIELDS["AnkiOpsQA"]}
        note_fields = qa_fields | {"Random": "r"}
        with pytest.raises(ValueError, match="Cannot determine note type"):
            models.Note.infer_note_type(note_fields)
