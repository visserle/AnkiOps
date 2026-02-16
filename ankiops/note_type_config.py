"""Configuration and registry for AnkiOps note types."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Field:
    """Definition of a field in a note type."""
    name: str
    prefix: str


# Common fields available to all note types
COMMON_FIELDS = [
    Field("Extra", "E:"),
    Field("More", "M:"),
    Field("Source", "S:"),
    Field("AI Notes", "AI:"),
]

# Identifying fields for built-in note types
IDENTIFYING_FIELDS = {
    "AnkiOpsQA": [
        Field("Question", "Q:"),
        Field("Answer", "A:"),
    ],
    "AnkiOpsReversed": [
        Field("Front", "F:"),
        Field("Back", "B:"),
    ],
    "AnkiOpsCloze": [
        Field("Text", "T:"),
    ],
    "AnkiOpsInput": [
        Field("Question", "Q:"),
        Field("Input", "I:"),
    ],
    "AnkiOpsChoice": [
        Field("Question", "Q:"),
        *[Field(f"Choice {i}", f"C{i}:") for i in range(1, 9)],
        Field("Answer", "A:"),
    ],
}


@dataclass
class NoteTypeConfig:
    """Configuration for a single note type."""

    name: str
    fields: list[Field]
    is_cloze: bool = False
    is_reversed: bool = False
    has_choices: bool = False
    custom: bool = False
    templates_dir: Path | None = None

    @property
    def identifying_fields(self) -> list[Field]:
        """Return identifying fields (excluding common fields)."""
        return self.fields

    @property
    def identifying_field_names(self) -> list[str]:
        """Return names of identifying fields."""
        return [f.name for f in self.fields]

    @property
    def identifying_prefixes(self) -> set[str]:
        """Return prefixes of identifying fields."""
        return {f.prefix for f in self.fields}

    def get_field_by_prefix(self, prefix: str) -> Field | None:
        """Get field definition by its prefix."""
        for field in self.fields:
            if field.prefix == prefix:
                return field
        return None


class NoteTypeRegistry:
    """Registry for all supported note types (built-in and custom)."""
    
    # Pre-computed sets for validation
    _COMMON_NAMES: ClassVar[set[str]] = {f.name for f in COMMON_FIELDS}
    _COMMON_PREFIXES: ClassVar[set[str]] = {f.prefix for f in COMMON_FIELDS}
    
    # Built-in identifying fields are also reserved to prevent ambiguity
    _BUILTIN_NAMES: ClassVar[set[str]] = set()
    _BUILTIN_PREFIXES: ClassVar[set[str]] = set()

    def __init__(self):
        self._configs: dict[str, NoteTypeConfig] = {}
        self.custom_css_path: Path | None = None
        self._initialize_builtins()

    def _initialize_builtins(self):
        """Register built-in note types."""
        # Config for built-ins
        builtin_configs = [
            NoteTypeConfig("AnkiOpsQA", IDENTIFYING_FIELDS["AnkiOpsQA"]),
            NoteTypeConfig("AnkiOpsReversed", IDENTIFYING_FIELDS["AnkiOpsReversed"], is_reversed=True),
            NoteTypeConfig("AnkiOpsCloze", IDENTIFYING_FIELDS["AnkiOpsCloze"], is_cloze=True),
            NoteTypeConfig("AnkiOpsInput", IDENTIFYING_FIELDS["AnkiOpsInput"]),
            NoteTypeConfig("AnkiOpsChoice", IDENTIFYING_FIELDS["AnkiOpsChoice"], has_choices=True),
        ]

        # Populate reserved sets from built-ins
        for config in builtin_configs:
            for field in config.fields:
                self._BUILTIN_NAMES.add(field.name)
                self._BUILTIN_PREFIXES.add(field.prefix)

        # Register them
        for config in builtin_configs:
            self.register(config)

    def register(self, config: NoteTypeConfig) -> None:
        """Register a new note type configuration with strict validation."""
        if config.name in self._configs:
            # Allow re-registration (idempotent for reloading)
            pass

        self._validate_reservation(config)
        self._validate_global_consistency(config)
        self._validate_distinctness(config)

        self._configs[config.name] = config

    def _validate_reservation(self, config: NoteTypeConfig) -> None:
        """Ensure fields do not accidentally start with reserved prefixes/names."""
        # 1. Determine reserved sets
        if config.custom:
             # Custom types: Cannot match Common OR Built-in fields
            reserved_prefixes = NoteTypeRegistry._COMMON_PREFIXES | NoteTypeRegistry._BUILTIN_PREFIXES
            reserved_names = NoteTypeRegistry._COMMON_NAMES | NoteTypeRegistry._BUILTIN_NAMES
            error_msg = "reserved built-in/common"
        else:
             # Built-in types: Cannot match Common fields
            reserved_prefixes = NoteTypeRegistry._COMMON_PREFIXES
            reserved_names = NoteTypeRegistry._COMMON_NAMES
            error_msg = "reserved common"

        # 2. Check each field
        for field in config.fields:
            if field.prefix in reserved_prefixes:
                # Special case: allow if it's strictly the SAME field (same name & prefix)?
                # Actually, common fields are implicitly added later, so the 'identifying'
                # fields config shouldn't contain them at all.
                raise ValueError(
                    f"Note type '{config.name}' uses {error_msg} prefix '{field.prefix}'"
                )
            if field.name in reserved_names:
                raise ValueError(
                    f"Note type '{config.name}' uses {error_msg} field name '{field.name}'"
                )

    def _validate_global_consistency(self, config: NoteTypeConfig) -> None:
        """Ensure a prefix always maps to the same field name globally."""
        global_map = self.prefix_to_field
        
        for field in config.fields:
            if field.prefix in global_map:
                existing_name = global_map[field.prefix]
                if existing_name != field.name:
                    raise ValueError(
                        f"Prefix '{field.prefix}' matches existing prefix for "
                        f"'{existing_name}', but maps to different field "
                        f"'{field.name}' in '{config.name}'"
                    )

    def _validate_distinctness(self, config: NoteTypeConfig) -> None:
        """Ensure the set of identifying fields is unique to avoid ambiguity."""
        new_prefixes = config.identifying_prefixes
        
        for existing_name, existing_config in self._configs.items():
            if existing_name == config.name:
                continue
                
            if new_prefixes == existing_config.identifying_prefixes:
                 raise ValueError(
                    f"Note type '{config.name}' has identical identifying fields "
                    f"to '{existing_name}' ({new_prefixes}), which makes inference ambiguous."
                )

    def get(self, name: str) -> NoteTypeConfig:
        """Get configuration for a note type."""
        if name not in self._configs:
            raise KeyError(f"Unknown note type: {name}")
        return self._configs[name]

    def load_custom(self, card_templates_dir: Path) -> None:
        """Load custom note types from a directory."""
        if not card_templates_dir.exists():
            return

        config_path = card_templates_dir / "note_types.yaml"
        if not config_path.exists():
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to parse {config_path}: {e}")
            return

        for name, info in data.items():
            try:
                fields = [
                    Field(f["name"], f["prefix"])
                    for f in info.get("fields", [])
                ]
                
                self.register(
                    NoteTypeConfig(
                        name=name,
                        fields=fields,
                        is_cloze=info.get("cloze", False),
                        is_reversed=info.get("reversed", False),
                        has_choices=info.get("choices", False),
                        custom=True,
                        templates_dir=card_templates_dir,
                    )
                )
                logger.debug(f"Registered custom note type: {name}")
            except Exception as e:
                logger.error(f"Failed to register custom type '{name}': {e}")

        # Check for custom styling
        css_path = card_templates_dir / "Styling.css"
        if css_path.exists():
            self.custom_css_path = css_path
            logger.debug(f"Found custom styling at {css_path}")

    @property
    def supported_note_types(self) -> list[str]:
        """List of all registered note type names."""
        return list(self._configs.keys())

    @property
    def prefix_to_field(self) -> dict[str, str]:
        """Global mapping of prefix -> field name for all registered types."""
        mapping = {}
        for field in COMMON_FIELDS:
            mapping[field.prefix] = field.name
            
        for config in self._configs.values():
            for field in config.fields:
                mapping[field.prefix] = field.name
        return mapping

    @property
    def field_to_prefix(self) -> dict[str, str]:
        """Global mapping of field name -> prefix."""
        return {v: k for k, v in self.prefix_to_field.items()}

    @property
    def note_config(self) -> dict[str, list[tuple[str, str]]]:
        """Legacy-style dictionary of note type -> all fields (including common).
        
        Returns:
            Dict mapping note type name to list of (field_name, prefix) tuples.
        """
        result = {}
        common_tuples = [(f.name, f.prefix) for f in COMMON_FIELDS]
        for name, config in self._configs.items():
            field_tuples = [(f.name, f.prefix) for f in config.fields]
            result[name] = field_tuples + common_tuples
        return result

    @property
    def identifying_fields(self) -> dict[str, list[tuple[str, str]]]:
        """Legacy-style dictionary of note type -> identifying fields.
        
        Returns:
            Dict mapping note type name to list of (field_name, prefix) tuples.
        """
        result = {}
        for name, config in self._configs.items():
             result[name] = [(f.name, f.prefix) for f in config.fields]
        return result


# Global registry instance
registry = NoteTypeRegistry()
