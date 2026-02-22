"""Configuration and registry for AnkiOps note types."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import ClassVar

import yaml

from ankiops.config import get_note_types_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Field:
    """Definition of a field in a note type."""

    name: str
    prefix: str | None  # None is used for the AnkiOps Key field
    is_identifying: bool = True


# The mandatory field for all AnkiOps note types
ANKIOPS_KEY_FIELD = Field("AnkiOps Key", None, is_identifying=False)

@dataclass
class NoteTypeConfig:
    """Configuration for a single note type."""

    name: str
    fields: list[Field]
    is_cloze: bool = False
    is_choice: bool = False
    package_dir: Path | None = None  # Directory containing note_type.yaml and templates
    styling_paths: list[Path] = field(default_factory=list)
    card_templates: list[dict[str, str]] | None = (
        None  # Explicit list of {Name, Front, Back}
    )

    @property
    def css(self) -> str:
        """Get CSS from explicit styling_paths."""
        css_parts = []
        for path in self.styling_paths:
            if path.exists():
                css_parts.append(path.read_text(encoding="utf-8"))
            else:
                logger.warning(f"CSS file not found: {path}")

        return "\n\n/* --- Added by Local Override --- */\n\n".join(css_parts)

    @property
    def identifying_prefixes(self) -> set[str]:
        """Return prefixes of identifying fields."""
        return {
            str(field.prefix)
            for field in self.fields
            if field.prefix is not None and field.is_identifying
        }

    def get_field_by_prefix(self, prefix: str) -> Field | None:
        """Get field definition by its prefix."""
        for field in self.fields:
            if field.prefix == prefix:
                return field
        return None


class NoteTypeRegistry:
    """Registry for all supported note types (built-in and custom)."""

    # Names that cannot be used by other fields
    _RESERVED_NAMES: ClassVar[set[str]] = {ANKIOPS_KEY_FIELD.name}

    def __init__(self):
        self._configs: dict[str, NoteTypeConfig] = {}
        self.discover_builtins()

    def register(self, config: NoteTypeConfig) -> None:
        """Register a new note type configuration with strict validation."""
        if config.name in self._configs:
            # Allow re-registration (idempotent for reloading)
            pass

        self._validate_reservation(config)
        self._validate_global_consistency(config)
        self._validate_distinctness(config)
        self._validate_choice_fields(config)

        self._configs[config.name] = config

    def _validate_reservation(self, config: NoteTypeConfig) -> None:
        """Ensure fields do not accidentally start with reserved prefixes/names."""
        is_builtin = config.name.startswith("AnkiOps")

        # 1. Names
        for field in config.fields:
            if field.name in self._RESERVED_NAMES:
                # Built-ins must have the correct key field; custom types cannot use it at all
                if is_builtin:
                    if field.prefix != ANKIOPS_KEY_FIELD.prefix:
                        raise ValueError(
                            f"Built-in note type '{config.name}' has invalid "
                            f"prefix for reserved field '{field.name}'"
                        )
                else:
                    raise ValueError(
                        f"Note type '{config.name}' uses reserved "
                        f"field name '{field.name}'"
                    )

        # 2. Prefixes
        reserved_prefixes: set[str] = set()
        # Add built-in prefixes from already registered built-ins
        for registered_config in self._configs.values():
            if registered_config.name.startswith("AnkiOps"):
                reserved_prefixes.update(registered_config.identifying_prefixes)

        for field in config.fields:
            if field.prefix is not None and field.prefix in reserved_prefixes:
                if is_builtin:
                    continue

                raise ValueError(
                    f"Note type '{config.name}' uses reserved built-in "
                    f"prefix '{field.prefix}'"
                )

    def _validate_global_consistency(self, config: NoteTypeConfig) -> None:
        """Ensure a prefix always maps to the same field name globally."""
        global_map = self.prefix_to_field

        for field in config.fields:
            if field.prefix is not None and field.prefix in global_map:
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
                    f"to '{existing_name}' ({new_prefixes}), "
                    "which makes inference ambiguous."
                )

    def _validate_choice_fields(self, config: NoteTypeConfig) -> None:
        """Ensure choice note types have at least one field containing 'choice'."""
        if config.is_choice:
            has_choice_field = any(
                "choice" in field.name.lower() for field in config.fields
            )
            if not has_choice_field:
                raise ValueError(
                    f"Note type '{config.name}' is marked as 'is_choice', but no fields "
                    "containing the word 'choice' were found. Choice note types must "
                    "have at least one field with 'choice' in its name."
                )

    def get(self, name: str) -> NoteTypeConfig:
        """Get configuration for a note type."""
        if name not in self._configs:
            raise KeyError(f"Unknown note type: {name}")
        return self._configs[name]

    def eject_builtin_note_types(self, dst_dir: Path, force: bool = False) -> None:
        """Eject all built-in note type definitions to the filesystem."""

        src_root = resources.files("ankiops.note_types")

        # We use as_file to handle cases where the package is in a zip/egg
        with resources.as_file(src_root) as src_path:
            shutil.copytree(
                src_path,
                dst_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__init__.py", "__pycache__", "*.pyc"),
            )
        logger.debug(f"Ejected built-in note type definitions to {dst_dir}")

    def discover_builtins(self) -> None:
        """Load built-in note types from the package resources."""
        # Use path relative to this file to avoid importlib.resources ambiguity
        # when multiple versions of the package might be present.
        src_path = Path(__file__).parent / "note_types"
        if src_path.exists():
            self.load(src_path)

    def load(self, directory: Path | None = None) -> None:
        """Scan a directory for Note Type definitions (folders with note_type.yaml)."""
        note_types_dir = directory or get_note_types_dir()
        if not note_types_dir.exists():
            return

        # 1. Look for note_types.yaml in the root (legacy/test support)
        root_config = note_types_dir / "note_types.yaml"
        if root_config.exists():
            data = yaml.safe_load(root_config.read_text(encoding="utf-8"))
            for name, config_data in data.items():
                fields = []
                for field_def in config_data.get("fields", []):
                    if isinstance(field_def, str):
                        fields.append(Field(field_def, None))
                    else:
                        fields.append(
                            Field(
                                field_def["name"],
                                field_def.get("prefix"),
                                field_def.get("is_identifying", True),
                            )
                        )

                # Parse styling paths
                if "styling" not in config_data:
                    raise ValueError(
                        f"Note type '{name}' is missing mandatory 'styling' key"
                    )

                styling_input = config_data["styling"]
                if isinstance(styling_input, str):
                    styling_raw_paths = [styling_input]
                else:
                    styling_raw_paths = styling_input

                config = NoteTypeConfig(
                    name=name,
                    fields=fields,
                    is_cloze=config_data.get("is_cloze", False),
                    is_choice=config_data.get("is_choice", False),
                    package_dir=note_types_dir / name,
                    styling_paths=[
                        (note_types_dir / p).resolve() for p in styling_raw_paths
                    ],
                )
                self.register(config)

        # 2. Look for per-note-type folders

        for subdir in note_types_dir.iterdir():
            if not subdir.is_dir():
                continue

            config_path = subdir / "note_type.yaml"
            if not config_path.exists():
                continue

            try:
                with open(config_path, "r", encoding="utf-8") as file:
                    info = yaml.safe_load(file) or {}

                # Use name from YAML if present, fallback to folder name
                name = info.get("name", subdir.name)

                fields = []
                for field in info.get("fields", []):
                    is_identifying = field.get("is_identifying", True)
                    fields.append(
                        Field(field["name"], field["prefix"], is_identifying=is_identifying)
                    )

                # Ensure AnkiOps Key is at the end
                if ANKIOPS_KEY_FIELD.name not in [field.name for field in fields]:
                    fields.append(ANKIOPS_KEY_FIELD)

                # Parse styling paths
                if "styling" not in info:
                    raise ValueError(
                        f"Note type '{name}' is missing mandatory 'styling' key in note_type.yaml"
                    )

                styling_input = info["styling"]
                if isinstance(styling_input, str):
                    styling_raw_paths = [styling_input]
                else:
                    styling_raw_paths = styling_input

                self.register(
                    NoteTypeConfig(
                        name=name,
                        fields=fields,
                        is_cloze=bool(info.get("is_cloze", False)),
                        is_choice=bool(info.get("is_choice", False)),
                        package_dir=subdir,
                        card_templates=info.get("templates"),
                        styling_paths=[(subdir / p).resolve() for p in styling_raw_paths],
                    )
                )
                logger.debug(f"Registered note type definition: {name}")
            except ValueError as e:
                # Re-raise validation errors to make them strictly mandatory
                raise
            except Exception as e:
                logger.error(
                    f"Failed to load note type definition '{subdir.name}': {e}"
                )

    @property
    def supported_note_types(self) -> list[str]:
        """List of all registered note type names."""
        return list(self._configs.keys())

    @property
    def prefix_to_field(self) -> dict[str, str]:
        """Global mapping of prefix -> field name for all registered types."""
        mapping = {}
        for config in self._configs.values():
            for field in config.fields:
                if field.prefix is not None:
                    mapping[str(field.prefix)] = field.name
        return mapping

    @property
    def field_to_prefix(self) -> dict[str, str]:
        """Global mapping of field name -> prefix."""
        return {v: k for k, v in self.prefix_to_field.items()}


# Global registry instance
registry = NoteTypeRegistry()
