"""Configuration and registry for AnkiOps note types."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
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
    identifying: bool = True


# The mandatory field for all AnkiOps note types
ANKIOPS_KEY_FIELD = Field("AnkiOps Key", None, identifying=False)

# Common fields shared across many note types
COMMON_FIELD_MAP = {
    "Extra": "E:",
    "More": "M:",
    "Source": "S:",
    "AI Notes": "AI:",
}


@dataclass
class NoteTypeConfig:
    """Configuration for a single note type."""

    name: str
    fields: list[Field]
    is_cloze: bool = False
    package_dir: Path | None = None  # Directory containing note_type.yaml and templates
    card_templates: list[dict[str, str]] | None = (
        None  # Explicit list of {Name, Front, Back}
    )

    @property
    def css(self) -> str:
        """Get CSS: Global root Styling.css + Local package Styling.css."""
        css_parts = []
        
        # 1. Global root Styling.css
        root_css = get_note_types_dir() / "Styling.css"
        if root_css.exists():
            css_parts.append(root_css.read_text(encoding="utf-8"))

        pkg_dir = self.package_dir
        if pkg_dir is not None:
            # 2. Parent-level Styling.css (important for tests using tempdirs)
            parent_css = pkg_dir.parent / "Styling.css"
            if parent_css.exists() and parent_css != root_css:
                css_parts.append(parent_css.read_text(encoding="utf-8"))

            # 3. Local package Styling.css
            local_css = pkg_dir / "Styling.css"
            if local_css.exists():
                css_parts.append(local_css.read_text(encoding="utf-8"))

        return "\n\n/* --- Added by Local Override --- */\n\n".join(css_parts)

    @property
    def identifying_prefixes(self) -> set[str]:
        """Return prefixes of identifying fields."""
        return {
            str(f.prefix) for f in self.fields if f.prefix is not None and f.identifying
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
        reserved_prefixes = set(COMMON_FIELD_MAP.values())
        # Add built-in prefixes from already registered built-ins
        for c in self._configs.values():
            if c.name.startswith("AnkiOps"):
                reserved_prefixes.update(c.identifying_prefixes)

        for field in config.fields:
            if field.prefix is not None and field.prefix in reserved_prefixes:
                if is_builtin:
                    continue

                raise ValueError(
                    f"Note type '{config.name}' uses reserved built-in/common "
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

    def get(self, name: str) -> NoteTypeConfig:
        """Get configuration for a note type."""
        if name not in self._configs:
            raise KeyError(f"Unknown note type: {name}")
        return self._configs[name]

    def eject_builtin_note_types(self, dst_dir: Path, force: bool = False) -> None:
        """Eject all built-in note type definitions to the filesystem."""

        src_root = resources.files("ankiops.card_templates")

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
        src_root = resources.files("ankiops.card_templates")
        with resources.as_file(src_root) as src_path:
            self.load(src_path)

    def load(self, directory: Path | None = None) -> None:
        """Scan a directory for Note Type definitions (folders with note_type.yaml)."""
        card_templates_dir = directory or get_note_types_dir()
        if not card_templates_dir.exists():
            return

        # 1. Look for note_types.yaml in the root (legacy/test support)
        root_config = card_templates_dir / "note_types.yaml"
        if root_config.exists():
            data = yaml.safe_load(root_config.read_text(encoding="utf-8"))
            for name, config_data in data.items():
                fields = []
                for f in config_data.get("fields", []):
                    if isinstance(f, str):
                        fields.append(Field(f, None))
                    else:
                        fields.append(
                            Field(
                                f["name"],
                                f.get("prefix"),
                                f.get("identifying", True),
                            )
                        )

                config = NoteTypeConfig(
                    name=name,
                    fields=fields,
                    is_cloze=config_data.get("cloze", False),
                    package_dir=card_templates_dir / name,
                )
                self.register(config)

        # 2. Look for per-note-type folders

        for subdir in card_templates_dir.iterdir():
            if not subdir.is_dir():
                continue

            config_path = subdir / "note_type.yaml"
            if not config_path.exists():
                continue

            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    info = yaml.safe_load(f) or {}

                # Use name from YAML if present, fallback to folder name
                name = info.get("name", subdir.name)

                fields = []
                for f in info.get("fields", []):
                    # Hardcoded default for common optional fields if not specified
                    is_common = f["name"] in COMMON_FIELD_MAP
                    identifying = f.get("identifying", not is_common)
                    fields.append(
                        Field(f["name"], f["prefix"], identifying=identifying)
                    )

                # Ensure AnkiOps Key is at the end
                if ANKIOPS_KEY_FIELD.name not in [f.name for f in fields]:
                    fields.append(ANKIOPS_KEY_FIELD)

                self.register(
                    NoteTypeConfig(
                        name=name,
                        fields=fields,
                        is_cloze=bool(info.get("cloze", False)),
                        package_dir=subdir,
                        card_templates=info.get("templates"),
                    )
                )
                logger.debug(f"Registered note type definition: {name}")
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
