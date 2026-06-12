"""Note type definitions, files, and Anki synchronization."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from ankiops.anki import Anki
    from ankiops.sync.state import SyncState

logger = logging.getLogger(__name__)
_NoteTypeCacheValue = tuple[tuple[tuple[str, int, int], ...], list["NoteType"]]
_NOTE_TYPE_CACHE: dict[Path, _NoteTypeCacheValue] = {}


@dataclass(frozen=True)
class NoteField:
    """Definition of a field in a note type."""

    name: str
    label: str | None
    identifying: bool


ANKIOPS_KEY_FIELD = NoteField("AnkiOps Key", None, identifying=False)


@dataclass
class NoteType:
    """Pure data configuration for a single note type."""

    name: str
    fields: list[NoteField]
    css: str = ""
    is_cloze: bool = False
    is_choice: bool = False
    templates: list[dict[str, str]] = field(default_factory=list)

    @property
    def identifying_labels(self) -> set[str]:
        """Return labels of identifying fields."""
        return {
            str(field_config.label)
            for field_config in self.fields
            if field_config.label is not None and field_config.identifying
        }

    @classmethod
    def validate_configs(cls, configs: list["NoteType"]) -> None:
        """Validate note type configs using global set invariants."""
        reserved_names = {ANKIOPS_KEY_FIELD.name}
        label_to_field_name: dict[str, str] = {}
        label_to_identifying: dict[str, bool] = {}
        identifying_signature_to_type: dict[frozenset[str], str] = {}

        for config in configs:
            seen_names: set[str] = set()
            seen_labels: set[str] = set()
            identifying_choice_labels: set[str] = set()
            identifying_non_choice_labels: set[str] = set()
            has_choice_field = False

            for field_config in config.fields:
                if field_config.name in seen_names:
                    raise ValueError(
                        f"Note type '{config.name}' has duplicate field name "
                        f"'{field_config.name}'. "
                        "Field names must be unique within a note type."
                    )
                seen_names.add(field_config.name)

                if field_config.label is not None:
                    if field_config.label in seen_labels:
                        raise ValueError(
                            f"Note type '{config.name}' has duplicate field label "
                            f"'{field_config.label}'. "
                            "Field labels must be unique within a "
                            "note type."
                        )
                    seen_labels.add(field_config.label)

                if field_config.name in reserved_names:
                    if field_config.label != ANKIOPS_KEY_FIELD.label:
                        raise ValueError(
                            f"Note type '{config.name}' uses reserved "
                            f"field name '{field_config.name}' with invalid label. "
                            "Expected no label."
                        )
                    if field_config.identifying != ANKIOPS_KEY_FIELD.identifying:
                        raise ValueError(
                            f"Note type '{config.name}' uses reserved "
                            f"field name '{field_config.name}' as identifying, "
                            "which is not allowed."
                        )

                if field_config.label is not None:
                    existing_name = label_to_field_name.get(field_config.label)
                    if existing_name is not None and existing_name != field_config.name:
                        raise ValueError(
                            f"Label '{field_config.label}' maps to both "
                            f"'{existing_name}' and '{field_config.name}'."
                        )
                    label_to_field_name[field_config.label] = field_config.name

                    existing_identifying = label_to_identifying.get(field_config.label)
                    if (
                        existing_identifying is not None
                        and existing_identifying != field_config.identifying
                    ):
                        raise ValueError(
                            f"Label '{field_config.label}' has conflicting "
                            "identifying flag across note types."
                        )
                    label_to_identifying[field_config.label] = field_config.identifying

                    if "choice" in field_config.name.lower():
                        has_choice_field = True
                        if field_config.identifying:
                            identifying_choice_labels.add(field_config.label)
                    elif field_config.identifying:
                        identifying_non_choice_labels.add(field_config.label)

            if config.is_choice:
                if not has_choice_field:
                    raise ValueError(
                        f"Note type '{config.name}' is marked as 'is_choice', "
                        "but no fields containing the word 'choice' were found. "
                        "Choice note types must have at least one field with "
                        "'choice' in its name."
                    )
                if not identifying_non_choice_labels:
                    raise ValueError(
                        f"Note type '{config.name}' is marked as 'is_choice', "
                        "but has no identifying non-choice field."
                    )
                if not identifying_choice_labels:
                    raise ValueError(
                        f"Note type '{config.name}' is marked as 'is_choice', "
                        "but has no identifying choice field."
                    )

            signature = frozenset(config.identifying_labels)
            existing_type = identifying_signature_to_type.get(signature)
            if existing_type is not None and existing_type != config.name:
                raise ValueError(
                    f"Note type '{config.name}' has identical identifying fields "
                    f"to '{existing_type}' ({set(signature)}), "
                    "which makes inference ambiguous."
                )
            identifying_signature_to_type[signature] = config.name


def build_label_to_field_map(configs: list[NoteType]) -> dict[str, str]:
    label_to_field: dict[str, str] = {}
    for config in configs:
        for field_config in config.fields:
            if field_config.label is None:
                continue
            existing_field = label_to_field.get(field_config.label)
            if existing_field is not None and existing_field != field_config.name:
                raise ValueError(
                    f"Label '{field_config.label}' maps to both "
                    f"'{existing_field}' and '{field_config.name}'."
                )
            label_to_field[field_config.label] = field_config.name
    return label_to_field


def note_type_signature(note_types_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Build a filesystem signature to invalidate note type cache on changes."""
    signature_parts: list[tuple[str, int, int]] = []
    if not note_types_dir.exists():
        return tuple(signature_parts)
    for path in sorted(
        note_types_dir.rglob("*"),
        key=lambda path_entry: str(path_entry),
    ):
        if path.is_file():
            stat = path.stat()
            signature_parts.append(
                (
                    str(path.relative_to(note_types_dir)),
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            )
    return tuple(signature_parts)


def load_note_types(note_types_dir: Path) -> list[NoteType]:
    if not note_types_dir.exists() or not note_types_dir.is_dir():
        raise ValueError(
            f"Note types directory not found: {note_types_dir}. "
            "Initialize or create local note_types definitions first."
        )

    cache_key = note_types_dir.resolve()
    signature = note_type_signature(note_types_dir)
    cached = _NOTE_TYPE_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    configs_dict: dict[str, NoteType] = {}
    for subdir in sorted(
        note_types_dir.iterdir(),
        key=lambda path_entry: path_entry.name,
    ):
        if not subdir.is_dir():
            continue

        config_path = subdir / "note_type.yaml"
        if not config_path.exists():
            raise ValueError(
                f"Note type directory '{subdir.name}' is missing note_type.yaml."
            )

        with open(config_path, "r", encoding="utf-8") as file:
            info = yaml.safe_load(file) or {}
        if not isinstance(info, dict):
            raise ValueError(
                f"Note type '{subdir.name}' config must be a YAML mapping."
            )

        name = subdir.name
        if "name" in info:
            raise ValueError(
                f"Note type '{name}' must not define 'name' in note_type.yaml. "
                "Use the directory name as the note type name."
            )

        if "styling" not in info:
            raise ValueError(
                f"Note type '{name}' is missing mandatory 'styling' key "
                "in note_type.yaml"
            )

        fields = [
            NoteField(
                field_data["name"],
                field_data["label"],
                identifying=field_data["identifying"],
            )
            for field_data in info.get("fields", [])
        ]

        if ANKIOPS_KEY_FIELD.name not in [field_config.name for field_config in fields]:
            fields.append(ANKIOPS_KEY_FIELD)

        css = _load_styling(subdir, name, info.get("styling"))
        templates = _load_templates(subdir, name, info)

        configs_dict[name] = NoteType(
            name=name,
            fields=fields,
            css=css,
            is_cloze=bool(info.get("is_cloze", False)),
            is_choice=bool(info.get("is_choice", False)),
            templates=templates,
        )

    configs = list(configs_dict.values())
    if not configs:
        raise ValueError(
            f"No note type definitions found in {note_types_dir}. "
            "Add at least one note type directory with note_type.yaml."
        )
    NoteType.validate_configs(configs)
    _NOTE_TYPE_CACHE[cache_key] = (signature, configs)
    return configs


def _load_styling(subdir: Path, name: str, styling_input: Any) -> str:
    if isinstance(styling_input, str):
        styling_files = [styling_input]
    elif isinstance(styling_input, list):
        styling_files = []
        for css_file in styling_input:
            if not isinstance(css_file, str) or not css_file.strip():
                raise ValueError(
                    f"Note type '{name}' has invalid styling entry "
                    f"'{css_file}'. Expected non-empty file names."
                )
            styling_files.append(css_file.strip())
    else:
        raise ValueError(
            f"Note type '{name}' has invalid 'styling' value. "
            "Expected a string or list of strings."
        )
    if not styling_files:
        raise ValueError(
            f"Note type '{name}' must reference at least one styling file."
        )

    css_parts = []
    for css_file in styling_files:
        css_path = subdir / css_file
        if not css_path.exists() or not css_path.is_file():
            raise ValueError(
                f"Note type '{name}' references missing styling file '{css_file}'."
            )
        css_parts.append(css_path.read_text(encoding="utf-8"))
    return "\n\n".join(css_parts)


def _load_templates(
    subdir: Path,
    name: str,
    info: dict[str, Any],
) -> list[dict[str, str]]:
    raw_templates = info.get("templates")
    templates: list[dict[str, str]] = []

    if raw_templates is None:
        front = subdir / "Front.template.anki"
        back = subdir / "Back.template.anki"
        if not front.exists() or not back.exists():
            missing = []
            if not front.exists():
                missing.append("Front.template.anki")
            if not back.exists():
                missing.append("Back.template.anki")
            missing_text = ", ".join(missing)
            raise ValueError(
                f"Note type '{name}' is missing template file(s): {missing_text}."
            )

        name_card1 = "Cloze" if info.get("is_cloze") else "Card 1"
        templates.append(
            {
                "Name": name_card1,
                "Front": front.read_text(encoding="utf-8"),
                "Back": back.read_text(encoding="utf-8"),
            }
        )

        template_index = 2
        while True:
            front_n = subdir / f"Front{template_index}.template.anki"
            back_n = subdir / f"Back{template_index}.template.anki"
            has_front = front_n.exists()
            has_back = back_n.exists()
            if has_front and has_back:
                templates.append(
                    {
                        "Name": f"Card {template_index}",
                        "Front": front_n.read_text(encoding="utf-8"),
                        "Back": back_n.read_text(encoding="utf-8"),
                    }
                )
                template_index += 1
            elif has_front != has_back:
                missing_file = (
                    f"Back{template_index}.template.anki"
                    if has_front
                    else f"Front{template_index}.template.anki"
                )
                raise ValueError(
                    f"Note type '{name}' has incomplete template pair for "
                    f"Card {template_index}; missing '{missing_file}'."
                )
            else:
                break
        return templates

    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError(
            f"Note type '{name}' must define a non-empty 'templates' list."
        )
    for template_data in raw_templates:
        if not isinstance(template_data, dict):
            raise ValueError(
                f"Note type '{name}' has invalid template entry "
                f"'{template_data}'. Expected a mapping."
            )

        front_ref = template_data.get("front")
        back_ref = template_data.get("back")
        if not isinstance(front_ref, str) or not front_ref.strip():
            raise ValueError(f"Note type '{name}' has template with invalid 'front'.")
        if not isinstance(back_ref, str) or not back_ref.strip():
            raise ValueError(f"Note type '{name}' has template with invalid 'back'.")

        front = subdir / front_ref.strip()
        back = subdir / back_ref.strip()
        if not front.exists() or not front.is_file():
            raise ValueError(
                f"Note type '{name}' references missing template file '{front_ref}'."
            )
        if not back.exists() or not back.is_file():
            raise ValueError(
                f"Note type '{name}' references missing template file '{back_ref}'."
            )

        templates.append(
            {
                "Name": str(template_data.get("name", "Card")),
                "Front": front.read_text(encoding="utf-8"),
                "Back": back.read_text(encoding="utf-8"),
            }
        )
    return templates


def eject_default_note_types(dst_dir: Path) -> None:
    """Copy built-in note type definitions to the filesystem."""
    src_root = resources.files("ankiops.default_note_types")
    with resources.as_file(src_root) as src_path:
        shutil.copytree(
            src_path,
            dst_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__init__.py", "__pycache__", "*.pyc"),
        )
    logger.debug("Ejected built-in note types to %s", dst_dir)


def _note_types_sync_hash(configs: list[NoteType]) -> str:
    payload = []
    for config in sorted(configs, key=lambda config_item: config_item.name):
        payload.append(
            {
                "name": config.name,
                "is_cloze": config.is_cloze,
                "is_choice": config.is_choice,
                "css": config.css,
                "fields": [
                    {
                        "name": field.name,
                        "label": field.label,
                        "identifying": field.identifying,
                    }
                    for field in config.fields
                ],
                "templates": [
                    {
                        "Name": template.get("Name", ""),
                        "Front": template.get("Front", ""),
                        "Back": template.get("Back", ""),
                    }
                    for template in config.templates
                ],
            }
        )

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _note_types_names_signature(configs: list[NoteType]) -> str:
    return ",".join(sorted(config.name for config in configs))


def sync_note_types(
    anki: "Anki",
    note_types_dir: Path,
    sync_state: "SyncState | None" = None,
) -> str | None:
    """Ensure all note types defined in `note_types_dir` exist in Anki."""
    return sync_note_type_configs(
        anki,
        load_note_types(note_types_dir),
        sync_state=sync_state,
    )


def sync_note_type_configs(
    anki: "Anki",
    configs: list[NoteType],
    *,
    sync_state: "SyncState | None" = None,
) -> str | None:
    """Ensure the provided note type configs exist in Anki."""
    if not configs:
        logger.debug("No note types provided")
        return None

    existing = set(anki.fetch_note_type_names())
    to_create = [config for config in configs if config.name not in existing]
    to_update = [config for config in configs if config.name in existing]

    local_hash = _note_types_sync_hash(configs)
    names_signature = _note_types_names_signature(configs)
    cached_state = (
        sync_state.get_note_type_sync_state() if sync_state is not None else None
    )
    if (
        sync_state is not None
        and not to_create
        and cached_state == (local_hash, names_signature)
    ):
        logger.debug(
            "Note types unchanged since last successful sync; skipping model diff"
        )
        return f"{len(to_update)} up to date (cached)"

    parts = []

    if to_create:
        names = ", ".join(config.name for config in to_create)
        logger.info("Note types: %s created (%s)", len(to_create), names)
        anki.create_note_types(to_create)
        parts.append(f"{len(to_create)} created")

    if to_update:
        logger.debug("Checking %s existing note types for updates...", len(to_update))
        states = anki.fetch_note_type_states([config.name for config in to_update])
        anki.update_note_types(to_update, states)
        names = ", ".join(config.name for config in to_update)
        logger.debug("Note types: %s synced (%s)", len(to_update), names)
        parts.append(f"{len(to_update)} synced")

    if not parts:
        logger.debug("Note types: up to date")
        return None

    if sync_state is not None:
        sync_state.set_note_type_sync_state(local_hash, names_signature)

    return ", ".join(parts)
