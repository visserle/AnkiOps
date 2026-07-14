DB_SCHEMA: dict[str, str] = {
    "note_state": """
CREATE TABLE note_state (
    note_key TEXT PRIMARY KEY,
    note_id INTEGER NOT NULL UNIQUE,
    source_path TEXT NOT NULL DEFAULT '.',
    import_md_hash TEXT,
    import_anki_hash TEXT,
    export_md_hash TEXT,
    export_anki_hash TEXT,
    CHECK (source_path <> ''),
    CHECK (
        (import_md_hash IS NULL) = (import_anki_hash IS NULL)
    ),
    CHECK (
        (export_md_hash IS NULL) = (export_anki_hash IS NULL)
    )
)
""",
    "deck_map": """
CREATE TABLE deck_map (
    deck_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL DEFAULT '.',
    md_path TEXT,
    CHECK (source_path <> '')
)
""",
    "app_state": """
CREATE TABLE app_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    profile_name TEXT CHECK (profile_name IS NULL OR profile_name <> ''),
    note_type_sync_hash TEXT CHECK (
        note_type_sync_hash IS NULL OR note_type_sync_hash <> ''
    ),
    note_type_names_signature TEXT CHECK (
        note_type_names_signature IS NULL OR note_type_names_signature <> ''
    ),
    CHECK (
        (note_type_sync_hash IS NULL) = (note_type_names_signature IS NULL)
    )
)
""",
    "markdown_media_cache": """
CREATE TABLE markdown_media_cache (
    md_path TEXT PRIMARY KEY,
    md_mtime_ns INTEGER NOT NULL CHECK (md_mtime_ns >= 0),
    md_size INTEGER NOT NULL CHECK (md_size >= 0),
    media_names_json TEXT NOT NULL
)
""",
    "media_files": """
CREATE TABLE media_files (
    source_path TEXT NOT NULL DEFAULT '.',
    name TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL CHECK (mtime_ns >= 0),
    size INTEGER NOT NULL CHECK (size >= 0),
    digest TEXT NOT NULL CHECK (digest <> ''),
    hashed_name TEXT NOT NULL CHECK (hashed_name <> ''),
    pushed_digest TEXT CHECK (pushed_digest IS NULL OR pushed_digest <> ''),
    PRIMARY KEY (source_path, name),
    CHECK (source_path <> '')
)
""",
}
