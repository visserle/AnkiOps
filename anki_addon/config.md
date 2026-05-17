# AnkiOps add-on

Adds import (**ma**) and export (**am**) buttons to Anki's top toolbar that shell out to the `ankiops` CLI (`ankiops ma` and `ankiops am`).

## Setup

1. Install the CLI: `pipx install ankiops`
2. Initialize a collection somewhere: `cd /path/to/collection && ankiops init`
3. Set the options below.

## Options

- **`collection_dir`** — absolute path to your AnkiOps collection directory (the folder containing `.ankiops.db`). Required.
- **`ankiops_path`** — absolute path to the `ankiops` binary. Optional; leave empty to auto-discover from `PATH` and common pipx locations (`~/.local/bin`, `/opt/homebrew/bin`, `/usr/local/bin`).

