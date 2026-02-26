# AnkiOps

[![Tests](https://github.com/visserle/AnkiOps/actions/workflows/test.yml/badge.svg)](https://github.com/visserle/AnkiOps/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/ankiops.svg)](https://pypi.org/project/ankiops/) 

**Anki decks ↔ Markdown files, in perfect sync**

Editing flashcards in Anki's UI is tedious when you could be using your favorite text editor, AI tools, and Git. **AnkiOps** is a bi-directional Anki ↔ Markdown bridge. Each deck becomes a Markdown file. Work in either Anki or your text editor, and let changes flow both ways. This brings AI assistance, batch editing, and version control to your flashcards.

## Features

- Simple CLI interface: after initialization, only two commands are needed for daily use
- Fully round-trip, bi-directional sync that handles note creation, deletion, movements across decks, and conflicts
- Markdown rendering with nearly all features (including syntax-highlighted code blocks, supported on desktop and mobile)
- Support for all standard note types, plus Single & Multiple Choice
- Embed images via VS Code where they are directly copied into your Anki media folder (automatically set up)
- Built-in Git integration with autocommit for tracking all changes
- High-performance processing: handles thousands of cards across hundreds of decks in mere seconds
- Thoroughly tested, bi-directional conversion between Markdown and Anki-compatible HTML
- Serialize/deserialize entire collections to JSON format for backup, sharing, or automated AI processing

> [!NOTE]
> AnkiOps syncs notes using built-in AnkiOps note types (`AnkiOpsQA`, `AnkiOpsReversed`, `AnkiOpsCloze`, `AnkiOpsClozeHideAll`, `AnkiOpsInput`, and `AnkiOpsChoice`) plus local custom note types defined under `note_types/`.

## Getting Started


1. **Install AnkiOps via [pipx](https://github.com/pypa/pipx)**: Pipx will make AnkiOps globally available in your terminal.
```bash
pipx install ankiops
```
2. **Initialize AnkiOps**: Make sure that Anki is running, with the [AnkiConnect add-on](https://ankiweb.net/shared/info/2055492159) enabled. Initialize AnkiOps in any empty directory of your choosing. This is where your text-based decks will live. The additional tutorial flag creates a sample Markdown deck.
```bash
ankiops init --tutorial
```
3. **Execute AnkiOps**: Import the tutorial deck into Anki using:
```bash
ankiops ma # markdown to anki (import)
```
4. **Keep everything in sync**: When editing your Markdown files, sync Markdown → Anki (and vice versa), as each sync makes one side match the other. After reviewing and editing your cards in Anki, you can sync Anki → Markdown using the following command:
```bash
ankiops am # anki to markdown (export)
```

## FAQ

### How is this different from other Markdown or Obsidian tools?

Most available tools are one-way importers: you write in Markdown or Obsidian and push to Anki, but edits in Anki don't sync back. AnkiOps is bi-directional: you can edit in either Anki or Markdown and sync in both directions. Additionally, AnkiOps uses a one-file-per-deck structure, making your collection easier to navigate and manage than approaches that use one file per card. This essentially lets you manage your entire Anki collection from your favorite text editor.

### Is it safe to use?

Yes, AnkiOps will never modify notes with non-AnkiOps note types. Your existing collection won't be affected and you can safely mix managed and unmanaged notes. Further, AnkiOps only syncs if the activated profiles matches the one it was initialized with. Concerning your Markdown files, AnkiOps automatically creates a Git commit of your collection folder before every sync, so you can always roll your files back if needed.

### How do I create new notes?

Create a new Markdown file in your initialized AnkiOps folder. For the first import, the file name acts as the deck name. Subdecks use `__` (for example, `Biology::Cell` -> `Biology__Cell.md`). If a deck name contains a literal `__`, AnkiOps escapes it in the filename so mapping stays reversible. Notes must be separated by a new line, three dashes `---`, and another new line. You can add new notes anywhere in an existing file.

```markdown
<!-- note_key: 123487556abc -->
Q: Question text here
A: Answer text here
E: Extra information (optional)
M: Content behind a "more" button (optional)

---

<!-- note_key: 123474567def -->
T: Text with {{c1::multiple}} {{c2::cloze deletions}}.
E: ![image with set width](im.png){width=700}

---

Q: What is this?
C1: A multiple choice note
C2: with
C3: automatically randomized answers.
A: 1,3

```

In this example, the last note is a new note which will get a `note_key` comment assigned on the next import.

### How are the different note types identified?

Each note type is identified by its field prefixes. `E:` (Extra) and `M:` (More, revealed on click) are optional fields shared across all note types.

| Note Type | Fields |
|---|---|
| **AnkiOpsQA** | `Q:`, `A:` |
| **AnkiOpsReversed** | `F:`, `B:` |
| **AnkiOpsCloze** | `T:` |
| **AnkiOpsInput** | `Q:`, `I:` |
| **AnkiOpsChoice** | `Q:`, `C1:``C8:`, `A:` |

### Which characters or symbols cannot be used?

Since notes are separated by horizontal lines (`---`), they cannot be used within the content fields of your notes. This includes all special Markdown characters that render these lines (`***`, `___`), and `<hr>`.

### How does it work?

On first import, AnkiOps assigns a stable `note_key` to each managed note. It is represented by a single-line HTML tag (e.g., `<!-- note_key: a1b2c3d4e5f6 -->`) above a note in the Markdown. With note keys in place, we can track what is new, changed, moved between decks, or deleted, and AnkiOps syncs accordingly. Content is automatically converted between Anki's HTML format and Markdown during sync operations. One AnkiOps folder represents one Anki profile.

### What is the recommended workflow?

We recommend using VS Code. It has excellent AI integration, a great [add-on](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced) for Markdown previews, and supports image pasting (which will be saved in your Anki media folder by default).

### How can I share my AnkiOps collection?

Use `ankiops serialize` to export your local AnkiOps collection to JSON. Recipients can import it with `ankiops deserialize --input <path>` into an initialized collection folder.

Alternatively, share your collection via native Anki export (`.apkg`) or by sharing Markdown files with your local `media/` folder.

### How can I migrate my existing notes into AnkiOps?

For standard note types, migration is straightforward:

1. Convert your existing notes to the matching AnkiOps note types via `Change Note Type…` in the Anki browser.
2. Export your notes from Anki to Markdown using `ankiops am`.
3. In the first re-import, some formatting may change because the original HTML from Anki may not follow the CommonMark standard. Formatting of your cards can be done automatically at a low cost using the included JSON serializer and AI tooling.

If your existing note format doesn't map cleanly to the AnkiOps format (e.g., notes with additional or custom fields), you'll need to adapt the code accordingly. This should be fairly simple for most cases: define your note type with unique field prefixes (and unique field names within that note type) for automatic note type detection, and add your card's templates to `ankiops/card_templates`.

### How can I develop AnkiOps locally?

Fork this repository and initialize the tutorial in your root folder (make sure Anki is running). This will create a folder called `collection` with the sample Markdown in it. Paths will adapt automatically to the development environment. You can run AnkiOps locally using the main script.

```bash
git clone https://github.com/visserle/ankiops.git
cd ankiops
uv sync
uv run python -m main init --tutorial
uv run python -m main ma
```

### What commands and flags are available in the CLI?

**Global:**
- `--debug` - Enable debug logging
- `--help` - Show help message

**`init`:**
- `--tutorial` - Create tutorial markdown file

**`anki-to-markdown` / `am`:**
- `--no-auto-commit`, `-n` - Skip automatic git commit

**`markdown-to-anki` / `ma`:**
- `--no-auto-commit`, `-n` - Skip automatic git commit

**`serialize`:**
- `--output`, `-o` - Output file path (default: `<collection-name>.json`)

**`deserialize`:**
- `--input`, `-i` - Input file path (default: `<collection-name>.json`)
- `--overwrite` - Overwrite existing markdown files

**`ai config`:**
- `--profile` - Model profile id from `ai/models/*.yaml`
- `--provider {groq,ollama,openai}` - Optional runtime provider override
- `--model` - Optional runtime model override
- `--base-url` - Optional runtime OpenAI-compatible base URL override
- `--api-key-env` - Optional runtime API key env var override
- `--api-key` - Optional runtime API key value
- `--timeout` - Optional runtime timeout override
- `--max-in-flight` - Optional runtime max concurrent request override

**`ai`:**
- `--include-deck`, `-d` - Include a deck and all subdecks recursively (repeatable)
- `--task` - Task file name/path from `ai/tasks/` (required)
- `--batch-size` - Override task batch size
- `--temperature` - Override task temperature (`0` to `2`)
- `--progress {auto,on,off}` - Show periodic AI task progress logs
- `--profile`, `--provider`, `--model`, `--base-url`, `--api-key-env`, `--api-key`, `--timeout`, `--max-in-flight` - Runtime overrides

### Where is AI config stored?

AnkiOps stores model profiles in `ai/models/*.yaml`.  
Built-in profiles are `ollama-fast`, `openai-fast`, and `groq-fast`.
API credentials are provider-level defaults (for example `OPENAI_API_KEY` for `openai`, `GROQ_API_KEY` for `groq`) and can be overridden at runtime via `--api-key-env`/`--api-key`.

### Task Infrastructure

AnkiOps initializes a local `ai/` folder and copies built-in model and task YAML files to:
- `ai/models/*.yaml`
- `ai/tasks/*.yaml`

Task YAML files define:
- Task instructions
- Optional model profile (`model`); omit it or set `model: default` to use the default profile
- Batch mode/size
- Deck and note-type scope
- `read_fields` (inline JSON context)
- `write_fields` (fields the AI may modify)
- Optional `temperature` (`0` to `2`, default `0.0`)

Inline editing always includes `note_key`, and batch responses are keyed by `note_key`.
