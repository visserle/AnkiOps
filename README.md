# AnkiOps

[![Tests](https://github.com/visserle/AnkiOps/actions/workflows/test.yml/badge.svg)](https://github.com/visserle/AnkiOps/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/ankiops.svg)](https://pypi.org/project/ankiops/)

AnkiOps is a bidirectional bridge between Anki and your filesystem. Each deck becomes a Markdown file so you can manage your collection from your favorite text editor. Edit in plain text, version with Git, enhance with LLMs, and sync changes both ways.

## Advantages

- **User-friendly**: Edit Anki decks as highly readable Markdown files
- **Full Anki support**: Two-way sync of notes, note types, decks and media files
- **Customization**: Define your own note types and card templates
- **Performance**: Sync thousands of notes in under a second
- **Collaboration**: Share decks on GitHub and collaborate with others

## How It Works

### Markdown Files

In a deck file, each note is separated by a blank line, three dashes, and another blank line:

```markdown
Q: How are the 86 billion neurons distributed across the human brain?
A: - Cerebellum: 80% (69 billion neurons)
- Cerebral Cortex: 19% (16 billion neurons)
- Subcortical Areas and Brainstem: 1% (<1 billion neurons)
E: --> The cerebellum contains the majority, despite being only about 10% of brain mass.

![human brain](media/human-brain.png){width=500}

---

<!-- tags: psychology brain -->
T: The {{c1::corpus callosum}} connects the left and right cerebral hemispheres.
E: Cutting it can stop information from passing directly between the hemispheres.

---

...
```

You can use any Markdown syntax (except a horizontal rule) in the note content, including italics, bold text, lists, tables, images, code blocks, math equations, and more. AnkiOps automatically converts Markdown to HTML for Anki and back again.

After the first sync, AnkiOps adds metadata comments for each note:

```markdown
<!-- note_key: 1a2b3c4d5e6f -->
<!-- note_type: AnkiOpsQA -->
Q: How are the 86 billion neurons distributed across the human brain?
A: - Cerebellum: 80% (69 billion neurons)
- Cerebral Cortex: 19% (16 billion neurons)
- Subcortical Areas and Brainstem: 1% (<1 billion neurons)
E: --> The cerebellum contains the majority, despite being only about 10% of brain mass.

![human brain](media/human-brain.png){width=500}

---

<!-- note_key: 8f2c1a7b9d0e -->
<!-- note_type: AnkiOpsCloze -->
<!-- tags: psychology brain -->
T: The {{c1::corpus callosum}} connects the left and right cerebral hemispheres.
E: Cutting it can stop information from passing directly between the hemispheres.
```

The `note_key` is a stable identifier independent of Anki's note IDs and it is used to track notes across syncs. The `note_type` comment is added just for the user's reference. Neither comment should be edited by hand (in contrast to the `tags` comment, which is user-editable and synced with Anki).

### Collection Structure

The basic structure of an AnkiOps collection is:

```text
media/                 
note_types/          
.ankiops.db         
Deck1.md         
Deck1__Subdeck1.md      
```

The `.ankiops.db` file is the heart of AnkiOps. It connects the `note_key` values in the Markdown files to Anki's internal note IDs.

### Note Types

> [!NOTE]
> AnkiOps only acts on note types defined within the `note_types/` folder.

AnkiOps automatically infers the note type for each note by a set of identifying field labels (e.g. `Q:` for Question, `A:` for Answer). These labels are defined in `note_type.yaml` for each note type. By default, a note with `Q:` and `A:` labels is an `AnkiOpsQA` note type. `note_type.yaml` defines field names, field labels, card templates, and which labels identify the note type.

```text
note_types/
  AnkiOpsQA/
    Front.template.anki
    Back.template.anki
    note_type.yaml
  AnkiOpsCloze/
    Front.template.anki
    Back.template.anki
    note_type.yaml
  AnkiOpsStyling.css
  SyntaxHighlighting.css
```

Note types are fully customizable. Built-in note types include:

| Note type | Identifying labels | 
| --- | --- | 
| `AnkiOpsQA` | `Q:`, `A:` | 
| `AnkiOpsCloze` | `T:` | 
| `AnkiOpsClozeHideAll` | `THA:`  |
| `AnkiOpsReversed` | `F:`, `B:`  |
| `AnkiOpsInput` | `Q:`, `I:` |
| `AnkiOpsChoice` | `Q:`, choice labels such as `C1:`, plus `A:`  |
| `AnkiOpsImageOcclusion` | `IO_*:` labels for image occlusion fields |

Every Anki note managed by AnkiOps has an additional field called `AnkiOps Key` that stores the `note_key` value and should never be edited manually. Generic, non-identifying labels such as `E:` for Extra can be added to any note type. To see the assigned labels in your collection, run `ankiops note-types`, or look up the note type definitions in `note_types/` manually.

### Synchronization

AnkiOps has two sync commands:

- `ankiops fa` (files to anki): After editing Markdown decks, media, or note types, and
- `ankiops af` (anki to files): After editing notes, tags, or decks in Anki.

Both sync operations can create update, move, and delete managed notes, and handle all media and note types. Before syncing, an automatic Git snapshot is created.

### Add-On

For basic usage, you can use AnkiOps without the add-on. The add-on enables AnkiOpsConnect, which AnkiOps needs for operations related to sharing. If you do not want to install the add-on, you can use AnkiOps with AnkiConnect (AnkiOpsConnect is twice as fast though). 



Another feature of the add-on are the toolbar buttons for `af` and `fa`:

![alt text](toolbar.png)


To install the add-on, download the folder and put it in your Anki add-ons directory.

## How To Get Started

1. Install AnkiOps with [pipx](https://github.com/pypa/pipx) or via [uv](https://github.com/astral-sh/uv). This will make the `ankiops` command available in your shell.

```bash
pipx install ankiops
# or
uv tool install ankiops
```

2. **Initialize AnkiOps**: Make sure that Anki is running, with AnkiOpsConnect enabled. Run `ankiops init` in an empty collection directory. AnkiOps creates a Git repository, `.ankiops.db`, the `note_types/` folder, and recommended configurations for VS Code. The tutorial flag further creates a sample Markdown deck.

```bash
cd my-collection
ankiops init --tutorial
```

3. **Sync files to Anki**: Apply the current files, including Markdown decks, media, and note types, to Anki.

```bash
ankiops fa
```

4. **Sync Anki back to files**: After you review or edit cards in Anki, apply those changes to your local files. Each sync makes one side match the other. Inspect the Git diff to easily track all changes.

```bash
ankiops af
```

## FAQ

### Why should I use AnkiOps?

There is a joke among software engineers that eventually every filesystem grows into a database, and every database grows into a filesystem. Both approaches have their merits. AnkiOps enables the filesystem solution for the Anki database. It mirrors a full collection in a folder of your choice, where each deck becomes a Markdown file, and media and note-type definitions are stored along with it. Having your collection represented in the filesystem enables straightforward version control, automation, and collaboration.

### How is this different from other Markdown tools?

Most Markdown-to-Anki tools import one way: you write Markdown and push it to Anki. AnkiOps lets you edit in either place and sync back. It stores each deck as one Markdown file, so you browse decks instead of hundreds of per-card files. It also keeps custom note type definitions beside your decks, which lets you edit both card content and card structure from your editor.

### Is it safe to use?

Yes, AnkiOps will never modify notes that are not defined within the `note_types/` folder. Your existing collection won't be affected, and you can safely mix managed and unmanaged notes within one deck. Additionally, AnkiOps only syncs if the activated profile matches the one it was initialized with. For Markdown files, AnkiOps automatically creates a Git commit of your collection folder before every sync, so you can always revert changes if needed.

### What is the recommended workflow?

We recommend VS Code because it has a stable Markdown previewer and handles images with the dedicated `media/` folder perfectly.

### How can I make use of LLMs with AnkiOps?

If you want it simple, just prompt an LLM (Codex, Claude Code, etc.) to edit your Markdown files. If you want it integrated, AnkiOps has a dedicated LLM module that runs through your serialized (JSON) collection and applies edits to the Markdown files. Only works with OpenAI's response API.

### What other features are there?


**Image Width Normalization**

Handling multiple images in a single Anki note often leads to inconsistent widths. AnkiOps can normalize similar widths or force a width across a deck:

```bash
ankiops fix-image-widths  # normalize widths within 10px tolerance
ankiops fix-image-widths --deck "Deck1" --width 500  # fix width
```

**JSON serialization**

Serialize a collection (or one deck tree) to portable JSON and deserialize it back using the `ankiops serialize` and `ankiops deserialize` commands. 

## How does sharing work? (experimental)

Shared decks are ordinary GitHub repositories cloned inside the collection. The
collection root remains one VS Code workspace, while VS Code shows each shared
source as its own repository.

Publish a local deck tree:

```bash
ankiops shared publish "Psychology" owner/psychology-deck
```

`shared publish` requires stable `note_key` metadata and authenticated GitHub CLI.
It moves the selected deck tree into `shared/<owner>/<repo>/`, copies referenced
media and note types, creates an independent repository, and pushes it to GitHub.
New repositories are private unless you pass `--public`.

Subscribe to and update a shared deck:

```bash
ankiops shared subscribe owner/psychology-deck
ankiops shared update owner/psychology-deck
ankiops fa
```

Submit local shared edits:

```bash
ankiops shared status owner/psychology-deck
ankiops shared submit owner/psychology-deck \
  --message "Clarify attention terminology"
```

`shared update` changes files only; run `ankiops fa` after reviewing them.
`shared submit` commits changes only inside the selected source and opens a pull
request. Contributors without write permission use an authenticated fork
automatically. If local and upstream edits overlap, the subscribed deck remains
unchanged. AnkiOps preserves editable base, local, and upstream copies; edit the
marked Markdown it reports and rerun `shared update`.

This architecture supports fresh collections only. Older databases and
subtree-based collections are rejected without migration or automatic recovery.

## Command Reference

Global options:

- `--debug`: Enable debug logging.
- `--version`: Show the installed version.
- `--help`: Show help.

Core commands:

- `ankiops init [--tutorial]`: Initialize the current directory as an AnkiOps collection.
- `ankiops anki-to-files` or `ankiops af`: Sync Anki changes into files.
- `ankiops files-to-anki` or `ankiops fa`: Sync file changes into Anki.
- `ankiops note-types`: Show note types and the label registry.
- `ankiops note-types --add <name>`: Copy a note type from Anki into `note_types/`.

Sync flags:

- `--no-auto-commit`, `-n`: Skip the pre-sync Git snapshot for `af`, `fa`, `deserialize`, `fix-image-widths`, and `llm <task> --run`.

Serialization:

- `ankiops serialize [-o path] [--deck name] [--no-subdecks]`
- `ankiops deserialize [-i path] [--overwrite]`

Image tools:

- `ankiops fix-image-widths [--deck name] [--no-subdecks] [--tolerance px] [--width px]`

LLM tools:

- `ankiops llm`: Show configured tasks and recent jobs.
- `ankiops llm <task> [--model model] [--deck deck]`: Show a dry-run plan.
- `ankiops llm <task> --run [--model model] [--deck deck]`: Run a task.
- `ankiops llm --job <id|latest>`: Inspect one job.

Shared deck tools:

- `ankiops shared publish <deck> <owner>/<repo> [--public|--private]`
- `ankiops shared subscribe <owner>/<repo>`
- `ankiops shared status [owner/repo]`
- `ankiops shared update [owner/repo]`
- `ankiops shared submit <owner>/<repo> [-m|--message text]`

## Contributing

Bug fixes, documentation improvements, tests, and PRs are welcome!

## License

MIT
