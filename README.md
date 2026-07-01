# AnkiOps

[![Tests](https://github.com/visserle/AnkiOps/actions/workflows/test.yml/badge.svg)](https://github.com/visserle/AnkiOps/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/ankiops.svg)](https://pypi.org/project/ankiops/)

AnkiOps is a bidirectional bridge between Anki and your filesystem. Each deck becomes a Markdown file so you can manage your collection from your favorite text editor.

## Features

- **User-friendly**: Edit Anki decks as highly readable Markdown files
- **Synchronization**: Two-way synchronization of notes, note types, decks and media between Anki and the filesystem
- **Customization**: Define your own note types and card templates
- **Performance**: AnkiOps can sync thousands of notes in under a second
- **Collaboration**: Share decks on GitHub and collaborate with others

## Basic Concept

There is a joke in software engineering that eventually every filesystem grows into a database, and every database grows into a filesystem, as both approaches have their merits. AnkiOps enables the filesystem solution for Anki. It mirrors your whole collection in a folder of your choice. Each deck becomes a a Markdown file, and media and note type definitions are stored along with it. Having your collection represented in the filesystem easily allows for:

- version control via automated Git commits,
- automation of various tasks using scripts or LLMs, and
- collaboration with reviewable pull requests. 

The basic structure of an AnkiOps collection folder is:

```text
media/                 
note_types/          
.ankiops.db         
Deck1.md         
Deck1__Subdeck1.md      
```

The file name maps to the Anki deck name, where `__` becomes the subdeck delimiter `::` in Anki. Common repository files such as `README.md` are ignored.

### Markdown Files

A deck file contains one note after another. Each note is separated by a blank line, three dashes, and another blank line.

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
```

You can use any Markdown syntax (except the horizontal rule) in the note content, including italics, bold font, lists, tables, images, code blocks, math equations, and so on. AnkiOps automatically converts Markdown to HTML for Anki and back again.

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

The `note_key` is a stable identifier independent of Anki's note IDs and it is used to track notes across syncs via the local `.ankiops.db` file. The `note_type` comment is solely added for the user's reference. Neither comment should be edited by hand (in contrast to the tags comment, which is user-editable and synced with Anki).

### Note Types

AnkiOps automatically infers the note type for each note by a set of identifying field labels (e.g. `Q:` for Question, `A:` for Answer). These labels are defined in `note_type.yaml` for each note type. Using the default, a note with `Q:` and `A:` labels is an `AnkiOpsQA` note type. `note_type.yaml` defines field names, field labels, card templates, and which labels identify the note type.

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

| Note type | Identifying labels | Description |
| --- | --- | --- |
| `AnkiOpsQA` | `Q:`, `A:` | Question and Answer |
| `AnkiOpsCloze` | `T:` | Cloze deletion |
| `AnkiOpsClozeHideAll` | `THA:` | Cloze deletion with all fields hidden |
| `AnkiOpsReversed` | `F:`, `B:` | Reversed (both directions) |
| `AnkiOpsInput` | `Q:`, `I:` | Input (user-provided text) |
| `AnkiOpsChoice` | `Q:`, choice labels such as `C1:`, plus `A:` | Single/multiple choice with rotating order of choices |
| `AnkiOpsImageOcclusion` | `IO_*:` labels for image occlusion fields | For comprehensiveness (awkward to manage in Markdown) |

Every Anki note managed by AnkiOps has an additional field called `AnkiOps Key` that stores the `note_key` value and should be left unchanged. Generic, non-identifying labels such as `E:` for Extra can be added to any note type. To see the assigned labels in your collection, run `ankiops note-types`, or look up the note type definitions in `note_types/` manually.

> [!NOTE]
> AnkiOps only acts on note types defined within the `note_types/` folder.

### Synchronization

AnkiOps has two main sync commands that run an automatic Git snapshot before writing changes:

| Command | Direction | Use it when |
| --- | --- | --- |
| `ankiops fa` | Files to Anki | After editing Markdown decks, media, or note types. |
| `ankiops af` | Anki to files | After editing notes, tags, or decks in Anki. |

`ankiops fa` pushes media, syncs note types, parses Markdown, creates new notes, updates existing notes, moves cards between decks, and deletes managed Anki notes that you removed from files.

`ankiops af` reads managed notes from Anki, converts their HTML back to Markdown, writes deck files, preserves keyless local notes, and pulls referenced media into `media/`.

Both commands create a automatic Git snapshot before syncing.

### Add-On

The AnkiOps add-on provides toolbar buttons for `af` and `fa`, and it enables AnkiOpsConnect, which AnkiOps uses for operations that AnkiConnect cannot perform. If you do not want to install the add-on, you can still use AnkiOps with AnkiConnect (though AnkiOpsConnect is much faster).

![alt text](toolbar.png)

The add-on is still experimental; to install it, download the folder and put it in your Anki add-ons directory.

## How To Get Started

1. Install AnkiOps with [pipx](https://github.com/pypa/pipx) or via [uv](https://github.com/astral-sh/uv). This will make the `ankiops` command available in your shell.

```bash
pipx install ankiops
uv tool install ankiops
```

2. **Initialize AnkiOps**: Make sure that Anki is running, with Anki(Ops)Connect enabled. Run `ankiops init` in an empty collection directory. AnkiOps creates a Git repository, `.ankiops.db`, the `note_types/` folder, and recommended configurations for VS code. The tutorial flag further creates a sample Markdown deck.

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

### How is this different from other Markdown tools?

Most Markdown-to-Anki tools import one way: you write Markdown and push it to Anki. AnkiOps lets you edit in either place and sync back. It stores each deck as one Markdown file, so you browse decks instead of hundreds of per-card files. It also keeps custom note type definitions beside your decks, which lets you edit both card content and card structure from your editor.

### Is it safe to use?

Yes, AnkiOps will never modify notes that are not defined within the `note_types/` folder. Your existing collection won't be affected and you can safely mix managed and unmanaged notes within one deck. Further, AnkiOps only syncs if the activated profiles matches the one it was initialized with. Concerning the Markdown files, AnkiOps automatically creates a Git commit of your collection folder before every sync, so you can always roll your files back if needed.

### What is the recommended workflow?

We recommend VS Code because it has a stable Markdown previewer and handles images with the dedicated `media/` folder perfectly.

### How can I make use of LLMs with AnkiOps?

If you want it simple, just prompt an LLM (Codex, Claude Code, etc.) to edit your Markdown files. If you want it integrated, AnkiOps has a dedicated LLM module that runs through your serialized (JSON) collection and applies edits to the Markdown files. Only works with OpenAI's response API.

### What features are there else?


**Image Width Normalization**

Handling multiple images in a singleAnki note often leads to inconsistent widths. AnkiOps can normalize close widths or force a width across a deck:

```bash
ankiops fix-image-widths  # normalize widths within 10px tolerance
ankiops fix-image-widths --deck "Deckt1" --width 500  # fix width
```

**JSON serialization**

Serialize a collection (or one deck tree) to portable JSON and deserialize it back using the `ankiops serialize` and `ankiops deserialize` commands. 

## How does sharing work? (experimental)

Shared decks are experimental. They use Git subtree operations and GitHub repositories.

Create a shared source from a local deck tree:

```bash
ankiops shared create "Psychology" owner/psychology-deck
```

`shared create` requires a Git-backed collection, a clean Git index, and `note_key` metadata on every selected note. It copies the selected deck files, referenced media, and used note types into `shared/<owner>/<repo>/`, scopes the note types, commits the move, and pushes the subtree to GitHub. New GitHub repositories are private unless you pass `--public`.

Add and update a shared source:

```bash
ankiops shared add owner/psychology-deck
ankiops shared update owner/psychology-deck --to-anki
```

Submit local shared edits:

```bash
ankiops shared submit owner/psychology-deck --from-anki \
  --message "Clarify attention terminology"
```

The optional message becomes the Git commit subject and pull request title. When
you omit it, AnkiOps uses `Update shared deck owner/psychology-deck`. Submitting
an unchanged source creates no branch or pull request. With the GitHub CLI
available, AnkiOps opens a pull request; otherwise it pushes the branch and tells
you how to open one manually.

Shared-source Git history created by older experimental AnkiOps versions is not
compatible with this workflow. Recreate or re-add those sources before updating
or submitting them.

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

- `ankiops shared create <deck> <owner>/<repo> [--public|--private]`
- `ankiops shared add <owner>/<repo>`
- `ankiops shared update [owner/repo] [--to-anki]`
- `ankiops shared submit <owner>/<repo> [--from-anki] [-m|--message text]`
- `ankiops shared list`

## Contributing

Bug fixes, documentation improvements, tests, and PRs are welcome!

## License

MIT
